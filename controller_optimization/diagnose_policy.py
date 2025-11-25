#!/usr/bin/env python3
"""
Diagnostic script to analyze why PolicyGenerator is input-insensitive.

Checks:
1. Tanh saturation (pre-tanh values)
2. UncertaintyPredictor variance (are inputs actually different?)
3. PolicyGenerator input sensitivity
4. Weight statistics
"""

import argparse
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate
from controller_optimization.src.utils.target_generation import generate_target_trajectory
from controller_optimization.configs.processes_config import PROCESSES


def diagnose_policy(checkpoint_dir: str):
    """Run diagnostics on a trained policy."""
    import os

    checkpoint_dir = Path(checkpoint_dir).resolve()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print("POLICY GENERATOR DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Device: {device}")

    # Change to controller_optimization directory so relative checkpoint paths work
    original_dir = os.getcwd()
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    # Load process chain
    print("\n[1] Loading trained model...")

    try:
        # Generate target trajectory (needed to initialize ProcessChain)
        target_trajectory = generate_target_trajectory(PROCESSES)

        # Create ProcessChain
        process_chain = ProcessChain(
            processes_config=PROCESSES,
            target_trajectory=target_trajectory,
            device=device
        )
    finally:
        os.chdir(original_dir)  # Restore original directory

    # Load trained weights - try multiple formats
    chain_path = checkpoint_dir / 'process_chain.pth'
    if chain_path.exists():
        state_dict = torch.load(chain_path, map_location=device, weights_only=False)
        process_chain.load_state_dict(state_dict)
        print(f"  Loaded weights from {chain_path}")
    else:
        # Try loading individual policy files (policy_0.pth, policy_1.pth, etc.)
        policy_files = sorted(checkpoint_dir.glob('policy_*.pth'))
        if policy_files:
            print(f"  Found {len(policy_files)} policy files")

            # Need to recreate policy generators with correct architecture from saved weights
            from controller_optimization.src.models.policy_generator import PolicyGenerator

            for i, policy_path in enumerate(policy_files):
                if i < len(process_chain.policy_generators):
                    state_dict = torch.load(policy_path, map_location=device, weights_only=False)

                    # Infer architecture from saved weights
                    hidden_sizes = []
                    layer_idx = 0
                    while f'network.{layer_idx}.weight' in state_dict:
                        weight_shape = state_dict[f'network.{layer_idx}.weight'].shape
                        hidden_sizes.append(weight_shape[0])
                        layer_idx += 3  # Each block: Linear, activation, dropout

                    input_size = state_dict['network.0.weight'].shape[1]
                    output_size = state_dict['output_head.weight'].shape[0]

                    print(f"    {policy_path.name}: input={input_size}, hidden={hidden_sizes}, output={output_size}")

                    # Get bounds from current policy generator (truncate to match output_size)
                    old_policy = process_chain.policy_generators[i]
                    output_min = old_policy.output_min[:output_size] if old_policy.output_min is not None else None
                    output_max = old_policy.output_max[:output_size] if old_policy.output_max is not None else None

                    # Create new policy with correct architecture
                    new_policy = PolicyGenerator(
                        input_size=input_size,
                        hidden_sizes=hidden_sizes,
                        output_size=output_size,
                        output_min=output_min,
                        output_max=output_max,
                        dropout_rate=0.1,
                        use_batchnorm=False
                    ).to(device)

                    # Load with strict=False to allow missing output_min/output_max buffers
                    new_policy.load_state_dict(state_dict, strict=False)
                    # Set bounds after loading using register_buffer to ensure proper registration
                    if output_min is not None:
                        new_policy.register_buffer('output_min', output_min.clone())
                    if output_max is not None:
                        new_policy.register_buffer('output_max', output_max.clone())
                    process_chain.policy_generators[i] = new_policy
                    print(f"    Loaded {policy_path.name} -> policy_generator[{i}] (bounds: {output_min.shape if output_min is not None else None})")
        else:
            print(f"  ERROR: No checkpoint found at {checkpoint_dir}")
            print(f"  Expected: process_chain.pth or policy_*.pth files")
            return

    process_chain.eval()

    # =========================================================================
    # DIAGNOSTIC 1: Check UncertaintyPredictor variance
    # =========================================================================
    print(f"\n{'='*70}")
    print("[2] UNCERTAINTY PREDICTOR VARIANCE")
    print("    (Are the sampled outputs actually different across seeds?)")
    print(f"{'='*70}")

    seeds = [42, 123, 456]

    with torch.no_grad():
        for i, process_name in enumerate(process_chain.process_names):
            print(f"\n  Process: {process_name}")

            if i == 0:
                # First process - fixed inputs
                inputs = process_chain.get_initial_inputs(batch_size=1, scenario_idx=0)
                inputs_scaled = process_chain.scale_inputs(inputs, i)
                outputs_mean, outputs_var = process_chain.uncertainty_predictors[i](inputs_scaled)
                outputs_mean = process_chain.unscale_outputs(outputs_mean, i)
                outputs_var = process_chain.unscale_variance(outputs_var, i)

                std = torch.sqrt(outputs_var + 1e-8)

                print(f"    Outputs mean: {outputs_mean[0].tolist()}")
                print(f"    Outputs std:  {std[0].tolist()}")
                print(f"    Coefficient of variation: {(std / (outputs_mean.abs() + 1e-8))[0].tolist()}")

                # Sample with different seeds
                sampled_outputs = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    epsilon = torch.randn_like(outputs_mean)
                    sampled = outputs_mean + epsilon * std
                    sampled_outputs.append(sampled[0].numpy())

                sampled_outputs = np.array(sampled_outputs)
                print(f"\n    Sampled outputs for seeds {seeds}:")
                for j, seed in enumerate(seeds):
                    print(f"      Seed {seed}: {sampled_outputs[j].tolist()}")
                print(f"    Std across seeds: {sampled_outputs.std(axis=0).tolist()}")

    # =========================================================================
    # DIAGNOSTIC 2: Check PolicyGenerator input sensitivity
    # =========================================================================
    print(f"\n{'='*70}")
    print("[3] POLICY GENERATOR INPUT SENSITIVITY")
    print("    (Does the policy output change when input changes?)")
    print(f"{'='*70}")

    with torch.no_grad():
        for i, policy in enumerate(process_chain.policy_generators):
            prev_process = process_chain.process_names[i]
            next_process = process_chain.process_names[i + 1]
            print(f"\n  Policy: {prev_process} -> {next_process}")

            # Get the expected input size from the policy's first layer
            expected_input_size = policy.network[0].weight.shape[1]
            print(f"    Expected input size: {expected_input_size}")

            # Create a synthetic input with realistic values
            # Use outputs from the uncertainty predictor as reference
            inputs = process_chain.get_initial_inputs(batch_size=1, scenario_idx=0)
            inputs_scaled = process_chain.scale_inputs(inputs, i)
            outputs_mean, outputs_var = process_chain.uncertainty_predictors[i](inputs_scaled)
            outputs_mean = process_chain.unscale_outputs(outputs_mean, i)
            outputs_var = process_chain.unscale_variance(outputs_var, i)

            # Build policy input with correct size
            # The policy was trained with [outputs_mean, outputs_var] (no scenario encoder)
            policy_input = torch.cat([outputs_mean, outputs_var], dim=1)

            # Pad or truncate to match expected size
            if policy_input.shape[1] < expected_input_size:
                padding = torch.zeros(1, expected_input_size - policy_input.shape[1], device=device)
                policy_input = torch.cat([policy_input, padding], dim=1)
            elif policy_input.shape[1] > expected_input_size:
                policy_input = policy_input[:, :expected_input_size]

            print(f"    Policy input shape: {policy_input.shape}")
            print(f"    Policy input: {policy_input[0].tolist()}")

            # Get baseline output
            baseline_output = policy(policy_input)
            print(f"\n    Baseline output: {baseline_output[0].tolist()}")

            # Perturb input and see if output changes
            print(f"\n    Perturbation test (±10% of input range):")
            input_range = max(policy_input.max() - policy_input.min(), 1.0)
            perturbations = [-0.1, -0.01, 0.01, 0.1]

            for pert in perturbations:
                perturbed_input = policy_input + pert * input_range
                perturbed_output = policy(perturbed_input)
                output_diff = (perturbed_output - baseline_output).abs().mean().item()
                print(f"      Perturbation {pert:+.2f}: output diff = {output_diff:.6f}")

            # Check individual input dimensions
            print(f"\n    Per-dimension sensitivity (perturb each dim by +10%):")
            for dim in range(min(5, policy_input.shape[1])):  # First 5 dims
                perturbed_input = policy_input.clone()
                perturbed_input[0, dim] *= 1.1
                perturbed_output = policy(perturbed_input)
                output_diff = (perturbed_output - baseline_output).abs().mean().item()
                print(f"      Dim {dim}: diff = {output_diff:.6f}")

    # =========================================================================
    # DIAGNOSTIC 3: Check tanh saturation
    # =========================================================================
    print(f"\n{'='*70}")
    print("[4] TANH SATURATION CHECK")
    print("    (Are pre-tanh values large, causing saturation?)")
    print(f"{'='*70}")

    with torch.no_grad():
        for i, policy in enumerate(process_chain.policy_generators):
            prev_process = process_chain.process_names[i]
            next_process = process_chain.process_names[i + 1]
            print(f"\n  Policy: {prev_process} -> {next_process}")

            # Get expected input size from policy
            expected_input_size = policy.network[0].weight.shape[1]

            # Create synthetic input
            inputs = process_chain.get_initial_inputs(batch_size=1, scenario_idx=0)
            inputs_scaled = process_chain.scale_inputs(inputs, i)
            outputs_mean, outputs_var = process_chain.uncertainty_predictors[i](inputs_scaled)
            outputs_mean = process_chain.unscale_outputs(outputs_mean, i)
            outputs_var = process_chain.unscale_variance(outputs_var, i)

            policy_input = torch.cat([outputs_mean, outputs_var], dim=1)

            # Pad or truncate to match expected size
            if policy_input.shape[1] < expected_input_size:
                padding = torch.zeros(1, expected_input_size - policy_input.shape[1], device=device)
                policy_input = torch.cat([policy_input, padding], dim=1)
            elif policy_input.shape[1] > expected_input_size:
                policy_input = policy_input[:, :expected_input_size]

            # Manually compute forward pass to inspect internals
            features = policy.network(policy_input)
            raw_actions = policy.output_head(features)
            tanh_out = torch.tanh(raw_actions)

            print(f"    Features: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
            print(f"    Raw actions (pre-tanh): {raw_actions[0].tolist()}")
            print(f"    Tanh output: {tanh_out[0].tolist()}")

            # Check saturation
            for dim in range(raw_actions.shape[1]):
                raw_val = raw_actions[0, dim].item()
                tanh_val = tanh_out[0, dim].item()
                saturation = "SATURATED" if abs(tanh_val) > 0.99 else "ok"
                print(f"      Dim {dim}: raw={raw_val:+.4f}, tanh={tanh_val:+.4f} [{saturation}]")

    # =========================================================================
    # DIAGNOSTIC 4: Weight statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print("[5] POLICY GENERATOR WEIGHT STATISTICS")
    print(f"{'='*70}")

    for i, policy in enumerate(process_chain.policy_generators):
        prev_process = process_chain.process_names[i]
        next_process = process_chain.process_names[i + 1]
        print(f"\n  Policy: {prev_process} -> {next_process}")

        for name, param in policy.named_parameters():
            print(f"    {name}: mean={param.data.mean().item():.4f}, "
                  f"std={param.data.std().item():.4f}, "
                  f"min={param.data.min().item():.4f}, "
                  f"max={param.data.max().item():.4f}")

    # =========================================================================
    # DIAGNOSTIC 5: Compare policy outputs across different inputs
    # =========================================================================
    print(f"\n{'='*70}")
    print("[6] POLICY OUTPUT VARIATION TEST")
    print("    (Do different inputs produce different outputs?)")
    print(f"{'='*70}")

    with torch.no_grad():
        for i, policy in enumerate(process_chain.policy_generators):
            prev_process = process_chain.process_names[i]
            next_process = process_chain.process_names[i + 1]
            print(f"\n  Policy: {prev_process} -> {next_process}")

            expected_input_size = policy.network[0].weight.shape[1]

            # Generate multiple different inputs and check outputs
            outputs_list = []
            for seed in seeds:
                torch.manual_seed(seed)

                # Create input with some variation based on seed
                inputs = process_chain.get_initial_inputs(batch_size=1, scenario_idx=0)
                inputs_scaled = process_chain.scale_inputs(inputs, i)
                outputs_mean, outputs_var = process_chain.uncertainty_predictors[i](inputs_scaled)
                outputs_mean = process_chain.unscale_outputs(outputs_mean, i)
                outputs_var = process_chain.unscale_variance(outputs_var, i)

                # Add noise based on variance
                std = torch.sqrt(outputs_var + 1e-8)
                epsilon = torch.randn_like(outputs_mean)
                sampled = outputs_mean + epsilon * std

                policy_input = torch.cat([sampled, outputs_var], dim=1)

                # Pad or truncate
                if policy_input.shape[1] < expected_input_size:
                    padding = torch.zeros(1, expected_input_size - policy_input.shape[1], device=device)
                    policy_input = torch.cat([policy_input, padding], dim=1)
                elif policy_input.shape[1] > expected_input_size:
                    policy_input = policy_input[:, :expected_input_size]

                output = policy(policy_input)
                outputs_list.append(output[0].numpy())
                print(f"    Seed {seed}: input={policy_input[0, :2].tolist()}, output={output[0].tolist()}")

            outputs_arr = np.array(outputs_list)
            print(f"    Output std across seeds: {outputs_arr.std(axis=0).tolist()}")
            if outputs_arr.std() < 1e-6:
                print(f"    ⚠️  WARNING: Outputs are IDENTICAL regardless of input!")

    print(f"\n{'='*70}")
    print("DIAGNOSTICS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose PolicyGenerator issues')
    parser.add_argument('checkpoint_dir', type=str,
                        help='Path to checkpoint directory (e.g., checkpoints/controller/run_001)')

    args = parser.parse_args()
    diagnose_policy(args.checkpoint_dir)
