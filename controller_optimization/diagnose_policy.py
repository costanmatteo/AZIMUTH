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
            for i, policy_path in enumerate(policy_files):
                if i < len(process_chain.policy_generators):
                    state_dict = torch.load(policy_path, map_location=device, weights_only=False)
                    process_chain.policy_generators[i].load_state_dict(state_dict)
                    print(f"    Loaded {policy_path.name} -> policy_generator[{i}]")
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

            # Get typical input to this policy
            # First run a forward pass to get realistic inputs
            torch.manual_seed(42)
            trajectory = process_chain.forward(batch_size=1, scenario_idx=0)

            # Get the actual policy input that was used
            prev_outputs = trajectory[prev_process]['outputs_sampled']
            prev_var = trajectory[prev_process]['outputs_var']

            # Build policy input
            if process_chain.use_scenario_encoder:
                structural_params = process_chain._extract_structural_params(0)
                structural_params = structural_params.unsqueeze(0)
                scenario_embedding = process_chain.scenario_encoder(structural_params)
                policy_input = torch.cat([prev_outputs, prev_var, scenario_embedding], dim=1)
            else:
                policy_input = torch.cat([prev_outputs, prev_var], dim=1)

            print(f"    Policy input shape: {policy_input.shape}")
            print(f"    Policy input: {policy_input[0].tolist()[:10]}..." if policy_input.shape[1] > 10 else f"    Policy input: {policy_input[0].tolist()}")

            # Get baseline output
            baseline_output = policy(policy_input)
            print(f"\n    Baseline output: {baseline_output[0].tolist()}")

            # Perturb input and see if output changes
            print(f"\n    Perturbation test (±10% of input range):")
            input_range = policy_input.max() - policy_input.min()
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

            # Run forward pass to get typical input
            torch.manual_seed(42)
            trajectory = process_chain.forward(batch_size=1, scenario_idx=0)

            prev_outputs = trajectory[prev_process]['outputs_sampled']
            prev_var = trajectory[prev_process]['outputs_var']

            if process_chain.use_scenario_encoder:
                structural_params = process_chain._extract_structural_params(0)
                structural_params = structural_params.unsqueeze(0)
                scenario_embedding = process_chain.scenario_encoder(structural_params)
                policy_input = torch.cat([prev_outputs, prev_var, scenario_embedding], dim=1)
            else:
                policy_input = torch.cat([prev_outputs, prev_var], dim=1)

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
    # DIAGNOSTIC 5: Compare outputs across seeds
    # =========================================================================
    print(f"\n{'='*70}")
    print("[6] OUTPUT COMPARISON ACROSS SEEDS")
    print("    (Final test: are outputs truly identical?)")
    print(f"{'='*70}")

    with torch.no_grad():
        all_outputs = {}

        for seed in seeds:
            torch.manual_seed(seed)
            trajectory = process_chain.forward(batch_size=1, scenario_idx=0)

            for process_name in process_chain.process_names:
                if process_name not in all_outputs:
                    all_outputs[process_name] = {'inputs': [], 'outputs': []}
                all_outputs[process_name]['inputs'].append(
                    trajectory[process_name]['inputs'][0].numpy()
                )
                all_outputs[process_name]['outputs'].append(
                    trajectory[process_name]['outputs_sampled'][0].numpy()
                )

        for process_name in process_chain.process_names:
            inputs = np.array(all_outputs[process_name]['inputs'])
            outputs = np.array(all_outputs[process_name]['outputs'])

            print(f"\n  Process: {process_name}")
            print(f"    Inputs across seeds:")
            for j, seed in enumerate(seeds):
                print(f"      Seed {seed}: {inputs[j].tolist()}")
            print(f"    Input std across seeds: {inputs.std(axis=0).tolist()}")

            print(f"    Outputs across seeds:")
            for j, seed in enumerate(seeds):
                print(f"      Seed {seed}: {outputs[j].tolist()}")
            print(f"    Output std across seeds: {outputs.std(axis=0).tolist()}")

    print(f"\n{'='*70}")
    print("DIAGNOSTICS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose PolicyGenerator issues')
    parser.add_argument('checkpoint_dir', type=str,
                        help='Path to checkpoint directory (e.g., checkpoints/controller/run_001)')

    args = parser.parse_args()
    diagnose_policy(args.checkpoint_dir)
