"""
Debug script to diagnose training issues.

Checks:
1. Gradient flow through the network
2. Loss scale and gradient magnitudes
3. Learning rate effects
4. Model parameter updates
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES, get_filtered_processes
from controller_optimization.configs.controller_config import CONTROLLER_CONFIG
from controller_optimization.src.utils.target_generation import generate_target_trajectory
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate


def check_gradients(process_chain, surrogate, scenario_idx=0, batch_size=32):
    """Check if gradients are flowing properly through the network."""

    print("\n" + "="*70)
    print("GRADIENT FLOW CHECK")
    print("="*70)

    # Forward pass
    process_chain.train()
    trajectory = process_chain.forward(batch_size=batch_size, scenario_idx=scenario_idx)

    # Compute reliability
    F = surrogate.compute_reliability(trajectory)

    # Compute simple loss (just reliability, no BC)
    F_star = surrogate.F_star[scenario_idx]
    F_star_tensor = torch.tensor(F_star, dtype=torch.float32, device=F.device)
    loss = (F - F_star_tensor) ** 2

    print(f"F (actual):     {F.item():.6f}")
    print(f"F* (target):    {F_star:.6f}")
    print(f"Loss:           {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients for each policy generator
    print(f"\nPolicy Generator Gradients:")
    for i, policy in enumerate(process_chain.policy_generators):
        total_grad_norm = 0.0
        n_params = 0
        max_grad = 0.0
        min_grad = float('inf')

        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())
                n_params += 1

        total_grad_norm = np.sqrt(total_grad_norm)

        print(f"  Policy {i}:")
        print(f"    Total grad norm: {total_grad_norm:.6e}")
        print(f"    Max grad:        {max_grad:.6e}")
        print(f"    Min grad:        {min_grad:.6e}")

        if total_grad_norm < 1e-10:
            print(f"    ⚠️  WARNING: Gradients are vanishingly small!")
        elif total_grad_norm > 100:
            print(f"    ⚠️  WARNING: Gradients are very large (potential instability)!")

    # Check scenario encoder gradients if enabled
    if hasattr(process_chain, 'scenario_encoder') and process_chain.scenario_encoder is not None:
        total_grad_norm = 0.0
        for param in process_chain.scenario_encoder.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = np.sqrt(total_grad_norm)
        print(f"\n  Scenario Encoder:")
        print(f"    Total grad norm: {total_grad_norm:.6e}")


def check_learning_rate_effect(process_chain, surrogate, scenario_idx=0, batch_size=32):
    """Check effect of different learning rates on parameter updates."""

    print("\n" + "="*70)
    print("LEARNING RATE EFFECT CHECK")
    print("="*70)

    # Test different learning rates
    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]

    print(f"\nTesting parameter update magnitudes with different learning rates:")
    print(f"(Using first policy generator as example)")

    for lr in learning_rates:
        # Create fresh optimizer
        optimizer = optim.Adam(process_chain.policy_generators[0].parameters(), lr=lr)

        # Save initial parameters
        initial_params = {name: param.clone().detach()
                         for name, param in process_chain.policy_generators[0].named_parameters()}

        # Forward pass
        process_chain.train()
        trajectory = process_chain.forward(batch_size=batch_size, scenario_idx=scenario_idx)

        # Compute loss
        F = surrogate.compute_reliability(trajectory)
        F_star_tensor = torch.tensor(surrogate.F_star[scenario_idx], dtype=torch.float32, device=F.device)
        loss = 100.0 * (F - F_star_tensor) ** 2  # Use same scale as training

        # Backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute parameter changes
        max_change = 0.0
        avg_change = 0.0
        n_params = 0

        for name, param in process_chain.policy_generators[0].named_parameters():
            change = (param - initial_params[name]).abs().max().item()
            max_change = max(max_change, change)
            avg_change += (param - initial_params[name]).abs().mean().item()
            n_params += 1

        avg_change /= n_params

        print(f"  LR = {lr:.6f}:")
        print(f"    Max param change: {max_change:.6e}")
        print(f"    Avg param change: {avg_change:.6e}")

        if avg_change < 1e-8:
            print(f"    ⚠️  Parameters barely moved! Increase learning rate.")
        elif avg_change > 0.1:
            print(f"    ⚠️  Parameters changed dramatically! Learning rate may be too high.")
        else:
            print(f"    ✓  Reasonable parameter update magnitude.")


def check_loss_components(process_chain, surrogate, scenario_idx=0, batch_size=32):
    """Check individual loss components and their scales."""

    print("\n" + "="*70)
    print("LOSS COMPONENTS CHECK")
    print("="*70)

    process_chain.train()
    trajectory = process_chain.forward(batch_size=batch_size, scenario_idx=scenario_idx)

    # Reliability loss
    F = surrogate.compute_reliability(trajectory)
    F_star = surrogate.F_star[scenario_idx]
    F_star_tensor = torch.tensor(F_star, dtype=torch.float32, device=F.device)

    reliability_loss_unscaled = (F - F_star_tensor) ** 2
    reliability_loss_scaled = 100.0 * reliability_loss_unscaled

    print(f"\nReliability Loss:")
    print(f"  F (actual):          {F.item():.6f}")
    print(f"  F* (target):         {F_star:.6f}")
    print(f"  Delta F:             {(F - F_star_tensor).item():.6f}")
    print(f"  Unscaled loss:       {reliability_loss_unscaled.item():.6f}")
    print(f"  Scaled loss (×100):  {reliability_loss_scaled.item():.6f}")

    # Behavior cloning loss
    # (We'd need target trajectory to compute this properly, skipping for now)

    print(f"\nReliability Score Breakdown:")
    # Get individual process contributions
    for process_name, data in trajectory.items():
        outputs = data['outputs_mean' if 'outputs_mean' in data else 'outputs_sampled']
        print(f"  {process_name}: {outputs.mean().item():.4f} ± {outputs.std().item():.4f}")


def main():
    """Run all diagnostic checks."""

    print("="*70)
    print("TRAINING DIAGNOSTICS")
    print("="*70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Filter processes
    process_names = CONTROLLER_CONFIG.get('process_names', None)
    selected_processes = get_filtered_processes(process_names)
    print(f"Selected processes: {[p['name'] for p in selected_processes]}")

    # Generate target trajectory (1 scenario for quick testing)
    print("\nGenerating target trajectory...")
    target_trajectory = generate_target_trajectory(
        process_configs=selected_processes,
        n_samples=1,
        seed=42
    )

    # Create process chain
    print("\nCreating process chain...")
    process_chain = ProcessChain(
        processes_config=selected_processes,
        target_trajectory=target_trajectory,
        policy_config=CONTROLLER_CONFIG['policy_generator'],
        device=device
    )

    # Create surrogate
    print("\nInitializing surrogate...")
    surrogate = ProTSurrogate(
        target_trajectory=target_trajectory,
        device=device,
        use_deterministic_sampling=CONTROLLER_CONFIG.get('surrogate', {}).get('use_deterministic_sampling', True)
    )
    print(f"F* (target): {surrogate.F_star[0]:.6f}")

    # Run diagnostics
    check_loss_components(process_chain, surrogate, scenario_idx=0, batch_size=32)
    check_gradients(process_chain, surrogate, scenario_idx=0, batch_size=32)
    check_learning_rate_effect(process_chain, surrogate, scenario_idx=0, batch_size=32)

    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)

    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Increase learning rate to 1e-4 or 1e-3")
    print("2. Increase n_train to at least 10-50 scenarios")
    print("3. Consider rebalancing surrogate weights (laser vs plasma)")
    print("4. Check if gradients are flowing properly (should be in 1e-5 to 1e-2 range)")


if __name__ == '__main__':
    main()
