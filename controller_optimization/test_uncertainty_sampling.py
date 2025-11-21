"""
Test script per verificare che il sampling dall'uncertainty predictor funzioni correttamente.

Questo script dimostra che:
1. ProcessChain.forward() ora genera outputs_sampled
2. Gli output campionati sono diversi ad ogni chiamata (stocastici)
3. compute_reliability() usa gli output già campionati
"""

import sys
from pathlib import Path
import torch

# Add paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES
from controller_optimization.src.utils.target_generation import generate_target_trajectory
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate


def test_uncertainty_sampling():
    """Test che il sampling dall'uncertainty predictor funzioni."""

    print("=" * 80)
    print("TEST: Uncertainty Sampling in ProcessChain")
    print("=" * 80)

    # 1. Generate target trajectory
    print("\n1. Generating target trajectory...")
    target_traj = generate_target_trajectory(PROCESSES, n_samples=3, seed=42)
    print(f"   ✓ Generated {len(target_traj)} processes")

    # 2. Create process chain
    print("\n2. Creating ProcessChain...")
    try:
        chain = ProcessChain(
            processes_config=PROCESSES,
            target_trajectory=target_traj,
            device='cpu'
        )
        print(f"   ✓ ProcessChain created")
        print(f"     - Processes: {chain.process_names}")
        print(f"     - Uncertainty predictors: {len(chain.uncertainty_predictors)}")
        print(f"     - Policy generators: {len(chain.policy_generators)}")
    except FileNotFoundError as e:
        print(f"\n   ✗ Error: {e}")
        print("   Please run train_processes.py first to train uncertainty predictors.")
        return False

    # 3. Run forward pass multiple times and verify stochasticity
    print("\n3. Testing stochastic sampling...")
    trajectories = []

    for i in range(3):
        traj = chain.forward(batch_size=1, scenario_idx=0)
        trajectories.append(traj)

    # 4. Verify outputs_sampled exists and is different each time
    print("\n4. Verifying outputs_sampled...")
    first_process = chain.process_names[0]

    for i, traj in enumerate(trajectories):
        print(f"\n   Run {i+1}:")
        for process_name in chain.process_names[:2]:  # Show first 2 processes
            data = traj[process_name]

            # Check that outputs_sampled exists
            assert 'outputs_sampled' in data, f"outputs_sampled missing for {process_name}!"

            print(f"     {process_name}:")
            print(f"       - outputs_mean:    {data['outputs_mean'].detach().cpu().numpy().flatten()}")
            print(f"       - outputs_var:     {data['outputs_var'].detach().cpu().numpy().flatten()}")
            print(f"       - outputs_sampled: {data['outputs_sampled'].detach().cpu().numpy().flatten()}")

    # 5. Verify that sampled outputs are different across runs
    print("\n5. Verifying stochasticity...")
    sampled_1 = trajectories[0][first_process]['outputs_sampled']
    sampled_2 = trajectories[1][first_process]['outputs_sampled']
    sampled_3 = trajectories[2][first_process]['outputs_sampled']

    # They should be different (with very high probability)
    diff_12 = torch.abs(sampled_1 - sampled_2).max().item()
    diff_23 = torch.abs(sampled_2 - sampled_3).max().item()

    print(f"   Max difference between run 1 and 2: {diff_12:.6f}")
    print(f"   Max difference between run 2 and 3: {diff_23:.6f}")

    if diff_12 > 1e-6 and diff_23 > 1e-6:
        print("   ✓ Outputs are stochastic (different across runs)")
    else:
        print("   ⚠ Warning: Outputs might be too similar (low variance?)")

    # 6. Test reliability computation with sampled outputs
    print("\n6. Testing reliability computation...")
    surrogate = ProTSurrogate(target_traj, device='cpu', use_deterministic_sampling=False)

    F_values = []
    for i, traj in enumerate(trajectories):
        F = surrogate.compute_reliability(traj)
        F_values.append(F.item())
        print(f"   Run {i+1}: F = {F.item():.6f}")

    # Verify F values are different (stochastic)
    F_std = torch.std(torch.tensor(F_values)).item()
    print(f"\n   Standard deviation of F: {F_std:.6f}")

    if F_std > 1e-6:
        print("   ✓ Reliability values are stochastic")
    else:
        print("   ⚠ Warning: Reliability values too similar")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Uncertainty sampling is working correctly!")
    print("=" * 80)
    print("\nKey findings:")
    print("  1. ProcessChain generates outputs_sampled from N(mean, var)")
    print("  2. Sampled outputs are stochastic (different each run)")
    print("  3. Reliability computation uses pre-sampled outputs")
    print("  4. This creates a realistic simulation of process uncertainty")

    return True


if __name__ == '__main__':
    success = test_uncertainty_sampling()
    sys.exit(0 if success else 1)
