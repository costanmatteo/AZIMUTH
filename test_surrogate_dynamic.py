"""Test script to verify dynamic reliability computation works with different process counts."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.src.models.surrogate import ProTSurrogate

print("="*70)
print("TESTING DYNAMIC RELIABILITY COMPUTATION")
print("="*70)

# Test 1: With 2 processes (laser + plasma)
print("\n[TEST 1] Testing with 2 processes (laser, plasma)")
print("-"*70)

target_trajectory_2 = {
    'laser': {
        'inputs': np.array([[0.5, 25.0]]),
        'outputs': np.array([[0.5]])  # At target
    },
    'plasma': {
        'inputs': np.array([[200.0, 30.0]]),
        'outputs': np.array([[5.0]])  # At target
    }
}

surrogate_2 = ProTSurrogate(target_trajectory_2, device='cpu')
print(f"✓ Surrogate created with 2 processes")
print(f"  F* (target reliability): {surrogate_2.F_star[0]:.6f}")

# Test reliability computation
test_traj_2 = {
    'laser': {
        'inputs': torch.tensor([[0.5, 25.0]]),
        'outputs_mean': torch.tensor([[0.5]]),  # At target
        'outputs_var': torch.tensor([[0.0]])
    },
    'plasma': {
        'inputs': torch.tensor([[200.0, 30.0]]),
        'outputs_mean': torch.tensor([[5.0]]),  # At target
        'outputs_var': torch.tensor([[0.0]])
    }
}

F_2 = surrogate_2.compute_reliability(test_traj_2)
print(f"  F (at target): {F_2.item():.6f}")
print(f"  Expected: ~1.0 (both processes at target)")

assert F_2.item() > 0.95, f"Expected F close to 1.0, got {F_2.item():.6f}"
print("✓ Test 1 passed!")

# Test 2: With 4 processes (all)
print("\n[TEST 2] Testing with 4 processes (laser, plasma, galvanic, microetch)")
print("-"*70)

target_trajectory_4 = {
    'laser': {
        'inputs': np.array([[0.5, 25.0]]),
        'outputs': np.array([[0.5]])
    },
    'plasma': {
        'inputs': np.array([[200.0, 30.0]]),
        'outputs': np.array([[5.0]])
    },
    'galvanic': {
        'inputs': np.array([[10.0, 50.0]]),
        'outputs': np.array([[10.0]])
    },
    'microetch': {
        'inputs': np.array([[30.0, 60.0]]),
        'outputs': np.array([[20.0]])
    }
}

surrogate_4 = ProTSurrogate(target_trajectory_4, device='cpu')
print(f"✓ Surrogate created with 4 processes")
print(f"  F* (target reliability): {surrogate_4.F_star[0]:.6f}")

test_traj_4 = {
    'laser': {
        'inputs': torch.tensor([[0.5, 25.0]]),
        'outputs_mean': torch.tensor([[0.5]]),
        'outputs_var': torch.tensor([[0.0]])
    },
    'plasma': {
        'inputs': torch.tensor([[200.0, 30.0]]),
        'outputs_mean': torch.tensor([[5.0]]),
        'outputs_var': torch.tensor([[0.0]])
    },
    'galvanic': {
        'inputs': torch.tensor([[10.0, 50.0]]),
        'outputs_mean': torch.tensor([[10.0]]),
        'outputs_var': torch.tensor([[0.0]])
    },
    'microetch': {
        'inputs': torch.tensor([[30.0, 60.0]]),
        'outputs_mean': torch.tensor([[20.0]]),
        'outputs_var': torch.tensor([[0.0]])
    }
}

F_4 = surrogate_4.compute_reliability(test_traj_4)
print(f"  F (at target): {F_4.item():.6f}")
print(f"  Expected: ~1.0 (all processes at target)")

assert F_4.item() > 0.95, f"Expected F close to 1.0, got {F_4.item():.6f}"
print("✓ Test 2 passed!")

# Test 3: With 1 process (just laser)
print("\n[TEST 3] Testing with 1 process (laser only)")
print("-"*70)

target_trajectory_1 = {
    'laser': {
        'inputs': np.array([[0.5, 25.0]]),
        'outputs': np.array([[0.5]])
    }
}

surrogate_1 = ProTSurrogate(target_trajectory_1, device='cpu')
print(f"✓ Surrogate created with 1 process")
print(f"  F* (target reliability): {surrogate_1.F_star[0]:.6f}")

test_traj_1 = {
    'laser': {
        'inputs': torch.tensor([[0.5, 25.0]]),
        'outputs_mean': torch.tensor([[0.5]]),
        'outputs_var': torch.tensor([[0.0]])
    }
}

F_1 = surrogate_1.compute_reliability(test_traj_1)
print(f"  F (at target): {F_1.item():.6f}")
print(f"  Expected: ~1.0 (process at target)")

assert F_1.item() > 0.95, f"Expected F close to 1.0, got {F_1.item():.6f}"
print("✓ Test 3 passed!")

# Test 4: Test gradient flow
print("\n[TEST 4] Testing gradient flow with 2 processes")
print("-"*70)

test_traj_grad = {
    'laser': {
        'inputs': torch.tensor([[0.5, 25.0]], requires_grad=True),
        'outputs_mean': torch.tensor([[0.48]], requires_grad=True),
        'outputs_var': torch.tensor([[0.01]])
    },
    'plasma': {
        'inputs': torch.tensor([[200.0, 30.0]], requires_grad=True),
        'outputs_mean': torch.tensor([[4.9]], requires_grad=True),
        'outputs_var': torch.tensor([[0.02]])
    }
}

F_grad = surrogate_2.compute_reliability(test_traj_grad)
F_grad.backward()

print(f"  F (slightly off target): {F_grad.item():.6f}")
print(f"  Laser output grad exists: {test_traj_grad['laser']['outputs_mean'].grad is not None}")
print(f"  Plasma output grad exists: {test_traj_grad['plasma']['outputs_mean'].grad is not None}")

if test_traj_grad['laser']['outputs_mean'].grad is not None:
    print(f"  Laser output grad: {test_traj_grad['laser']['outputs_mean'].grad.item():.6f}")
if test_traj_grad['plasma']['outputs_mean'].grad is not None:
    print(f"  Plasma output grad: {test_traj_grad['plasma']['outputs_mean'].grad.item():.6f}")

assert test_traj_grad['laser']['outputs_mean'].grad is not None
assert test_traj_grad['plasma']['outputs_mean'].grad is not None
print("✓ Test 4 passed!")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nDynamic reliability computation works correctly with:")
print("  - 1 process (laser only)")
print("  - 2 processes (laser + plasma)")
print("  - 4 processes (all)")
print("  - Gradient flow is maintained")
