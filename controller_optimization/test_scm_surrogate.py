"""
Test rapido del surrogato SCM per identificare eventuali NaN.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES
from controller_optimization.src.models.scm_surrogate import create_scm_surrogate_for_process

def test_process_surrogate(process_config):
    """Test surrogate for a single process."""
    process_name = process_config['name']
    print(f"\n{'='*70}")
    print(f"Testing {process_name} surrogate")
    print(f"{'='*70}")

    # Create surrogate
    print(f"  Creating surrogate...")
    try:
        surrogate = create_scm_surrogate_for_process(process_config, device='cpu')
        print(f"  ✓ Surrogate created")
    except Exception as e:
        print(f"  ✗ Error creating surrogate: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get realistic input ranges from config
    input_labels = process_config['input_labels']
    print(f"  Input labels: {input_labels}")

    # Create test inputs with realistic values
    batch_size = 4

    # Define realistic ranges for each process
    if process_name == 'laser':
        # PowerTarget: 0.1-1.0, AmbientTemp: 15-35
        test_inputs = torch.tensor([
            [0.5, 25.0],  # Nominal
            [0.2, 20.0],  # Low power, cool
            [0.8, 30.0],  # High power, warm
            [0.1, 15.0],  # Threshold power, cold
        ], dtype=torch.float32)
    elif process_name == 'plasma':
        # RF_Power: 100-400, Duration: 10-60
        test_inputs = torch.tensor([
            [200.0, 30.0],  # Nominal
            [150.0, 20.0],  # Low power
            [350.0, 50.0],  # High power
            [100.0, 10.0],  # Min values
        ], dtype=torch.float32)
    elif process_name == 'galvanic':
        # CurrentDensity: 1-5, Duration: 600-3600
        test_inputs = torch.tensor([
            [3.0, 1800.0],  # Nominal
            [2.0, 1000.0],  # Low
            [4.5, 3000.0],  # High
            [1.0, 600.0],   # Min values
        ], dtype=torch.float32)
    elif process_name == 'microetch':
        # Temperature: 293-323K, Concentration: 0.5-3.0, Duration: 30-180
        test_inputs = torch.tensor([
            [298.0, 1.5, 60.0],   # Nominal
            [293.0, 0.5, 30.0],   # Cool, dilute, short
            [323.0, 3.0, 180.0],  # Hot, concentrated, long
            [310.0, 2.0, 120.0],  # Mid-range
        ], dtype=torch.float32)
    else:
        print(f"  ⚠ Unknown process, using random inputs")
        input_dim = len(input_labels)
        test_inputs = torch.randn(batch_size, input_dim)

    print(f"  Test inputs shape: {test_inputs.shape}")
    print(f"  Input ranges:")
    for i, label in enumerate(input_labels):
        print(f"    {label}: [{test_inputs[:, i].min().item():.2f}, {test_inputs[:, i].max().item():.2f}]")

    # Forward pass
    print(f"\n  Running forward pass...")
    try:
        mean, variance = surrogate(test_inputs)
        print(f"  ✓ Forward pass completed")
    except Exception as e:
        print(f"  ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check shapes
    print(f"\n  Output shapes:")
    print(f"    Mean: {mean.shape}")
    print(f"    Variance: {variance.shape}")

    # Check for NaN/Inf
    has_nan_mean = torch.isnan(mean).any().item()
    has_inf_mean = torch.isinf(mean).any().item()
    has_nan_var = torch.isnan(variance).any().item()
    has_inf_var = torch.isinf(variance).any().item()

    print(f"\n  Value checks:")
    print(f"    Mean contains NaN: {has_nan_mean}")
    print(f"    Mean contains Inf: {has_inf_mean}")
    print(f"    Variance contains NaN: {has_nan_var}")
    print(f"    Variance contains Inf: {has_inf_var}")

    if has_nan_mean or has_inf_mean or has_nan_var or has_inf_var:
        print(f"  ✗ FOUND NaN/Inf VALUES!")
        print(f"\n  Mean values:")
        print(mean)
        print(f"\n  Variance values:")
        print(variance)
        return False

    # Print value ranges
    print(f"\n  Output ranges:")
    print(f"    Mean: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"    Variance: [{variance.min().item():.6f}, {variance.max().item():.6f}]")

    # Test differentiability
    print(f"\n  Testing differentiability...")
    test_inputs_grad = test_inputs.clone().requires_grad_(True)
    mean_grad, variance_grad = surrogate(test_inputs_grad)
    loss = mean_grad.sum()
    try:
        loss.backward()
        print(f"  ✓ Backward pass completed")

        has_grad = test_inputs_grad.grad is not None
        print(f"    Gradient exists: {has_grad}")

        if has_grad:
            has_nan_grad = torch.isnan(test_inputs_grad.grad).any().item()
            has_inf_grad = torch.isinf(test_inputs_grad.grad).any().item()
            grad_norm = test_inputs_grad.grad.norm().item()

            print(f"    Gradient contains NaN: {has_nan_grad}")
            print(f"    Gradient contains Inf: {has_inf_grad}")
            print(f"    Gradient norm: {grad_norm:.6f}")

            if has_nan_grad or has_inf_grad:
                print(f"  ✗ FOUND NaN/Inf IN GRADIENTS!")
                return False
    except Exception as e:
        print(f"  ✗ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  ✓ All checks passed for {process_name}!")
    return True


if __name__ == '__main__':
    print("="*70)
    print("SCM SURROGATE TEST")
    print("="*70)

    # Test only laser and plasma (as configured)
    process_names = ['laser', 'plasma']

    all_passed = True
    for process_config in PROCESSES:
        if process_config['name'] in process_names:
            passed = test_process_surrogate(process_config)
            all_passed = all_passed and passed

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print(f"{'='*70}")
