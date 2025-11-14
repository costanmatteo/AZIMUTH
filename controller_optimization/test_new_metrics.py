"""
Quick test for new metrics and visualization functions.
Tests both single and multiple scenario cases.
"""

import numpy as np
import sys
from pathlib import Path

# Add controller_optimization to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.src.utils.metrics import (
    compute_worst_case_gap,
    compute_success_rate,
    compute_train_test_gap,
    compute_scenario_diversity
)

def test_metrics_single_scenario():
    """Test metrics with single scenario"""
    print("\n" + "="*70)
    print("TEST 1: Single Scenario")
    print("="*70)

    F_star = np.array([0.95])
    F_actual = np.array([0.90])

    # Worst case gap
    worst_case = compute_worst_case_gap(F_star, F_actual)
    print(f"\nWorst-case gap: {worst_case['worst_case_gap']:.6f}")
    assert worst_case['worst_case_gap'] == 0.05, "Worst case gap should be 0.05"

    # Success rate
    success_rate = compute_success_rate(F_star, F_actual, threshold=0.95)
    print(f"Success rate (95%): {success_rate['success_rate_pct']:.1f}%")
    assert success_rate['n_successful'] == 1, "Should have 1 successful scenario"

    # Scenario diversity
    structural_conditions = {
        'AmbientTemp': np.array([25.0]),
        'Humidity': np.array([0.5])
    }
    diversity = compute_scenario_diversity(structural_conditions)
    print(f"Diversity score: {diversity['diversity_score']:.4f}")
    assert diversity['diversity_score'] == 0.0, "Single scenario should have 0 diversity"

    print("\n✓ Single scenario tests PASSED")


def test_metrics_multiple_scenarios():
    """Test metrics with multiple scenarios"""
    print("\n" + "="*70)
    print("TEST 2: Multiple Scenarios")
    print("="*70)

    # Train set
    F_star_train = np.array([0.95, 0.92, 0.96, 0.93, 0.94])
    F_actual_train = np.array([0.90, 0.88, 0.91, 0.89, 0.90])

    # Test set
    F_star_test = np.array([0.94, 0.93, 0.95])
    F_actual_test = np.array([0.89, 0.88, 0.90])

    # Worst case gap
    worst_case_train = compute_worst_case_gap(F_star_train, F_actual_train)
    print(f"\nWorst-case gap (train): {worst_case_train['worst_case_gap']:.6f} at scenario {worst_case_train['worst_case_scenario_idx']}")
    assert worst_case_train['worst_case_gap'] > 0, "Should have positive gap"

    # Success rate
    success_rate_train = compute_success_rate(F_star_train, F_actual_train, threshold=0.95)
    print(f"Success rate (train, 95%): {success_rate_train['success_rate_pct']:.1f}% ({success_rate_train['n_successful']}/{success_rate_train['n_total']})")

    success_rate_test = compute_success_rate(F_star_test, F_actual_test, threshold=0.95)
    print(f"Success rate (test, 95%): {success_rate_test['success_rate_pct']:.1f}% ({success_rate_test['n_successful']}/{success_rate_test['n_total']})")

    # Train-test gap
    train_test_gap = compute_train_test_gap(F_star_train, F_actual_train, F_star_test, F_actual_test)
    print(f"\nTrain-test gap: {train_test_gap['train_test_gap']:.6f}")
    print(f"  Mean gap (train): {train_test_gap['mean_gap_train']:.6f}")
    print(f"  Mean gap (test):  {train_test_gap['mean_gap_test']:.6f}")

    # Scenario diversity
    structural_conditions = {
        'AmbientTemp': np.array([20.0, 25.0, 30.0, 22.0, 28.0]),
        'Humidity': np.array([0.3, 0.5, 0.7, 0.4, 0.6]),
        'Pressure': np.array([1.0, 1.1, 1.2, 1.05, 1.15])
    }
    diversity = compute_scenario_diversity(structural_conditions)
    print(f"\nDiversity score: {diversity['diversity_score']:.4f}")
    print(f"Per-condition CV:")
    for var, cv in diversity['per_condition_cv'].items():
        print(f"  {var}: {cv:.4f}")
    assert diversity['diversity_score'] > 0, "Multiple scenarios should have positive diversity"

    print("\n✓ Multiple scenario tests PASSED")


def test_visualization_functions():
    """Test that visualization functions can be imported and called"""
    print("\n" + "="*70)
    print("TEST 3: Visualization Functions")
    print("="*70)

    from controller_optimization.src.utils.visualization import (
        plot_target_vs_actual_scatter,
        plot_gap_distribution
    )

    # Test with simple data (don't save plots)
    F_star = np.array([0.95, 0.92, 0.96])
    F_actual = np.array([0.90, 0.88, 0.91])

    print("\nTesting scatter plot function...")
    try:
        # Just test that function can be called (no actual plot saved)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        plot_target_vs_actual_scatter(F_star, F_actual, save_path='/tmp/test_scatter.png')
        print("  ✓ Scatter plot function works")
    except Exception as e:
        print(f"  ✗ Scatter plot failed: {e}")
        raise

    print("\nTesting gap distribution function...")
    try:
        plot_gap_distribution(F_star, F_actual, save_path='/tmp/test_gap.png')
        print("  ✓ Gap distribution function works")
    except Exception as e:
        print(f"  ✗ Gap distribution failed: {e}")
        raise

    print("\n✓ Visualization tests PASSED")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING NEW METRICS AND VISUALIZATIONS")
    print("="*70)

    try:
        test_metrics_single_scenario()
        test_metrics_multiple_scenarios()
        test_visualization_functions()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nNew features are ready to use:")
        print("  - Worst-case gap metric")
        print("  - Success rate metric (configurable threshold)")
        print("  - Train-test gap metric")
        print("  - Scenario diversity score")
        print("  - Target vs Actual scatter plot")
        print("  - Gap distribution histogram")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n✗ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
