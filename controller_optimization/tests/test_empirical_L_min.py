"""
Tests for empirical L_min calculation module.

These tests verify that the Monte Carlo estimation of L_min works correctly
for sequential processes with adaptive targets.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.src.analysis.empirical_L_min import (
    compute_empirical_L_min,
    compute_empirical_L_min_multi_scenario,
    compute_aggregate_empirical_L_min,
    EmpiricalLminResult,
    EmpiricalLminTracker,
    compare_analytical_vs_empirical
)


class MockProcessChain:
    """Mock ProcessChain for testing."""

    def __init__(self, mean_output=0.5, variance=0.01):
        self.mean_output = mean_output
        self.variance = variance
        self.process_names = ['mock_process']
        self.target_trajectory = {
            'mock_process': {
                'inputs': np.array([[1.0, 2.0]]),
                'outputs': np.array([[mean_output]])
            }
        }

    def eval(self):
        pass

    def forward(self, batch_size=1, scenario_idx=0):
        # Generate stochastic output
        std = np.sqrt(self.variance)
        sampled_output = self.mean_output + std * np.random.randn()

        return {
            'mock_process': {
                'inputs': torch.tensor([[1.0, 2.0]]),
                'outputs_mean': torch.tensor([[self.mean_output]]),
                'outputs_var': torch.tensor([[self.variance]]),
                'outputs_sampled': torch.tensor([[sampled_output]])
            }
        }


class MockSurrogate:
    """Mock surrogate for testing."""

    def __init__(self, target=0.5, scale=0.1):
        self.target = target
        self.scale = scale
        self.use_deterministic_sampling = False
        self.F_star = np.array([0.9])  # Target reliability

    def compute_reliability(self, trajectory):
        # Quality function: Q = exp(-(o - target)^2 / scale)
        output = trajectory['mock_process']['outputs_sampled']
        if isinstance(output, torch.Tensor):
            output = output.item()
        quality = np.exp(-((output - self.target)**2) / self.scale)
        return torch.tensor(quality)


class TestEmpiricalLminBasic:
    """Basic functionality tests."""

    def test_result_structure(self):
        """Test that EmpiricalLminResult has correct structure."""
        chain = MockProcessChain()
        surrogate = MockSurrogate()

        result = compute_empirical_L_min(
            process_chain=chain,
            surrogate=surrogate,
            F_star=0.9,
            n_samples=50,
            verbose=False
        )

        assert isinstance(result, EmpiricalLminResult)
        assert result.n_samples == 50
        assert 0 <= result.E_F <= 1
        assert result.Var_F >= 0
        assert result.L_min >= 0
        assert result.F_star == 0.9

    def test_L_min_decomposition(self):
        """Test that L_min = Var[F] + Bias^2."""
        chain = MockProcessChain()
        surrogate = MockSurrogate()

        result = compute_empirical_L_min(
            process_chain=chain,
            surrogate=surrogate,
            F_star=0.9,
            n_samples=100,
            verbose=False
        )

        computed = result.Var_F + result.Bias2
        assert abs(result.L_min - computed) < 1e-10, \
            f"L_min decomposition failed: {result.L_min} != {computed}"

    def test_deterministic_case(self):
        """Test that with zero variance, Var[F] is near zero."""
        chain = MockProcessChain(variance=0.0)  # No variance
        surrogate = MockSurrogate()

        result = compute_empirical_L_min(
            process_chain=chain,
            surrogate=surrogate,
            F_star=0.9,
            n_samples=50,
            verbose=False
        )

        # With zero variance, all F values should be the same
        # so Var[F] should be very small (just numerical noise)
        assert result.Var_F < 0.001, \
            f"Var[F] should be near 0 for deterministic case, got {result.Var_F}"

    def test_confidence_interval(self):
        """Test that confidence interval contains L_min."""
        chain = MockProcessChain()
        surrogate = MockSurrogate()

        result = compute_empirical_L_min(
            process_chain=chain,
            surrogate=surrogate,
            F_star=0.9,
            n_samples=500,
            verbose=False
        )

        ci_low, ci_high = result.confidence_interval
        assert ci_low <= result.L_min <= ci_high, \
            f"L_min {result.L_min} not in CI [{ci_low}, {ci_high}]"


class TestEmpiricalLminMultiScenario:
    """Tests for multi-scenario computation."""

    def test_multi_scenario_basic(self):
        """Test multi-scenario computation."""
        chain = MockProcessChain()
        surrogate = MockSurrogate()
        surrogate.F_star = np.array([0.9, 0.85, 0.95])

        F_star_per_scenario = {0: 0.9, 1: 0.85, 2: 0.95}

        results = compute_empirical_L_min_multi_scenario(
            process_chain=chain,
            surrogate=surrogate,
            F_star_per_scenario=F_star_per_scenario,
            n_samples_per_scenario=50,
            verbose=False
        )

        assert len(results) == 3
        for idx in [0, 1, 2]:
            assert idx in results
            assert isinstance(results[idx], EmpiricalLminResult)
            assert results[idx].F_star == F_star_per_scenario[idx]

    def test_aggregate_computation(self):
        """Test aggregation across scenarios."""
        # Create mock results
        result1 = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=100, std_error_E_F=0.01,
            std_error_Var_F=0.001, confidence_interval=(0.008, 0.012),
            computation_time=1.0
        )
        result2 = EmpiricalLminResult(
            L_min=0.02, E_F=0.80, Var_F=0.01, Bias2=0.01,
            F_star=0.85, n_samples=100, std_error_E_F=0.01,
            std_error_Var_F=0.002, confidence_interval=(0.016, 0.024),
            computation_time=1.0
        )

        scenario_results = {0: result1, 1: result2}
        aggregate = compute_aggregate_empirical_L_min(scenario_results)

        # Check uniform weighting
        assert abs(aggregate.L_min - 0.015) < 0.001  # (0.01 + 0.02) / 2
        assert abs(aggregate.E_F - 0.825) < 0.001  # (0.85 + 0.80) / 2
        assert aggregate.n_samples == 200  # Total samples


class TestEmpiricalLminTracker:
    """Tests for the tracker class."""

    def test_tracker_update(self):
        """Test tracker update functionality."""
        tracker = EmpiricalLminTracker(loss_scale=100.0)

        result = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=100, std_error_E_F=0.01,
            std_error_Var_F=0.001, confidence_interval=(0.008, 0.012),
            computation_time=1.0
        )

        tracker.update(epoch=1, observed_loss=2.0, empirical_result=result)

        assert len(tracker.history['epoch']) == 1
        assert tracker.history['epoch'][0] == 1
        assert tracker.history['observed_loss'][0] == 2.0
        assert tracker.history['empirical_L_min'][0] == 1.0  # 0.01 * 100

    def test_tracker_efficiency(self):
        """Test efficiency calculation."""
        tracker = EmpiricalLminTracker(loss_scale=100.0)

        result = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=100, std_error_E_F=0.01,
            std_error_Var_F=0.001, confidence_interval=(0.008, 0.012),
            computation_time=1.0
        )

        # Observed loss = 2.0, L_min (scaled) = 1.0
        # Efficiency = L_min / observed = 1.0 / 2.0 = 0.5
        tracker.update(epoch=1, observed_loss=2.0, empirical_result=result)

        assert abs(tracker.history['efficiency'][0] - 0.5) < 0.001

    def test_tracker_serialization(self):
        """Test tracker can be serialized to dict."""
        tracker = EmpiricalLminTracker(loss_scale=100.0)

        result = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=100, std_error_E_F=0.01,
            std_error_Var_F=0.001, confidence_interval=(0.008, 0.012),
            computation_time=1.0
        )

        tracker.update(epoch=1, observed_loss=2.0, empirical_result=result)

        data = tracker.to_dict()
        assert 'history' in data
        assert 'summary' in data
        assert 'loss_scale' in data


class TestCompareAnalyticalVsEmpirical:
    """Tests for analytical vs empirical comparison."""

    def test_comparison_within_tolerance(self):
        """Test comparison when values are close."""
        result = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=1000, std_error_E_F=0.001,
            std_error_Var_F=0.0001, confidence_interval=(0.009, 0.011),
            computation_time=1.0
        )

        comparison = compare_analytical_vs_empirical(
            analytical_L_min=1.0,  # Already scaled
            empirical_result=result,
            loss_scale=100.0,
            verbose=False
        )

        assert comparison['analytical_L_min'] == 1.0
        assert abs(comparison['empirical_L_min_scaled'] - 1.0) < 0.01
        assert abs(comparison['ratio'] - 1.0) < 0.02

    def test_comparison_with_discrepancy(self):
        """Test comparison when values differ significantly."""
        result = EmpiricalLminResult(
            L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
            F_star=0.9, n_samples=1000, std_error_E_F=0.001,
            std_error_Var_F=0.0001, confidence_interval=(0.009, 0.011),
            computation_time=1.0
        )

        comparison = compare_analytical_vs_empirical(
            analytical_L_min=3.0,  # 3x the empirical value (scaled)
            empirical_result=result,
            loss_scale=100.0,
            verbose=False
        )

        assert comparison['ratio'] == 3.0
        assert not comparison['analytical_within_CI']


def test_result_to_dict():
    """Test that EmpiricalLminResult can be serialized."""
    result = EmpiricalLminResult(
        L_min=0.01, E_F=0.85, Var_F=0.005, Bias2=0.005,
        F_star=0.9, n_samples=100, std_error_E_F=0.01,
        std_error_Var_F=0.001, confidence_interval=(0.008, 0.012),
        computation_time=1.0
    )

    data = result.to_dict()
    assert data['L_min'] == 0.01
    assert data['E_F'] == 0.85
    assert data['n_samples'] == 100
    assert isinstance(data['confidence_interval'], list)


if __name__ == '__main__':
    # Run tests
    print("Running empirical L_min tests...")
    print("="*60)

    # Basic tests
    test_basic = TestEmpiricalLminBasic()
    test_basic.test_result_structure()
    print("  test_result_structure passed")

    test_basic.test_L_min_decomposition()
    print("  test_L_min_decomposition passed")

    test_basic.test_deterministic_case()
    print("  test_deterministic_case passed")

    test_basic.test_confidence_interval()
    print("  test_confidence_interval passed")

    # Multi-scenario tests
    test_multi = TestEmpiricalLminMultiScenario()
    test_multi.test_multi_scenario_basic()
    print("  test_multi_scenario_basic passed")

    test_multi.test_aggregate_computation()
    print("  test_aggregate_computation passed")

    # Tracker tests
    test_tracker = TestEmpiricalLminTracker()
    test_tracker.test_tracker_update()
    print("  test_tracker_update passed")

    test_tracker.test_tracker_efficiency()
    print("  test_tracker_efficiency passed")

    test_tracker.test_tracker_serialization()
    print("  test_tracker_serialization passed")

    # Comparison tests
    test_compare = TestCompareAnalyticalVsEmpirical()
    test_compare.test_comparison_within_tolerance()
    print("  test_comparison_within_tolerance passed")

    test_compare.test_comparison_with_discrepancy()
    print("  test_comparison_with_discrepancy passed")

    # Serialization test
    test_result_to_dict()
    print("  test_result_to_dict passed")

    print("="*60)
    print("All tests passed!")
