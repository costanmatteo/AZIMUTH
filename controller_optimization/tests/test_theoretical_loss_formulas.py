"""
Validation tests for theoretical loss formulas.

These tests verify that the implementation matches the mathematical formulas
from "On the Structural Bias in Reliability Loss Functions with Stochastic
Reparameterization".

Test categories:
1. Limit tests (σ² → 0)
2. Perfect policy tests (δ = 0)
3. Positivity tests
4. Independent process tests (ρ = 0)
5. Multi-process consistency tests
6. Numerical verification with manual calculations
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.theoretical_loss_analysis import (
    compute_theoretical_E_F,
    compute_theoretical_E_F2,
    compute_theoretical_L_min,
    compute_per_process_Q_stats,
    compute_multi_process_L_min,
    TheoreticalLossComponents
)


class TestLimitCases:
    """Test behavior as σ² → 0 (deterministic limit)."""

    def test_E_F_deterministic_limit(self):
        """When σ² = 0, E[F] should equal F*."""
        F_star = 0.85
        delta = 0.2
        sigma2 = 0.0
        s = 1.0

        E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
        assert np.isclose(E_F, F_star), f"E[F] = {E_F}, expected {F_star}"

    def test_E_F2_deterministic_limit(self):
        """When σ² = 0, E[F²] should equal F*²."""
        F_star = 0.85
        delta = 0.2
        sigma2 = 0.0
        s = 1.0

        E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)
        assert np.isclose(E_F2, F_star**2), f"E[F²] = {E_F2}, expected {F_star**2}"

    def test_L_min_deterministic_limit(self):
        """When σ² = 0, L_min should be 0."""
        F_star = 0.85
        delta = 0.2
        sigma2 = 0.0
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert np.isclose(components.L_min, 0.0), f"L_min = {components.L_min}, expected 0"
        assert np.isclose(components.Var_F, 0.0), f"Var[F] = {components.Var_F}, expected 0"

    def test_Var_F_approaches_zero(self):
        """Var[F] should approach 0 as σ² → 0."""
        F_star = 0.9
        delta = 0.1
        s = 1.0

        variances = []
        for sigma2 in [0.1, 0.01, 0.001, 0.0001]:
            components = compute_theoretical_L_min(F_star, delta, sigma2, s)
            variances.append(components.Var_F)

        # Verify monotonically decreasing
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i+1], "Var[F] should decrease with σ²"


class TestPerfectPolicy:
    """Test perfect policy case (δ = 0, meaning μ = τ)."""

    def test_F_star_equals_one(self):
        """When δ = 0, F* = exp(-δ²/s) = 1."""
        delta = 0.0
        s = 1.0
        F_star = np.exp(-delta**2 / s)
        assert np.isclose(F_star, 1.0), f"F* = {F_star}, expected 1.0"

    def test_E_F_perfect_policy(self):
        """When δ = 0, E[F] = 1/√(1 + 2σ²/s)."""
        F_star = 1.0  # δ = 0 implies F* = 1
        delta = 0.0
        sigma2 = 0.5
        s = 1.0

        E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
        expected = 1.0 / np.sqrt(1 + 2 * sigma2 / s)
        assert np.isclose(E_F, expected), f"E[F] = {E_F}, expected {expected}"

    def test_E_F2_perfect_policy(self):
        """When δ = 0, E[F²] = 1/√(1 + 4σ²/s)."""
        F_star = 1.0  # δ = 0 implies F* = 1
        delta = 0.0
        sigma2 = 0.5
        s = 1.0

        E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)
        expected = 1.0 / np.sqrt(1 + 4 * sigma2 / s)
        assert np.isclose(E_F2, expected), f"E[F²] = {E_F2}, expected {expected}"

    def test_Var_F_perfect_policy_formula(self):
        """When δ = 0: Var[F] = 1/√(1+4σ²/s) - 1/(1+2σ²/s)."""
        delta = 0.0
        sigma2 = 0.5
        s = 1.0

        E_Q, E_Q2, Var_Q = compute_per_process_Q_stats(
            Q_star=1.0, delta=delta, sigma2=sigma2, s=s
        )

        # From Corollary 17
        expected_Var = 1.0/np.sqrt(1 + 4*sigma2/s) - 1.0/(1 + 2*sigma2/s)
        assert np.isclose(Var_Q, expected_Var, rtol=1e-5), \
            f"Var[F] = {Var_Q}, expected {expected_Var}"

    def test_bias_perfect_policy(self):
        """When δ = 0: Bias = 1/√(1 + 2σ²/s) - 1 < 0."""
        F_star = 1.0
        delta = 0.0
        sigma2 = 0.5
        s = 1.0

        E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
        bias = E_F - F_star

        expected_bias = 1.0/np.sqrt(1 + 2*sigma2/s) - 1.0
        assert np.isclose(bias, expected_bias), f"Bias = {bias}, expected {expected_bias}"
        assert bias < 0, "Bias should be negative for concave function with σ² > 0"


class TestPositivity:
    """Test positivity constraints (L_min > 0, Var[F] > 0 when σ² > 0)."""

    def test_L_min_positive_when_stochastic(self):
        """L_min > 0 when σ² > 0."""
        F_star = 0.9
        delta = 0.1
        sigma2 = 0.05
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert components.L_min > 0, f"L_min should be > 0, got {components.L_min}"

    def test_Var_F_positive_when_stochastic(self):
        """Var[F] > 0 when σ² > 0."""
        F_star = 0.9
        delta = 0.1
        sigma2 = 0.05
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert components.Var_F > 0, f"Var[F] should be > 0, got {components.Var_F}"

    def test_L_min_positive_perfect_policy(self):
        """L_min > 0 even with perfect policy (δ=0) when σ² > 0."""
        F_star = 1.0
        delta = 0.0
        sigma2 = 0.1
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert components.L_min > 0, f"L_min should be > 0, got {components.L_min}"

    def test_Bias_squared_non_negative(self):
        """Bias² is always non-negative."""
        test_cases = [
            (0.9, 0.1, 0.05, 1.0),
            (0.85, 0.2, 0.1, 0.5),
            (1.0, 0.0, 0.1, 2.0),
        ]

        for F_star, delta, sigma2, s in test_cases:
            components = compute_theoretical_L_min(F_star, delta, sigma2, s)
            assert components.Bias2 >= 0, f"Bias² should be >= 0, got {components.Bias2}"


class TestIndependentProcesses:
    """Test independent process case (ρ = 0)."""

    def test_variance_formula_independent(self):
        """Var[F] = (1/W²) Σᵢ wᵢ² Var[Qᵢ] for independent processes."""
        process_params = {
            'laser': {'F_star': 0.95, 'delta': 0.05, 'sigma2': 0.01, 's': 0.1},
            'plasma': {'F_star': 0.90, 'delta': 0.1, 'sigma2': 0.02, 's': 2.0},
        }
        process_weights = {'laser': 1.0, 'plasma': 1.0}

        combined, per_process = compute_multi_process_L_min(
            process_params, process_weights, loss_scale=1.0
        )

        # Manual calculation
        W = sum(process_weights.values())
        expected_Var_F = 0.0
        for name, params in process_params.items():
            _, _, Var_Q = compute_per_process_Q_stats(
                params['F_star'], params['delta'], params['sigma2'], params['s']
            )
            w = process_weights[name]
            expected_Var_F += (w**2) * Var_Q

        expected_Var_F /= (W**2)

        assert np.isclose(combined.Var_F, expected_Var_F, rtol=1e-5), \
            f"Var[F] = {combined.Var_F}, expected {expected_Var_F}"

    def test_cross_covariance_zero_independent(self):
        """Cov[Qᵢ, Qⱼ] = 0 for i ≠ j when processes are independent."""
        # This is implicitly tested by the variance formula above
        # When independent, Var[F] = Σ wᵢ² Var[Qᵢ] / W²
        # If there were non-zero covariances, Var[F] would include cross terms
        pass  # Covered by test_variance_formula_independent


class TestMultiProcessConsistency:
    """Test that multi-process case reduces correctly to single process."""

    def test_single_process_equivalence(self):
        """Multi-process with n=1 should match single process calculation."""
        F_star = 0.9
        delta = 0.15
        sigma2 = 0.05
        s = 1.0

        # Single process
        single = compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale=1.0)

        # Multi-process with one process
        process_params = {'only': {'F_star': F_star, 'delta': delta, 'sigma2': sigma2, 's': s}}
        process_weights = {'only': 1.0}
        combined, _ = compute_multi_process_L_min(process_params, process_weights, loss_scale=1.0)

        assert np.isclose(single.L_min, combined.L_min, rtol=1e-5), \
            f"Single L_min = {single.L_min}, Multi L_min = {combined.L_min}"
        assert np.isclose(single.E_F, combined.E_F, rtol=1e-5), \
            f"Single E[F] = {single.E_F}, Multi E[F] = {combined.E_F}"

    def test_equal_weights_symmetric(self):
        """With equal weights and identical processes, combined F* equals individual."""
        params = {'F_star': 0.9, 'delta': 0.1, 'sigma2': 0.03, 's': 1.0}
        process_params = {
            'p1': params.copy(),
            'p2': params.copy(),
        }
        process_weights = {'p1': 1.0, 'p2': 1.0}

        combined, per_process = compute_multi_process_L_min(
            process_params, process_weights, loss_scale=1.0
        )

        # Combined F* should equal individual F*
        assert np.isclose(combined.F_star, params['F_star']), \
            f"Combined F* = {combined.F_star}, expected {params['F_star']}"


class TestNumericalVerification:
    """Verify formulas with manual calculations."""

    def test_numerical_example_1(self):
        """Test with s=1, σ²=0.5, δ=0 (perfect policy)."""
        s = 1.0
        sigma2 = 0.5
        delta = 0.0
        F_star = 1.0

        # Manual calculations
        # E[F] = 1/√(1 + 2*0.5/1) = 1/√2 ≈ 0.7071
        expected_E_F = 1.0 / np.sqrt(2.0)

        # E[F²] = 1/√(1 + 4*0.5/1) = 1/√3 ≈ 0.5774
        expected_E_F2 = 1.0 / np.sqrt(3.0)

        # Var[F] = E[F²] - E[F]² = 1/√3 - 1/2 ≈ 0.0774
        expected_Var = expected_E_F2 - expected_E_F**2

        # Bias = E[F] - F* = 1/√2 - 1 ≈ -0.2929
        expected_Bias = expected_E_F - F_star

        # L_min = Var[F] + Bias² ≈ 0.0774 + 0.0858 = 0.1632
        expected_L_min = expected_Var + expected_Bias**2

        components = compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale=1.0)

        assert np.isclose(components.E_F, expected_E_F, rtol=1e-4), \
            f"E[F] = {components.E_F}, expected {expected_E_F}"
        assert np.isclose(components.E_F2, expected_E_F2, rtol=1e-4), \
            f"E[F²] = {components.E_F2}, expected {expected_E_F2}"
        assert np.isclose(components.Var_F, expected_Var, rtol=1e-4), \
            f"Var[F] = {components.Var_F}, expected {expected_Var}"
        assert np.isclose(components.L_min, expected_L_min, rtol=1e-4), \
            f"L_min = {components.L_min}, expected {expected_L_min}"

    def test_numerical_example_2(self):
        """Test with s=1, σ²=0.25, δ=0.5."""
        s = 1.0
        sigma2 = 0.25
        delta = 0.5
        F_star = np.exp(-delta**2 / s)  # = exp(-0.25) ≈ 0.7788

        # E[F] = F* / √(1 + 2*0.25/1) × exp(2*0.25*0.25 / (1*(1 + 0.5)))
        #      = F* / √1.5 × exp(0.125 / 1.5)
        #      = F* × 0.8165 × 1.0870
        factor1 = 1.0 / np.sqrt(1 + 2 * sigma2 / s)
        exp_arg1 = 2 * delta**2 * sigma2 / (s * (s + 2 * sigma2))
        factor2 = np.exp(exp_arg1)
        expected_E_F = F_star * factor1 * factor2

        # E[F²] = F*² / √(1 + 4*0.25/1) × exp(8*0.25*0.25 / (1*(1 + 1)))
        #       = F*² / √2 × exp(0.5 / 2)
        #       = F*² × 0.7071 × 1.2840
        factor1_sq = 1.0 / np.sqrt(1 + 4 * sigma2 / s)
        exp_arg2 = 8 * delta**2 * sigma2 / (s * (s + 4 * sigma2))
        factor2_sq = np.exp(exp_arg2)
        expected_E_F2 = F_star**2 * factor1_sq * factor2_sq

        components = compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale=1.0)

        assert np.isclose(components.E_F, expected_E_F, rtol=1e-4), \
            f"E[F] = {components.E_F}, expected {expected_E_F}"
        assert np.isclose(components.E_F2, expected_E_F2, rtol=1e-4), \
            f"E[F²] = {components.E_F2}, expected {expected_E_F2}"

    def test_variance_theorem_18_proof(self):
        """Verify Theorem 18: Var[F]|_{δ=0} > 0 for σ² > 0.

        Proof: Define x = 2σ²/s > 0. Must show:
            1/√(1+2x) > 1/(1+x)
        Equivalent to:
            1+x > √(1+2x)
        Squaring: (1+x)² > 1+2x
                  1+2x+x² > 1+2x
                  x² > 0  ✓
        """
        for x in [0.1, 0.5, 1.0, 2.0, 5.0]:
            sigma2 = x / 2  # x = 2σ²/s with s=1
            s = 1.0
            delta = 0.0
            F_star = 1.0

            E_Q, E_Q2, Var_Q = compute_per_process_Q_stats(F_star, delta, sigma2, s)

            # LHS = 1/√(1+2x) = E[F²] when δ=0
            lhs = 1.0 / np.sqrt(1 + 2*x)
            # RHS = 1/(1+x) = (E[F])² when δ=0
            rhs = 1.0 / (1 + x)

            assert lhs > rhs, f"Theorem 18 violated: {lhs} <= {rhs} for x={x}"
            assert Var_Q > 0, f"Var[F] should be > 0, got {Var_Q} for x={x}"


class TestFormulaConsistency:
    """Test that E[F], E[F²], Var[F] are consistent."""

    def test_variance_equals_E_F2_minus_E_F_squared(self):
        """Var[F] = E[F²] - E[F]²."""
        test_cases = [
            (0.9, 0.1, 0.05, 1.0),
            (0.85, 0.2, 0.1, 0.5),
            (1.0, 0.0, 0.2, 2.0),
            (0.7, 0.3, 0.15, 1.5),
        ]

        for F_star, delta, sigma2, s in test_cases:
            E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
            E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)
            computed_Var = E_F2 - E_F**2

            components = compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale=1.0)

            assert np.isclose(components.Var_F, computed_Var, rtol=1e-5), \
                f"Var mismatch: stored={components.Var_F}, computed={computed_Var}"

    def test_L_min_equals_Var_plus_Bias2(self):
        """L_min = Var[F] + Bias²."""
        test_cases = [
            (0.9, 0.1, 0.05, 1.0),
            (0.85, 0.2, 0.1, 0.5),
            (1.0, 0.0, 0.2, 2.0),
        ]

        for F_star, delta, sigma2, s in test_cases:
            components = compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale=1.0)

            computed_L_min = components.Var_F + components.Bias2
            assert np.isclose(components.L_min, computed_L_min, rtol=1e-5), \
                f"L_min mismatch: stored={components.L_min}, computed={computed_L_min}"


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_sigma2(self):
        """Test with very small σ² (near deterministic)."""
        F_star = 0.9
        delta = 0.1
        sigma2 = 1e-10
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert np.isclose(components.E_F, F_star, rtol=1e-3), \
            f"E[F] should be close to F* for small σ²"
        assert components.L_min < 1e-6, "L_min should be very small for tiny σ²"

    def test_large_sigma2(self):
        """Test with large σ² (high uncertainty)."""
        F_star = 0.9
        delta = 0.1
        sigma2 = 10.0
        s = 1.0

        components = compute_theoretical_L_min(F_star, delta, sigma2, s)
        assert components.L_min > 0, "L_min should still be positive"
        assert components.E_F < F_star, "E[F] should be less than F* for large σ²"

    def test_various_scale_parameters(self):
        """Test with different scale parameters s."""
        F_star = 0.9
        delta = 0.1
        sigma2 = 0.1

        for s in [0.1, 0.5, 1.0, 2.0, 5.0]:
            components = compute_theoretical_L_min(F_star, delta, sigma2, s)
            assert components.L_min > 0, f"L_min should be > 0 for s={s}"
            assert 0 < components.E_F <= 1, f"E[F] should be in (0,1] for s={s}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
