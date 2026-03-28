"""
Tests for compute_empirical_lmin and compute_training_efficiency_emp.
"""

import warnings

import pytest
import torch

from controller_optimization.src.utils.metrics import (
    compute_empirical_lmin,
    compute_training_efficiency_emp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_deterministic_predictor(output_dim=2):
    """Uncertainty predictor that returns fixed mu with zero variance."""
    def predictor(a_t):
        mu = a_t[:output_dim] if a_t.dim() == 1 else a_t[..., :output_dim]
        var = torch.zeros_like(mu)
        return mu, var
    return predictor


def make_noisy_predictor(sigma_sq, output_dim=2):
    """Uncertainty predictor with constant variance sigma_sq."""
    def predictor(a_t):
        mu = a_t[:output_dim] if a_t.dim() == 1 else a_t[..., :output_dim]
        var = torch.full_like(mu, sigma_sq)
        return mu, var
    return predictor


def make_constant_surrogate(value):
    """Surrogate that always returns a fixed scalar."""
    def surrogate(trajectory):
        return torch.tensor(value, dtype=torch.float32)
    return surrogate


def make_mean_surrogate():
    """Surrogate that returns mean of all trajectory values."""
    def surrogate(trajectory):
        return trajectory.mean()
    return surrogate


# ---------------------------------------------------------------------------
# Tests for compute_empirical_lmin
# ---------------------------------------------------------------------------

class TestComputeEmpiricalLmin:

    def test_lmin_non_negative(self):
        """L̂_min must always be >= 0."""
        target_traj = [torch.tensor([0.5, 0.3]) for _ in range(4)]
        predictor = make_noisy_predictor(sigma_sq=0.1)
        surrogate = make_mean_surrogate()
        lmin = compute_empirical_lmin(predictor, surrogate, target_traj,
                                      F_star=0.5, N=200)
        assert lmin >= 0.0

    def test_lmin_zero_deterministic_perfect(self):
        """With σ²=0 and surrogate always returning F*, L̂_min must be 0."""
        F_star = 0.85
        target_traj = [torch.tensor([0.5, 0.3]) for _ in range(4)]
        predictor = make_deterministic_predictor()
        surrogate = make_constant_surrogate(F_star)
        lmin = compute_empirical_lmin(predictor, surrogate, target_traj,
                                      F_star=F_star, N=200)
        assert lmin == pytest.approx(0.0, abs=1e-9)

    def test_lmin_increases_with_noise(self):
        """Higher σ² should produce higher L̂_min (harder dataset)."""
        target_traj = [torch.tensor([0.5, 0.3]) for _ in range(4)]
        surrogate = make_mean_surrogate()
        F_star = 0.4

        lmin_low = compute_empirical_lmin(
            make_noisy_predictor(sigma_sq=0.001), surrogate,
            target_traj, F_star=F_star, N=500)
        lmin_high = compute_empirical_lmin(
            make_noisy_predictor(sigma_sq=1.0), surrogate,
            target_traj, F_star=F_star, N=500)

        assert lmin_high > lmin_low

    def test_low_n_warning(self):
        """N < 100 should raise a warning about high variance."""
        target_traj = [torch.tensor([0.5, 0.3]) for _ in range(2)]
        predictor = make_deterministic_predictor()
        surrogate = make_constant_surrogate(0.5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_empirical_lmin(predictor, surrogate, target_traj,
                                   F_star=0.5, N=50)
            assert any("high variance" in str(warning.message) for warning in w)

    def test_reproducibility(self):
        """Two calls with same inputs should return the same result."""
        target_traj = [torch.tensor([0.5, 0.3]) for _ in range(4)]
        predictor = make_noisy_predictor(sigma_sq=0.1)
        surrogate = make_mean_surrogate()

        lmin1 = compute_empirical_lmin(predictor, surrogate, target_traj,
                                       F_star=0.4, N=300)
        lmin2 = compute_empirical_lmin(predictor, surrogate, target_traj,
                                       F_star=0.4, N=300)
        assert lmin1 == pytest.approx(lmin2, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests for compute_training_efficiency_emp
# ---------------------------------------------------------------------------

class TestComputeTrainingEfficiencyEmp:

    def test_eta_in_range(self):
        """η_emp must lie in [0, 1]."""
        eta = compute_training_efficiency_emp(L_phi=0.5, L_hat_min=0.1)
        assert 0.0 <= eta <= 1.0

    def test_eta_one_when_equal(self):
        """η_emp = 1 when L(Φ) = L̂_min."""
        eta = compute_training_efficiency_emp(L_phi=0.3, L_hat_min=0.3)
        assert eta == pytest.approx(1.0)

    def test_eta_clipped_to_one(self):
        """η_emp should be clipped to 1 even if L̂_min > L(Φ)."""
        eta = compute_training_efficiency_emp(L_phi=0.1, L_hat_min=0.5)
        assert eta == pytest.approx(1.0)

    def test_eta_zero_when_lmin_zero(self):
        """η_emp = 0 when L̂_min = 0."""
        eta = compute_training_efficiency_emp(L_phi=0.5, L_hat_min=0.0)
        assert eta == pytest.approx(0.0)

    def test_eta_raises_on_non_positive_lphi(self):
        """Should raise ValueError for non-positive L_phi."""
        with pytest.raises(ValueError):
            compute_training_efficiency_emp(L_phi=0.0, L_hat_min=0.1)
        with pytest.raises(ValueError):
            compute_training_efficiency_emp(L_phi=-0.1, L_hat_min=0.1)
