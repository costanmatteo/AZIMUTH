"""
Tests for the multi-mode adaptive target computation.

Verifies that _apply_adaptive_mode() produces correct shapes, values,
and gradients for all 6 supported modes.
"""

import math
import pytest
import torch

from scm_ds.compute_reliability import _apply_adaptive_mode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def delta():
    """A small batch of delta values requiring grad."""
    return torch.tensor([-1.0, 0.0, 0.5, 2.0], requires_grad=True)


@pytest.fixture
def coeff():
    return 0.3


# ---------------------------------------------------------------------------
# Shape and value tests — one per mode
# ---------------------------------------------------------------------------

class TestLinearMode:
    def test_value(self, delta, coeff):
        result = _apply_adaptive_mode(delta, coeff, 'linear', {})
        expected = coeff * delta
        assert result.shape == delta.shape
        assert torch.allclose(result, expected)


class TestPolynomialMode:
    def test_value(self, delta, coeff):
        coeff2 = 0.1
        result = _apply_adaptive_mode(delta, coeff, 'polynomial', {'coeff2': coeff2})
        expected = coeff * delta + coeff2 * delta ** 2
        assert result.shape == delta.shape
        assert torch.allclose(result, expected)

    def test_default_coeff2(self, delta, coeff):
        """coeff2 defaults to 0 → same as linear."""
        result = _apply_adaptive_mode(delta, coeff, 'polynomial', {})
        expected = coeff * delta
        assert torch.allclose(result, expected)


class TestPowerMode:
    def test_value(self, delta, coeff):
        alpha = 0.5
        result = _apply_adaptive_mode(delta, coeff, 'power', {'alpha': alpha})
        expected = coeff * torch.sign(delta) * (torch.abs(delta) + 1e-8) ** alpha
        assert result.shape == delta.shape
        assert torch.allclose(result, expected)

    def test_default_alpha(self, delta, coeff):
        result = _apply_adaptive_mode(delta, coeff, 'power', {})
        expected = coeff * torch.sign(delta) * (torch.abs(delta) + 1e-8) ** 0.5
        assert torch.allclose(result, expected)


class TestSoftplusMode:
    def test_value(self, delta, coeff):
        k = 2.0
        result = _apply_adaptive_mode(delta, coeff, 'softplus', {'k': k})
        expected = (1.0 / k) * torch.log(1.0 + torch.exp(k * coeff * delta))
        assert result.shape == delta.shape
        assert torch.allclose(result, expected)

    def test_always_positive(self, delta, coeff):
        result = _apply_adaptive_mode(delta, coeff, 'softplus', {'k': 2.0})
        assert (result >= 0).all()


class TestDeadbandMode:
    def test_value(self, coeff):
        band = 0.5
        d = torch.tensor([-2.0, -0.3, 0.0, 0.3, 2.0], requires_grad=True)
        result = _apply_adaptive_mode(d, coeff, 'deadband', {'band': band})
        abs_d = torch.abs(d)
        expected = coeff * torch.clamp(abs_d - band, min=0.0) * torch.sign(d)
        assert result.shape == d.shape
        assert torch.allclose(result, expected)

    def test_zero_inside_band(self, coeff):
        """Values with |delta| < band must produce zero adjustment."""
        band = 1.0
        d = torch.tensor([-0.5, 0.0, 0.9], requires_grad=True)
        result = _apply_adaptive_mode(d, coeff, 'deadband', {'band': band})
        assert torch.allclose(result, torch.zeros_like(result))


class TestTanhMode:
    def test_value(self, delta, coeff):
        max_shift = 1.5
        result = _apply_adaptive_mode(delta, coeff, 'tanh', {'max_shift': max_shift})
        expected = max_shift * torch.tanh(coeff * delta / max_shift)
        assert result.shape == delta.shape
        assert torch.allclose(result, expected)

    def test_bounded_by_max_shift(self):
        """Output magnitude must never exceed max_shift."""
        max_shift = 0.5
        d = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0], requires_grad=True)
        result = _apply_adaptive_mode(d, 1.0, 'tanh', {'max_shift': max_shift})
        assert (torch.abs(result) <= max_shift + 1e-7).all()


# ---------------------------------------------------------------------------
# Differentiability tests — all modes must support torch.autograd
# ---------------------------------------------------------------------------

_MODE_PARAMS = {
    'linear':     {},
    'polynomial': {'coeff2': 0.1},
    'power':      {'alpha': 0.5},
    'softplus':   {'k': 2.0},
    'deadband':   {'band': 0.5},
    'tanh':       {'max_shift': 1.5},
}


@pytest.mark.parametrize("mode,params", list(_MODE_PARAMS.items()))
class TestDifferentiability:
    def test_grad_flows(self, mode, params):
        """Verify backward pass produces finite gradients."""
        d = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)
        result = _apply_adaptive_mode(d, 0.3, mode, params)
        loss = result.sum()
        loss.backward()
        assert d.grad is not None
        assert torch.isfinite(d.grad).all(), f"Non-finite grad for mode={mode}"


# ---------------------------------------------------------------------------
# Backward compatibility — linear matches old hardcoded behavior exactly
# ---------------------------------------------------------------------------

class TestLinearBackwardCompat:
    def test_matches_old_formula(self):
        """linear mode must produce exactly coeff * delta."""
        d = torch.randn(16, requires_grad=True)
        coeff = 0.42
        result = _apply_adaptive_mode(d, coeff, 'linear', {})
        expected = coeff * d
        assert torch.allclose(result, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Unknown mode raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownMode:
    def test_raises(self, delta, coeff):
        with pytest.raises(ValueError, match="Unknown adaptive_mode"):
            _apply_adaptive_mode(delta, coeff, 'invalid_mode', {})
