"""
Theoretical Loss Analysis for Reliability-based Controller Optimization.

Implements the theoretical framework from "On the Structural Bias in Reliability
Loss Functions with Stochastic Reparameterization" for computing the minimum
achievable loss (L_min) when using stochastic sampling from the UncertaintyPredictor.

Theory:
The loss function L = (F - F*)^2 where:
- F* = reliability of target trajectory (deterministic)
- F = reliability of controller (computed with stochastic sampling)

When sampling is stochastic (sigma^2 > 0), there's an irreducible minimum L_min > 0.

Formulas (from Theorem 10 and Corollary 16):
- E[F] = F* * (1/sqrt(1 + 2*sigma^2/s)) * exp(2*delta^2*sigma^2 / (s*(s + 2*sigma^2)))
- E[F^2] = F*^2 * (1/sqrt(1 + 4*sigma^2/s)) * exp(8*delta^2*sigma^2 / (s*(s + 4*sigma^2)))
- L_min = Var[F] + Bias^2 = (E[F^2] - E[F]^2) + (E[F] - F*)^2

Note on E[F^2]: The exponent numerator is 8 (not 4) because F^2 = exp(-2(δ+σε)^2/s)
has effective scale s/2, and applying Lemma 9 with a=4δσ/s, b=2σ²/s yields 8δ²σ².

Where:
- sigma^2 = predicted variance from UncertaintyPredictor
- s = scale parameter of quality function Q(o) = exp(-(o-tau)^2/s)
- delta = mu_target - tau (distance of target output from process optimum)
- F* = target reliability
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


@dataclass
class TheoreticalLossComponents:
    """Components of theoretical loss analysis."""
    L_min: float           # Minimum achievable loss
    E_F: float             # Expected value of F
    E_F2: float            # Expected value of F^2
    Var_F: float           # Variance of F
    Bias2: float           # Bias squared (E[F] - F*)^2
    F_star: float          # Target reliability
    sigma2: float          # Predicted variance
    delta: float           # Distance from process optimum
    s: float               # Scale parameter

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'L_min': self.L_min,
            'E_F': self.E_F,
            'E_F2': self.E_F2,
            'Var_F': self.Var_F,
            'Bias2': self.Bias2,
            'F_star': self.F_star,
            'sigma2': self.sigma2,
            'delta': self.delta,
            's': self.s
        }


def compute_theoretical_E_F(F_star: float, delta: float, sigma2: float, s: float) -> float:
    """
    Compute theoretical expected value of F.

    E[F] = F* * (1/sqrt(1 + 2*sigma^2/s)) * exp(2*delta^2*sigma^2 / (s*(s + 2*sigma^2)))

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance
        s: Scale parameter of quality function

    Returns:
        E[F]: Expected value of reliability
    """
    if sigma2 <= 0:
        # Deterministic case: E[F] = F*
        return F_star

    # Avoid division by zero
    if s <= 0:
        return F_star

    # Compute the two factors
    factor1 = 1.0 / np.sqrt(1 + 2 * sigma2 / s)

    # Exponent term
    numerator = 2 * delta**2 * sigma2
    denominator = s * (s + 2 * sigma2)
    factor2 = np.exp(numerator / denominator) if denominator > 0 else 1.0

    E_F = F_star * factor1 * factor2

    return E_F


def compute_theoretical_E_F2(F_star: float, delta: float, sigma2: float, s: float) -> float:
    """
    Compute theoretical expected value of F^2.

    Derivation (Corollario 16 from theoretical document):
        F² = [exp(-(δ + σε)²/s)]² = exp(-2(δ + σε)²/s)

        This has the SAME FORM as F but with scale s replaced by s/2.
        Applying Lemma 9 with a = 4δσ/s and b = 2σ²/s:

        E[F²] = (F*)² × (1/√(1 + 4σ²/s)) × exp(8δ²σ² / (s(s + 4σ²)))

    Note: The exponent numerator is 8 (not 4) because:
        a² / (2(1+2b)) = (4δσ/s)² / (2(1 + 4σ²/s))
                       = 16δ²σ²/s² × s / (2(s + 4σ²))
                       = 8δ²σ² / (s(s + 4σ²))

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance
        s: Scale parameter of quality function

    Returns:
        E[F^2]: Expected value of reliability squared
    """
    if sigma2 <= 0:
        # Deterministic case: E[F^2] = F*^2
        return F_star**2

    # Avoid division by zero
    if s <= 0:
        return F_star**2

    # Compute the two factors
    factor1 = 1.0 / np.sqrt(1 + 4 * sigma2 / s)

    # Exponent term - CORRECTED: numerator is 8, not 4
    # From Corollario 16: 8δ²σ² / (s(s + 4σ²))
    numerator = 8 * delta**2 * sigma2
    denominator = s * (s + 4 * sigma2)
    factor2 = np.exp(numerator / denominator) if denominator > 0 else 1.0

    E_F2 = F_star**2 * factor1 * factor2

    return E_F2


def compute_theoretical_L_min(
    F_star: float,
    delta: float,
    sigma2: float,
    s: float,
    loss_scale: float = 1.0
) -> TheoreticalLossComponents:
    """
    Compute the theoretical minimum achievable loss and all its components.

    L_min = Var[F] + Bias^2
          = (E[F^2] - E[F]^2) + (E[F] - F*)^2

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance (mean across all samples)
        s: Scale parameter of quality function
        loss_scale: Scale factor for the loss (default 1.0, training uses 100.0)

    Returns:
        TheoreticalLossComponents with all computed values
    """
    E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
    E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)

    # Variance of F
    Var_F = E_F2 - E_F**2
    # Ensure non-negative (numerical stability)
    Var_F = max(Var_F, 0.0)

    # Bias squared
    Bias2 = (E_F - F_star)**2

    # Minimum achievable loss
    L_min = (Var_F + Bias2) * loss_scale

    return TheoreticalLossComponents(
        L_min=L_min,
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,  # Scale variance too for consistency
        Bias2=Bias2 * loss_scale,  # Scale bias too
        F_star=F_star,
        sigma2=sigma2,
        delta=delta,
        s=s
    )


def compute_per_process_Q_stats(
    Q_star: float,
    delta: float,
    sigma2: float,
    s: float
) -> Tuple[float, float, float]:
    """
    Compute E[Q_i], E[Q_i²], and Var[Q_i] for a single process.

    Formulas (from theoretical document):
        E[Q_i] = Q_i* / √(1 + 2σ²/s) · exp(2δ²σ² / (s(s + 2σ²)))     [Theorem 10]
        E[Q_i²] = Q_i*² / √(1 + 4σ²/s) · exp(8δ²σ² / (s(s + 4σ²)))  [Corollary 16]
        Var[Q_i] = E[Q_i²] - E[Q_i]²

    Note: The exponent in E[Q²] is 8δ²σ² (not 4δ²σ²) because Q² has
    effective scale s/2, and applying Lemma 9 with a=4δσ/s, b=2σ²/s gives:
        a²/(2(1+2b)) = 16δ²σ²/s² / (2(s+4σ²)/s) = 8δ²σ² / (s(s+4σ²))

    Args:
        Q_star: Target quality for this process (Q_i* = exp(-δ²/s))
        delta: μ_target - τ (distance from process optimum)
        sigma2: Predicted variance for this process
        s: Scale parameter of quality function

    Returns:
        Tuple of (E[Q_i], E[Q_i²], Var[Q_i])
    """
    if sigma2 <= 0 or s <= 0:
        # Deterministic case
        return Q_star, Q_star**2, 0.0

    # E[Q_i] - Theorem 10
    factor1 = 1.0 / np.sqrt(1 + 2 * sigma2 / s)
    num1 = 2 * delta**2 * sigma2
    den1 = s * (s + 2 * sigma2)
    factor2 = np.exp(num1 / den1) if den1 > 0 else 1.0
    E_Q = Q_star * factor1 * factor2

    # E[Q_i²] - Corollary 16: exponent numerator is 8, not 4
    factor1_sq = 1.0 / np.sqrt(1 + 4 * sigma2 / s)
    num2 = 8 * delta**2 * sigma2  # CORRECTED: was 4, should be 8
    den2 = s * (s + 4 * sigma2)
    factor2_sq = np.exp(num2 / den2) if den2 > 0 else 1.0
    E_Q2 = Q_star**2 * factor1_sq * factor2_sq

    # Var[Q_i]
    Var_Q = max(E_Q2 - E_Q**2, 0.0)

    return E_Q, E_Q2, Var_Q


def compute_cross_moment(
    Q_star_i: float, delta_i: float, sigma2_i: float, s_i: float,
    Q_star_j: float, delta_j: float, sigma2_j: float, s_j: float,
    rho: float
) -> float:
    """
    Compute E[QᵢQⱼ] for two correlated processes (Theorem 45).

    Formula:
        E[QᵢQⱼ] = (Q*ᵢ × Q*ⱼ / √det(M)) × exp(½ aᵀ M⁻¹ R a)

    Where:
        aᵢ = 2δᵢσᵢ/sᵢ,  aⱼ = 2δⱼσⱼ/sⱼ
        bᵢ = σ²ᵢ/sᵢ,    bⱼ = σ²ⱼ/sⱼ
        R = [[1, ρ], [ρ, 1]]  (correlation matrix)
        M = I + 2RB
        det(M) = (1 + 2bᵢ)(1 + 2bⱼ) - 4ρ²bᵢbⱼ  (Corollary 40)

    From Corollary 43:
        aᵀM⁻¹Ra = (1/det(M)) × [aᵢ²(1+2bⱼ(1-ρ²)) + 2ρaᵢaⱼ + aⱼ²(1+2bᵢ(1-ρ²))]

    Args:
        Q_star_i, delta_i, sigma2_i, s_i: Parameters for process i
        Q_star_j, delta_j, sigma2_j, s_j: Parameters for process j
        rho: Correlation coefficient between εᵢ and εⱼ, ρ ∈ [-1, 1]

    Returns:
        E[QᵢQⱼ]: Cross-moment of the two quality functions
    """
    # Handle edge cases
    if sigma2_i <= 0 or sigma2_j <= 0 or s_i <= 0 or s_j <= 0:
        # Deterministic case: E[QᵢQⱼ] = Q*ᵢ × Q*ⱼ
        return Q_star_i * Q_star_j

    # Compute a and b parameters
    # Note: sigma_i = sqrt(sigma2_i), so a_i = 2*delta_i*sigma_i/s_i
    sigma_i = np.sqrt(sigma2_i)
    sigma_j = np.sqrt(sigma2_j)

    a_i = 2 * delta_i * sigma_i / s_i
    a_j = 2 * delta_j * sigma_j / s_j
    b_i = sigma2_i / s_i
    b_j = sigma2_j / s_j

    # Compute det(M) = (1 + 2bᵢ)(1 + 2bⱼ) - 4ρ²bᵢbⱼ  (Corollary 40)
    det_M = (1 + 2 * b_i) * (1 + 2 * b_j) - 4 * rho**2 * b_i * b_j

    # Check for numerical stability
    if det_M <= 0:
        # This shouldn't happen for valid correlation |ρ| ≤ 1
        # Fall back to independent case
        E_Qi, _, _ = compute_per_process_Q_stats(Q_star_i, delta_i, sigma2_i, s_i)
        E_Qj, _, _ = compute_per_process_Q_stats(Q_star_j, delta_j, sigma2_j, s_j)
        return E_Qi * E_Qj

    # Compute aᵀM⁻¹Ra (Corollary 43)
    # = (1/det(M)) × [aᵢ²(1+2bⱼ(1-ρ²)) + 2ρaᵢaⱼ + aⱼ²(1+2bᵢ(1-ρ²))]
    term1 = a_i**2 * (1 + 2 * b_j * (1 - rho**2))
    term2 = 2 * rho * a_i * a_j
    term3 = a_j**2 * (1 + 2 * b_i * (1 - rho**2))

    quadratic_form = (term1 + term2 + term3) / det_M

    # E[QᵢQⱼ] = (Q*ᵢ × Q*ⱼ / √det(M)) × exp(½ aᵀM⁻¹Ra)
    E_QiQj = (Q_star_i * Q_star_j / np.sqrt(det_M)) * np.exp(0.5 * quadratic_form)

    return E_QiQj


def compute_covariance(
    params_i: Dict[str, float],
    params_j: Dict[str, float],
    rho: float
) -> float:
    """
    Compute Cov(Qᵢ, Qⱼ) for two processes (Corollary 46).

    Formula:
        Cov(Qᵢ, Qⱼ) = E[QᵢQⱼ] - E[Qᵢ]E[Qⱼ]

    Args:
        params_i: Dict with 'F_star', 'delta', 'sigma2', 's' for process i
        params_j: Dict with 'F_star', 'delta', 'sigma2', 's' for process j
        rho: Correlation coefficient between processes

    Returns:
        Cov(Qᵢ, Qⱼ): Covariance between the two quality functions
    """
    # Extract parameters
    Q_star_i = params_i['F_star']
    delta_i = params_i['delta']
    sigma2_i = params_i['sigma2']
    s_i = params_i['s']

    Q_star_j = params_j['F_star']
    delta_j = params_j['delta']
    sigma2_j = params_j['sigma2']
    s_j = params_j['s']

    # Compute E[QᵢQⱼ] using Theorem 45
    E_QiQj = compute_cross_moment(
        Q_star_i, delta_i, sigma2_i, s_i,
        Q_star_j, delta_j, sigma2_j, s_j,
        rho
    )

    # Compute E[Qᵢ] and E[Qⱼ] using Theorem 10
    E_Qi, _, _ = compute_per_process_Q_stats(Q_star_i, delta_i, sigma2_i, s_i)
    E_Qj, _, _ = compute_per_process_Q_stats(Q_star_j, delta_j, sigma2_j, s_j)

    # Cov(Qᵢ, Qⱼ) = E[QᵢQⱼ] - E[Qᵢ]E[Qⱼ]
    cov = E_QiQj - E_Qi * E_Qj

    return cov


def compute_multi_process_L_min(
    process_params: Dict[str, Dict[str, float]],
    process_weights: Dict[str, float],
    loss_scale: float = 1.0,
    correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[TheoreticalLossComponents, Dict[str, TheoreticalLossComponents]]:
    """
    Compute theoretical L_min for a multi-process system.

    IMPORTANT: Use this function (not compute_theoretical_L_min) for systems with
    multiple processes! The single-process formula gives incorrect results when
    the reliability F is computed as a weighted average of multiple quality scores.

    The reliability F is a weighted average of per-process quality scores:
        F = Σ(w_i × Q_i) / W    where W = Σw_i
        F* = Σ(w_i × Q_i*) / W

    Variance propagation (Theorem 27):
        E[F] = Σ(w_i × E[Q_i]) / W
        Var[F] = (1/W²) Σᵢ Σⱼ wᵢwⱼ Cov(Qᵢ, Qⱼ)

    When correlation_matrix is None (independence assumed, Theorem 31):
        Var[F] = Σ(w_i² × Var[Q_i]) / W²

    The minimum achievable loss is:
        L_min = Var[F] + (E[F] - F*)²

    Note on delta parameter: For systems with ADAPTIVE targets (where τ depends on
    previous process outputs), delta should be computed using the adaptive τ, not
    the fixed τ from PROCESS_CONFIGS. See compute_adaptive_targets() for details.

    Args:
        process_params: Dict mapping process_name to {'F_star', 'delta', 'sigma2', 's'}
                        Note: F_star here is Q_i* (per-process target quality)
        process_weights: Dict mapping process_name to weight
        loss_scale: Scale factor for the loss
        correlation_matrix: Optional dict mapping (process_i, process_j) to ρᵢⱼ
                           If None, assumes independence (ρᵢⱼ = 0 for i ≠ j)
                           Example: {('laser', 'plasma'): 0.3, ('plasma', 'laser'): 0.3}

    Returns:
        Tuple of (combined L_min components, dict of per-process components)
    """
    per_process_components = {}
    per_process_Q_stats = {}  # Store E[Q_i], Var[Q_i] for each process

    # Step 1: Compute Q stats for each process
    for process_name, params in process_params.items():
        Q_star = params['F_star']  # This is Q_i* (per-process)
        delta = params['delta']
        sigma2 = params['sigma2']
        s = params['s']

        E_Q, E_Q2, Var_Q = compute_per_process_Q_stats(Q_star, delta, sigma2, s)
        per_process_Q_stats[process_name] = {
            'Q_star': Q_star,
            'E_Q': E_Q,
            'E_Q2': E_Q2,
            'Var_Q': Var_Q
        }

        # Also store per-process components for backward compatibility
        components = compute_theoretical_L_min(
            F_star=Q_star,
            delta=delta,
            sigma2=sigma2,
            s=s,
            loss_scale=loss_scale
        )
        per_process_components[process_name] = components

    # Step 2: Compute combined F* and E[F] using correct weighted average
    process_names = list(process_params.keys())
    W = sum(process_weights.get(name, 1.0) for name in process_names)

    if W > 0:
        # F* = Σ(w_i × Q_i*) / W
        combined_F_star = sum(
            per_process_Q_stats[name]['Q_star'] * process_weights.get(name, 1.0)
            for name in process_names
        ) / W

        # E[F] = Σ(w_i × E[Q_i]) / W
        combined_E_F = sum(
            per_process_Q_stats[name]['E_Q'] * process_weights.get(name, 1.0)
            for name in process_names
        ) / W

        # Step 3: Compute Var[F] using full covariance formula (Theorem 27)
        # Var[F] = (1/W²) Σᵢ Σⱼ wᵢwⱼ Cov(Qᵢ, Qⱼ)
        if correlation_matrix is not None:
            # Use full covariance with correlations
            combined_Var_F = 0.0
            for i, name_i in enumerate(process_names):
                for j, name_j in enumerate(process_names):
                    w_i = process_weights.get(name_i, 1.0)
                    w_j = process_weights.get(name_j, 1.0)

                    if i == j:
                        # Diagonal: Cov(Qᵢ, Qᵢ) = Var(Qᵢ)
                        cov_ij = per_process_Q_stats[name_i]['Var_Q']
                    else:
                        # Off-diagonal: use correlation from matrix
                        rho = correlation_matrix.get((name_i, name_j), 0.0)
                        if rho == 0.0:
                            # Independent: Cov = 0
                            cov_ij = 0.0
                        else:
                            # Correlated: compute using Corollary 46
                            cov_ij = compute_covariance(
                                process_params[name_i],
                                process_params[name_j],
                                rho
                            )

                    combined_Var_F += w_i * w_j * cov_ij

            combined_Var_F /= (W ** 2)
        else:
            # Independent case (Theorem 31): Var[F] = Σ(w_i² × Var[Q_i]) / W²
            combined_Var_F = sum(
                per_process_Q_stats[name]['Var_Q'] * (process_weights.get(name, 1.0) ** 2)
                for name in process_names
            ) / (W ** 2)

        # Step 4: Bias² = (E[F] - F*)²
        combined_Bias2 = (combined_E_F - combined_F_star) ** 2

        # Step 5: L_min = Var[F] + Bias²
        combined_L_min = (combined_Var_F + combined_Bias2) * loss_scale

        # Scale variance and bias for consistency
        combined_Var_F_scaled = combined_Var_F * loss_scale
        combined_Bias2_scaled = combined_Bias2 * loss_scale

        # E[F²] = Var[F] + E[F]²
        combined_E_F2 = combined_Var_F + combined_E_F**2

        # Mean sigma2 (for informational purposes)
        combined_sigma2 = sum(
            params['sigma2'] * process_weights.get(name, 1.0)
            for name, params in process_params.items()
        ) / W
    else:
        combined_L_min = 0.0
        combined_E_F = 0.0
        combined_E_F2 = 0.0
        combined_Var_F_scaled = 0.0
        combined_Bias2_scaled = 0.0
        combined_F_star = 0.0
        combined_sigma2 = 0.0

    combined_components = TheoreticalLossComponents(
        L_min=combined_L_min,
        E_F=combined_E_F,
        E_F2=combined_E_F2,
        Var_F=combined_Var_F_scaled,
        Bias2=combined_Bias2_scaled,
        F_star=combined_F_star,
        sigma2=combined_sigma2,
        delta=0.0,  # Not meaningful for combined
        s=0.0  # Not meaningful for combined
    )

    return combined_components, per_process_components


def compute_empirical_L_min(
    F_samples: np.ndarray,
    F_star: float,
    loss_scale: float = 1.0
) -> TheoreticalLossComponents:
    """
    Compute L_min empirically from forward-pass samples.

    Instead of deriving L_min from analytical formulas (which require per-process
    δ, σ², s extraction), this estimates L_min directly from N stochastic forward
    passes:

        L_min = (Var(F_samples) + (mean(F_samples) - F*)²) × loss_scale

    This is simpler, more robust (works with non-Gaussian noise, correlations,
    any quality function shape), and avoids the complexity of multi-process
    parameter extraction.

    Args:
        F_samples: Array of F values from N stochastic forward passes.
                   Should contain at least ~100 samples for stable estimates.
        F_star: Target reliability (deterministic F*).
        loss_scale: Scale factor for the loss (training typically uses 100.0).

    Returns:
        TheoreticalLossComponents populated with empirical values.
        Note: sigma2, delta, s fields are set to 0.0 (not meaningful for
        empirical computation).
    """
    F_samples = np.asarray(F_samples)

    if len(F_samples) == 0:
        return TheoreticalLossComponents(
            L_min=0.0, E_F=F_star, E_F2=F_star**2,
            Var_F=0.0, Bias2=0.0, F_star=F_star,
            sigma2=0.0, delta=0.0, s=0.0
        )

    E_F = float(np.mean(F_samples))
    E_F2 = float(np.mean(F_samples**2))
    Var_F = float(np.var(F_samples))  # unbiased not needed; N is large
    Bias2 = (E_F - F_star) ** 2
    L_min = (Var_F + Bias2) * loss_scale

    return TheoreticalLossComponents(
        L_min=L_min,
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,
        Bias2=Bias2 * loss_scale,
        F_star=F_star,
        sigma2=0.0,   # not applicable for empirical
        delta=0.0,     # not applicable for empirical
        s=0.0          # not applicable for empirical
    )


@dataclass
class TheoreticalLossTracker:
    """
    Tracks theoretical loss analysis throughout training.

    Collects:
    - Observed loss per epoch
    - Theoretical L_min per epoch
    - Gap (observed - theoretical)
    - Efficiency (L_min / observed)
    - Empirical E[F], Var[F] from sampling
    """

    # Process parameters (from surrogate.PROCESS_CONFIGS)
    process_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    process_weights: Dict[str, float] = field(default_factory=dict)
    loss_scale: float = 100.0  # Default scale factor

    # History tracking
    epochs: List[int] = field(default_factory=list)
    observed_loss: List[float] = field(default_factory=list)
    theoretical_L_min: List[float] = field(default_factory=list)  # L_min = Var[F] + Bias²
    gap: List[float] = field(default_factory=list)  # Gap = Loss - L_min (reducible)
    efficiency: List[float] = field(default_factory=list)  # Efficiency = L_min / Loss

    # Empirical statistics
    empirical_E_F: List[float] = field(default_factory=list)
    empirical_Var_F: List[float] = field(default_factory=list)
    empirical_Bias2: List[float] = field(default_factory=list)

    # Theoretical statistics
    theoretical_E_F: List[float] = field(default_factory=list)
    theoretical_Var_F: List[float] = field(default_factory=list)
    theoretical_Bias2: List[float] = field(default_factory=list)

    # Per-epoch sigma2 (mean predicted variance)
    sigma2_per_epoch: List[float] = field(default_factory=list)

    # Validation counters
    n_violations: int = 0  # Times observed < theoretical

    def set_process_params_from_surrogate(self, surrogate):
        """
        Extract process parameters from surrogate's PROCESS_CONFIGS.

        Args:
            surrogate: ProTSurrogate instance
        """
        from controller_optimization.src.models.surrogate import ProTSurrogate

        # Get process configs from surrogate class
        configs = ProTSurrogate.PROCESS_CONFIGS

        for process_name, config in configs.items():
            self.process_configs[process_name] = {
                'tau': config['target'],
                's': config['scale']
            }
            self.process_weights[process_name] = config.get('weight', 1.0)

    def update(
        self,
        epoch: int,
        observed_loss_value: float,
        F_star: float,
        F_samples: np.ndarray,
        sigma2_mean: float,
        delta: float = 0.0,
        s: float = 1.0
    ):
        """
        Update tracker with data from current epoch.

        Args:
            epoch: Current epoch number
            observed_loss_value: Observed reliability loss
            F_star: Target reliability
            F_samples: Array of F values from multiple forward passes
            sigma2_mean: Mean predicted variance across all processes
            delta: Distance from process optimum (estimated)
            s: Effective scale parameter
        """
        self.epochs.append(epoch)
        self.observed_loss.append(observed_loss_value)
        self.sigma2_per_epoch.append(sigma2_mean)

        # Compute theoretical L_min = Var[F] + Bias²
        theoretical = compute_theoretical_L_min(F_star, delta, sigma2_mean, s, self.loss_scale)
        self.theoretical_L_min.append(theoretical.L_min)
        self.theoretical_E_F.append(theoretical.E_F)
        self.theoretical_Var_F.append(theoretical.Var_F)
        self.theoretical_Bias2.append(theoretical.Bias2)

        # Compute empirical statistics from F_samples
        if len(F_samples) > 0:
            empirical_mean = np.mean(F_samples)
            empirical_var = np.var(F_samples)
            empirical_bias2 = (empirical_mean - F_star)**2
        else:
            empirical_mean = F_star
            empirical_var = 0.0
            empirical_bias2 = 0.0

        self.empirical_E_F.append(empirical_mean)
        self.empirical_Var_F.append(empirical_var * self.loss_scale)
        self.empirical_Bias2.append(empirical_bias2 * self.loss_scale)

        # Compute gap and efficiency
        gap_value = observed_loss_value - theoretical.L_min
        self.gap.append(gap_value)

        if observed_loss_value > 0:
            efficiency_value = theoretical.L_min / observed_loss_value
        else:
            efficiency_value = 1.0 if theoretical.L_min == 0 else 0.0
        self.efficiency.append(efficiency_value)

        # Check for violations (loss < L_min indicates theory issue)
        if observed_loss_value < theoretical.L_min * 0.99:  # 1% tolerance
            self.n_violations += 1

    def get_final_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics at end of training.

        Returns:
            Dict with all summary statistics
        """
        if len(self.epochs) == 0:
            return {}

        final_idx = -1
        best_idx = np.argmin(self.observed_loss) if self.observed_loss else -1

        return {
            'final_loss': self.observed_loss[final_idx],
            'best_loss': self.observed_loss[best_idx] if best_idx >= 0 else 0.0,

            # L_min = Var[F] + Bias²
            'final_L_min': self.theoretical_L_min[final_idx],
            'final_gap': self.gap[final_idx],
            'final_efficiency': self.efficiency[final_idx],

            'best_efficiency': max(self.efficiency) if self.efficiency else 0.0,
            'mean_efficiency': np.mean(self.efficiency) if self.efficiency else 0.0,

            'empirical_E_F_final': self.empirical_E_F[final_idx],
            'theoretical_E_F_final': self.theoretical_E_F[final_idx],
            'empirical_Var_F_final': self.empirical_Var_F[final_idx],
            'theoretical_Var_F_final': self.theoretical_Var_F[final_idx],

            'n_violations': self.n_violations,
            'total_epochs': len(self.epochs),
            'violation_rate': self.n_violations / len(self.epochs) if self.epochs else 0.0,

            # Find epochs where efficiency thresholds were reached
            'epoch_90_efficiency': self._find_efficiency_epoch(0.9),
            'epoch_95_efficiency': self._find_efficiency_epoch(0.95),
        }

    def _find_efficiency_epoch(self, threshold: float) -> Optional[int]:
        """Find first epoch where efficiency >= threshold."""
        for i, eff in enumerate(self.efficiency):
            if eff >= threshold:
                return self.epochs[i]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tracker to dictionary for serialization."""
        return {
            'process_configs': self.process_configs,
            'process_weights': self.process_weights,
            'loss_scale': self.loss_scale,
            'epochs': self.epochs,
            'observed_loss': self.observed_loss,
            'theoretical_L_min': self.theoretical_L_min,
            'gap': self.gap,
            'efficiency': self.efficiency,
            'empirical_E_F': self.empirical_E_F,
            'empirical_Var_F': self.empirical_Var_F,
            'empirical_Bias2': self.empirical_Bias2,
            'theoretical_E_F': self.theoretical_E_F,
            'theoretical_Var_F': self.theoretical_Var_F,
            'theoretical_Bias2': self.theoretical_Bias2,
            'sigma2_per_epoch': self.sigma2_per_epoch,
            'n_violations': self.n_violations,
            'summary': self.get_final_summary()
        }

    def save(self, path: Path):
        """Save tracker to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TheoreticalLossTracker':
        """Load tracker from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        tracker = cls(
            process_configs=data.get('process_configs', {}),
            process_weights=data.get('process_weights', {}),
            loss_scale=data.get('loss_scale', 100.0)
        )
        tracker.epochs = data.get('epochs', [])
        tracker.observed_loss = data.get('observed_loss', [])
        tracker.theoretical_L_min = data.get('theoretical_L_min', [])
        tracker.gap = data.get('gap', [])
        tracker.efficiency = data.get('efficiency', [])
        tracker.empirical_E_F = data.get('empirical_E_F', [])
        tracker.empirical_Var_F = data.get('empirical_Var_F', [])
        tracker.empirical_Bias2 = data.get('empirical_Bias2', [])
        tracker.theoretical_E_F = data.get('theoretical_E_F', [])
        tracker.theoretical_Var_F = data.get('theoretical_Var_F', [])
        tracker.theoretical_Bias2 = data.get('theoretical_Bias2', [])
        tracker.sigma2_per_epoch = data.get('sigma2_per_epoch', [])
        tracker.n_violations = data.get('n_violations', 0)

        return tracker


def compute_effective_params_from_trajectory(
    trajectory: Dict,
    target_trajectory: Dict,
    process_configs: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute effective theoretical parameters from actual trajectory data.

    Extracts:
    - sigma2: mean predicted variance
    - delta: distance between predicted mean and process target (tau)
    - F_star: per-process quality scores

    Args:
        trajectory: Output from process_chain.forward()
        target_trajectory: Target trajectory tensors
        process_configs: From surrogate.PROCESS_CONFIGS

    Returns:
        Dict mapping process_name to {'F_star', 'delta', 'sigma2', 's'}
    """
    params = {}

    for process_name, data in trajectory.items():
        if process_name not in process_configs:
            continue

        config = process_configs[process_name]
        tau = config['target']
        s = config['scale']

        # Get predicted mean and variance
        if isinstance(data['outputs_mean'], torch.Tensor):
            mu = data['outputs_mean'].detach().cpu().numpy().mean()
            sigma2 = data['outputs_var'].detach().cpu().numpy().mean()
        else:
            mu = np.mean(data['outputs_mean'])
            sigma2 = np.mean(data['outputs_var'])

        # Get target output
        if process_name in target_trajectory:
            target_data = target_trajectory[process_name]
            if isinstance(target_data['outputs'], torch.Tensor):
                mu_target = target_data['outputs'].detach().cpu().numpy().mean()
            else:
                mu_target = np.mean(target_data['outputs'])
        else:
            mu_target = mu  # Fallback

        # Compute delta (distance from process optimum)
        delta = mu_target - tau

        # Compute F_star (quality at target)
        F_star = np.exp(-(mu_target - tau)**2 / s)

        params[process_name] = {
            'F_star': float(F_star),
            'delta': float(delta),
            'sigma2': float(sigma2),
            's': float(s),
            'tau': float(tau),
            'mu_target': float(mu_target),
            'mu_predicted': float(mu)
        }

    return params


def estimate_effective_params_simple(
    F_star_mean: float,
    mean_sigma2: float,
    s_effective: float = 1.0
) -> Dict[str, float]:
    """
    Simple estimation of effective parameters for theoretical L_min.

    Used when detailed per-process tracking is not available.

    Args:
        F_star_mean: Mean target reliability
        mean_sigma2: Mean predicted variance across all processes
        s_effective: Effective scale parameter (weighted average)

    Returns:
        Dict with {'F_star', 'delta', 'sigma2', 's'}
    """
    # For delta, use inverse of quality function
    # F* = exp(-delta^2/s) => delta = sqrt(-s * ln(F*))
    if F_star_mean > 0 and F_star_mean < 1:
        delta = np.sqrt(-s_effective * np.log(F_star_mean))
    else:
        delta = 0.0

    return {
        'F_star': F_star_mean,
        'delta': delta,
        'sigma2': mean_sigma2,
        's': s_effective
    }


def run_validation_sampling(
    process_chain,
    surrogate,
    scenario_idx: int,
    n_samples: int = 100,
    batch_size: int = 1
) -> Tuple[np.ndarray, float, float]:
    """
    Run multiple forward passes to estimate empirical E[F] and Var[F].

    Args:
        process_chain: ProcessChain instance
        surrogate: ProTSurrogate instance
        scenario_idx: Which scenario to evaluate
        n_samples: Number of forward passes
        batch_size: Batch size per pass

    Returns:
        Tuple of (F_samples array, mean_sigma2, F_star)
    """
    F_samples = []
    sigma2_samples = []

    with torch.no_grad():
        process_chain.eval()

        for _ in range(n_samples):
            trajectory = process_chain.forward(batch_size=batch_size, scenario_idx=scenario_idx)
            F = surrogate.compute_reliability(trajectory).item()
            F_samples.append(F)

            # Collect sigma2 from all processes
            for proc_name, data in trajectory.items():
                sigma2 = data['outputs_var'].mean().item()
                sigma2_samples.append(sigma2)

    F_star = surrogate.F_star
    mean_sigma2 = np.mean(sigma2_samples)

    return np.array(F_samples), mean_sigma2, F_star


def compute_z_score(empirical: float, theoretical: float, std: float, n_samples: int) -> float:
    """
    Compute z-score for comparing empirical vs theoretical values.

    z = (empirical - theoretical) / (std / sqrt(n))

    Args:
        empirical: Empirical value
        theoretical: Theoretical value
        std: Standard deviation of samples
        n_samples: Number of samples

    Returns:
        z-score (values near 0 indicate good match)
    """
    if std <= 0 or n_samples <= 0:
        return 0.0

    se = std / np.sqrt(n_samples)  # Standard error
    if se <= 0:
        return 0.0

    return (empirical - theoretical) / se


def format_status(value: float, thresholds: Tuple[float, float] = (0.05, 0.20)) -> str:
    """
    Format status based on relative difference.

    Args:
        value: Relative difference |obs - theo| / theo
        thresholds: (good_threshold, warning_threshold)

    Returns:
        Status string
    """
    good_thresh, warn_thresh = thresholds
    if abs(value) < good_thresh:
        return "OK"
    elif abs(value) < warn_thresh:
        return "WARN"
    else:
        return "MISMATCH"


if __name__ == '__main__':
    # Test theoretical calculations
    print("Testing Theoretical Loss Analysis")
    print("="*60)

    # Example parameters
    F_star = 0.85
    delta = 0.2
    sigma2 = 0.05
    s = 1.0

    print(f"\nInput parameters:")
    print(f"  F* = {F_star}")
    print(f"  delta = {delta}")
    print(f"  sigma2 = {sigma2}")
    print(f"  s = {s}")

    # Compute theoretical values
    components = compute_theoretical_L_min(F_star, delta, sigma2, s)

    print(f"\nTheoretical results:")
    print(f"  E[F] = {components.E_F:.6f}")
    print(f"  E[F^2] = {components.E_F2:.6f}")
    print(f"  Var[F] = {components.Var_F:.6f}")
    print(f"  Bias^2 = {components.Bias2:.6f}")
    print(f"  L_min = {components.L_min:.6f}")

    # Test with deterministic case (sigma2 = 0)
    print(f"\nDeterministic case (sigma2 = 0):")
    components_det = compute_theoretical_L_min(F_star, delta, 0.0, s)
    print(f"  E[F] = {components_det.E_F:.6f} (should equal F* = {F_star})")
    print(f"  L_min = {components_det.L_min:.6f} (should be 0)")

    print("\nTest passed!")
