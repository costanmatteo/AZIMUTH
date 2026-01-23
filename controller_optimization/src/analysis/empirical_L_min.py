"""
Empirical L_min Calculation for Sequential Processes with Adaptive Targets.

This module implements Monte Carlo estimation of the minimum achievable loss (L_min)
for systems where analytical formulas are not applicable due to sequential process
dependencies and adaptive targets.

Background:
-----------
The AZIMUTH system has SEQUENTIAL processes with ADAPTIVE TARGETS:

    laser -> plasma -> galvanic -> microetch

    tau_plasma = 3.0 + 0.2 * (o_laser - 0.8)
    tau_galvanic = 10.0 + 0.5 * (o_plasma - 5.0) + 0.4 * (o_laser - 0.5)
    tau_microetch = 20.0 + 1.5 * (o_laser - 0.5) + 0.3 * (o_plasma - 5.0) - 0.15 * (o_galvanic - 10.0)

The analytical formulas from Jensen (Theorem 10, Corollary 16) assume that
delta = mu - tau is a DETERMINISTIC CONSTANT. But with adaptive targets:

    delta_plasma = mu_plasma - tau_plasma(o_laser)
                 = mu_plasma - tau_plasma(mu_laser + sigma_laser * eps_laser)

delta DEPENDS on eps_laser, so it's a RANDOM VARIABLE, not a constant!
This makes the analytical formulas INVALID for AZIMUTH.

Empirical Approach:
-------------------
Instead of computing L_min analytically, we MEASURE it through Monte Carlo sampling:

1. Run N forward passes with stochastic sampling (independent eps ~ N(0,1) each time)
2. Each pass generates a reliability realization F_k
3. Compute empirical statistics:
   - E[F] ~ (1/N) * sum(F_k)
   - Var[F] ~ (1/(N-1)) * sum((F_k - E[F])^2)
4. L_min_empirical = Var[F] + (E[F] - F*)^2

This correctly captures all sequential dependencies that analytical formulas miss.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time


@dataclass
class EmpiricalLminResult:
    """Results from empirical L_min computation."""
    L_min: float              # Minimum achievable loss (empirical, unscaled)
    E_F: float                # E[F] - expected reliability
    Var_F: float              # Var[F] - variance of reliability (unscaled)
    Bias2: float              # (E[F] - F*)^2 - squared bias (unscaled)
    F_star: float             # Target reliability
    n_samples: int            # Number of Monte Carlo samples used
    std_error_E_F: float      # Standard error of E[F] estimate
    std_error_Var_F: float    # Standard error of Var[F] estimate
    confidence_interval: Tuple[float, float]  # 95% CI for L_min
    computation_time: float   # Time taken in seconds
    F_samples: Optional[np.ndarray] = None  # Raw samples (optional, for diagnostics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'L_min': float(self.L_min),
            'E_F': float(self.E_F),
            'Var_F': float(self.Var_F),
            'Bias2': float(self.Bias2),
            'F_star': float(self.F_star),
            'n_samples': int(self.n_samples),
            'std_error_E_F': float(self.std_error_E_F),
            'std_error_Var_F': float(self.std_error_Var_F),
            'confidence_interval': [float(self.confidence_interval[0]), float(self.confidence_interval[1])],
            'computation_time': float(self.computation_time)
        }


def compute_empirical_L_min(
    process_chain,
    surrogate,
    F_star: float,
    n_samples: int = 1000,
    batch_size: int = 1,
    scenario_idx: int = 0,
    device: str = 'cpu',
    return_samples: bool = False,
    verbose: bool = True
) -> EmpiricalLminResult:
    """
    Compute L_min empirically via Monte Carlo sampling.

    This function runs many forward passes through the process chain,
    each with independent stochastic sampling (eps ~ N(0,1) for each process).
    It then estimates E[F], Var[F], and L_min from the collected samples.

    Args:
        process_chain: ProcessChain instance with trained controller
        surrogate: ProTSurrogate instance for computing reliability
        F_star: Target reliability (computed deterministically from target trajectory)
        n_samples: Number of Monte Carlo samples (default: 1000)
        batch_size: Batch size for forward passes (default: 1)
        scenario_idx: Which scenario to evaluate (default: 0)
        device: Device for computation (default: 'cpu')
        return_samples: Whether to include raw F samples in result (default: False)
        verbose: Whether to print progress (default: True)

    Returns:
        EmpiricalLminResult with all computed statistics

    Theory:
        L_min = Var[F] + (E[F] - F*)^2

        Where:
        - F = reliability computed with stochastic sampling
        - F* = target reliability (deterministic)
        - Var[F] = irreducible variance due to stochastic sampling
        - (E[F] - F*)^2 = squared bias
    """
    if verbose:
        print(f"\n{'='*60}")
        print("EMPIRICAL L_min COMPUTATION (Monte Carlo)")
        print(f"{'='*60}")
        print(f"  Samples: {n_samples}")
        print(f"  F*: {F_star:.6f}")
        print(f"  Scenario: {scenario_idx}")

    start_time = time.time()
    F_samples = []

    # Ensure process chain is in eval mode but with stochastic sampling
    process_chain.eval()

    # Temporarily enable stochastic sampling in surrogate
    original_deterministic = getattr(surrogate, 'use_deterministic_sampling', True)
    surrogate.use_deterministic_sampling = False

    # Collect samples
    try:
        with torch.no_grad():
            for i in range(n_samples):
                # Forward pass with stochastic sampling
                # Each call generates new eps for each process via outputs_sampled
                trajectory = process_chain.forward(
                    batch_size=batch_size,
                    scenario_idx=scenario_idx
                )

                # Compute reliability for this realization
                F = surrogate.compute_reliability(trajectory)

                # Handle tensor output
                if isinstance(F, torch.Tensor):
                    F = F.item()

                F_samples.append(F)

                # Progress update
                if verbose and n_samples >= 10 and (i + 1) % (n_samples // 10) == 0:
                    print(f"  Progress: {i+1}/{n_samples} samples collected")
    finally:
        # Restore original sampling mode
        surrogate.use_deterministic_sampling = original_deterministic

    # Convert to numpy array
    F_samples = np.array(F_samples)

    # Compute statistics
    E_F = np.mean(F_samples)
    Var_F = np.var(F_samples, ddof=1)  # ddof=1 for unbiased estimator
    Bias2 = (E_F - F_star) ** 2
    L_min = Var_F + Bias2

    # Standard errors
    std_F = np.std(F_samples, ddof=1)
    std_error_E_F = std_F / np.sqrt(n_samples)

    # Standard error of variance estimator (assuming normality)
    # SE(Var) ~ Var * sqrt(2/(n-1))
    if n_samples > 1:
        std_error_Var_F = Var_F * np.sqrt(2 / (n_samples - 1))
    else:
        std_error_Var_F = 0.0

    # 95% Confidence interval for L_min (approximate, using delta method)
    # This is a rough approximation
    L_min_std_error = np.sqrt(std_error_Var_F**2 + (2 * abs(E_F - F_star) * std_error_E_F)**2)
    confidence_interval = (
        L_min - 1.96 * L_min_std_error,
        L_min + 1.96 * L_min_std_error
    )

    computation_time = time.time() - start_time

    if verbose:
        print(f"\n  Results:")
        print(f"    E[F]:       {E_F:.6f} +/- {std_error_E_F:.6f}")
        print(f"    Var[F]:     {Var_F:.8f} +/- {std_error_Var_F:.8f}")
        print(f"    Bias^2:     {Bias2:.8f}")
        print(f"    L_min:      {L_min:.8f}")
        print(f"    95% CI:     [{confidence_interval[0]:.8f}, {confidence_interval[1]:.8f}]")
        print(f"    Time:       {computation_time:.2f}s")
        print(f"{'='*60}\n")

    return EmpiricalLminResult(
        L_min=L_min,
        E_F=E_F,
        Var_F=Var_F,
        Bias2=Bias2,
        F_star=F_star,
        n_samples=n_samples,
        std_error_E_F=std_error_E_F,
        std_error_Var_F=std_error_Var_F,
        confidence_interval=confidence_interval,
        computation_time=computation_time,
        F_samples=F_samples if return_samples else None
    )


def compute_empirical_L_min_multi_scenario(
    process_chain,
    surrogate,
    F_star_per_scenario: Dict[int, float],
    n_samples_per_scenario: int = 500,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[int, EmpiricalLminResult]:
    """
    Compute empirical L_min for multiple scenarios.

    Args:
        process_chain: ProcessChain instance
        surrogate: ProTSurrogate instance
        F_star_per_scenario: Dict mapping scenario_idx to F* value
        n_samples_per_scenario: Samples per scenario
        device: Computation device
        verbose: Print progress

    Returns:
        Dict mapping scenario_idx to EmpiricalLminResult
    """
    results = {}

    total_scenarios = len(F_star_per_scenario)
    for idx, (scenario_idx, F_star) in enumerate(F_star_per_scenario.items()):
        if verbose:
            print(f"\n--- Scenario {scenario_idx} ({idx+1}/{total_scenarios}) ---")

        results[scenario_idx] = compute_empirical_L_min(
            process_chain=process_chain,
            surrogate=surrogate,
            F_star=F_star,
            n_samples=n_samples_per_scenario,
            scenario_idx=scenario_idx,
            device=device,
            verbose=verbose
        )

    return results


def compute_aggregate_empirical_L_min(
    scenario_results: Dict[int, EmpiricalLminResult],
    weights: Optional[Dict[int, float]] = None
) -> EmpiricalLminResult:
    """
    Aggregate L_min results across scenarios.

    Args:
        scenario_results: Results per scenario
        weights: Optional weights per scenario (uniform if None)

    Returns:
        Aggregated EmpiricalLminResult
    """
    if weights is None:
        weights = {k: 1.0 for k in scenario_results.keys()}

    W = sum(weights.values())

    # Weighted averages
    L_min = sum(weights[k] * r.L_min for k, r in scenario_results.items()) / W
    E_F = sum(weights[k] * r.E_F for k, r in scenario_results.items()) / W
    Var_F = sum(weights[k] * r.Var_F for k, r in scenario_results.items()) / W
    Bias2 = sum(weights[k] * r.Bias2 for k, r in scenario_results.items()) / W
    F_star = sum(weights[k] * r.F_star for k, r in scenario_results.items()) / W

    total_samples = sum(r.n_samples for r in scenario_results.values())
    total_time = sum(r.computation_time for r in scenario_results.values())

    # Approximate aggregated standard errors (weighted combination)
    std_error_E_F = np.sqrt(sum((weights[k]/W * r.std_error_E_F)**2
                                 for k, r in scenario_results.items()))
    std_error_Var_F = np.sqrt(sum((weights[k]/W * r.std_error_Var_F)**2
                                   for k, r in scenario_results.items()))

    # Approximate confidence interval
    L_min_std_error = np.sqrt(std_error_Var_F**2 + (2 * abs(E_F - F_star) * std_error_E_F)**2)
    confidence_interval = (L_min - 1.96 * L_min_std_error, L_min + 1.96 * L_min_std_error)

    return EmpiricalLminResult(
        L_min=L_min,
        E_F=E_F,
        Var_F=Var_F,
        Bias2=Bias2,
        F_star=F_star,
        n_samples=total_samples,
        std_error_E_F=std_error_E_F,
        std_error_Var_F=std_error_Var_F,
        confidence_interval=confidence_interval,
        computation_time=total_time,
        F_samples=None
    )


class EmpiricalLminTracker:
    """
    Tracks empirical L_min during training for comparison with observed loss.

    This replaces TheoreticalLossTracker when using sequential processes
    with adaptive targets where analytical formulas are invalid.
    """

    def __init__(self, loss_scale: float = 1.0):
        """
        Initialize tracker.

        Args:
            loss_scale: Scale factor applied to loss (e.g., 100.0 in training)
        """
        self.loss_scale = loss_scale
        self.history = {
            'epoch': [],
            'observed_loss': [],
            'empirical_L_min': [],
            'E_F': [],
            'Var_F': [],
            'Bias2': [],
            'gap': [],
            'efficiency': []
        }

    def update(
        self,
        epoch: int,
        observed_loss: float,
        empirical_result: EmpiricalLminResult
    ):
        """
        Record metrics for one epoch.

        Args:
            epoch: Current epoch number
            observed_loss: Observed loss from training (already scaled)
            empirical_result: Result from compute_empirical_L_min
        """
        # Scale empirical L_min to match training loss scale
        scaled_L_min = empirical_result.L_min * self.loss_scale
        scaled_Var_F = empirical_result.Var_F * self.loss_scale
        scaled_Bias2 = empirical_result.Bias2 * self.loss_scale

        gap = observed_loss - scaled_L_min
        efficiency = scaled_L_min / observed_loss if observed_loss > 0 else 0.0

        self.history['epoch'].append(epoch)
        self.history['observed_loss'].append(observed_loss)
        self.history['empirical_L_min'].append(scaled_L_min)
        self.history['E_F'].append(empirical_result.E_F)
        self.history['Var_F'].append(scaled_Var_F)
        self.history['Bias2'].append(scaled_Bias2)
        self.history['gap'].append(gap)
        self.history['efficiency'].append(efficiency)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.history['epoch']:
            return {}

        return {
            'final_observed_loss': self.history['observed_loss'][-1],
            'final_L_min': self.history['empirical_L_min'][-1],
            'final_gap': self.history['gap'][-1],
            'final_efficiency': self.history['efficiency'][-1],
            'final_E_F': self.history['E_F'][-1],
            'final_Var_F': self.history['Var_F'][-1],
            'final_Bias2': self.history['Bias2'][-1],
            'best_efficiency': max(self.history['efficiency']) if self.history['efficiency'] else 0.0,
            'n_violations': sum(1 for g in self.history['gap'] if g < -0.001),  # Small tolerance
            'total_epochs': len(self.history['epoch'])
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export all data for serialization."""
        return {
            'history': self.history,
            'summary': self.get_summary(),
            'loss_scale': self.loss_scale
        }

    def save(self, filepath: Path):
        """Save tracker data to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compare_analytical_vs_empirical(
    analytical_L_min: float,
    empirical_result: EmpiricalLminResult,
    loss_scale: float = 1.0,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare analytical (Jensen) vs empirical L_min estimates.

    This helps diagnose when analytical formulas are inappropriate.
    A large discrepancy indicates the analytical formulas are NOT valid
    for the given system (e.g., due to sequential dependencies).

    Args:
        analytical_L_min: L_min from Jensen formulas (already scaled)
        empirical_result: Result from empirical computation (unscaled)
        loss_scale: Scale factor used in training
        verbose: Print comparison

    Returns:
        Dict with comparison metrics
    """
    # Scale empirical to match analytical
    empirical_L_min_scaled = empirical_result.L_min * loss_scale

    ratio = analytical_L_min / empirical_L_min_scaled if empirical_L_min_scaled > 0 else float('inf')
    difference = analytical_L_min - empirical_L_min_scaled
    relative_error = abs(difference) / empirical_L_min_scaled if empirical_L_min_scaled > 0 else float('inf')

    # Scale confidence interval
    ci_scaled = (
        empirical_result.confidence_interval[0] * loss_scale,
        empirical_result.confidence_interval[1] * loss_scale
    )

    comparison = {
        'analytical_L_min': analytical_L_min,
        'empirical_L_min_scaled': empirical_L_min_scaled,
        'ratio': ratio,
        'difference': difference,
        'relative_error': relative_error,
        'analytical_within_CI': ci_scaled[0] <= analytical_L_min <= ci_scaled[1]
    }

    if verbose:
        print("\n" + "="*60)
        print("ANALYTICAL vs EMPIRICAL L_min COMPARISON")
        print("="*60)
        print(f"  Analytical (Jensen):  {analytical_L_min:.6f}")
        print(f"  Empirical (MC):       {empirical_L_min_scaled:.6f}")
        print(f"  Ratio (A/E):          {ratio:.2f}x")
        print(f"  Difference:           {difference:.6f}")
        print(f"  Relative Error:       {relative_error*100:.1f}%")
        print(f"  Empirical 95% CI:     [{ci_scaled[0]:.6f}, {ci_scaled[1]:.6f}]")
        print(f"  Analytical in CI:     {'Yes' if comparison['analytical_within_CI'] else 'NO - SIGNIFICANT DISCREPANCY'}")

        if ratio > 2.0 or ratio < 0.5:
            print("\n  WARNING: Analytical L_min differs >2x from empirical value!")
            print("      This indicates Jensen formulas are NOT valid for this system.")
            print("      Use EMPIRICAL L_min for all analysis.")
        elif relative_error > 0.3:
            print("\n  WARNING: Relative error >30% between analytical and empirical.")
            print("      Consider using empirical L_min for more accurate analysis.")
        else:
            print("\n  Analytical and empirical L_min are reasonably consistent.")

        print("="*60 + "\n")

    return comparison


def compute_F_star_deterministic(
    surrogate,
    process_chain,
    scenario_idx: int
) -> float:
    """
    Compute F* (target reliability) deterministically for a scenario.

    This uses the mean outputs directly without stochastic sampling,
    which gives the target reliability F* that the controller should achieve.

    Args:
        surrogate: ProTSurrogate instance
        process_chain: ProcessChain instance
        scenario_idx: Scenario index

    Returns:
        F*: Deterministic target reliability
    """
    # Use the precomputed F* from surrogate if available
    if hasattr(surrogate, 'F_star') and surrogate.F_star is not None:
        if isinstance(surrogate.F_star, np.ndarray):
            return float(surrogate.F_star[scenario_idx])
        elif isinstance(surrogate.F_star, (list, tuple)):
            return float(surrogate.F_star[scenario_idx])

    # Otherwise, compute from target trajectory
    # Build trajectory from target data using mean values only
    target_traj = {}
    for process_name in process_chain.process_names:
        if process_name in process_chain.target_trajectory:
            data = process_chain.target_trajectory[process_name]
            inputs = data['inputs'][scenario_idx:scenario_idx+1]
            outputs = data['outputs'][scenario_idx:scenario_idx+1]

            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs, dtype=torch.float32)

            target_traj[process_name] = {
                'inputs': inputs,
                'outputs_mean': outputs,
                'outputs_var': torch.zeros_like(outputs),
                'outputs_sampled': outputs  # Use mean as "sampled" for deterministic F*
            }

    # Temporarily enable deterministic mode
    original_mode = getattr(surrogate, 'use_deterministic_sampling', True)
    surrogate.use_deterministic_sampling = True

    try:
        F_star = surrogate.compute_reliability(target_traj)
        if isinstance(F_star, torch.Tensor):
            F_star = F_star.item()
    finally:
        surrogate.use_deterministic_sampling = original_mode

    return F_star


if __name__ == '__main__':
    """Basic test of empirical L_min computation."""
    print("Testing Empirical L_min Module")
    print("="*60)

    # Create mock objects for testing
    class MockProcessChain:
        def __init__(self):
            self.process_names = ['mock_process']
            self.target_trajectory = {
                'mock_process': {
                    'inputs': np.array([[1.0, 2.0]]),
                    'outputs': np.array([[0.5]])
                }
            }

        def eval(self):
            pass

        def forward(self, batch_size=1, scenario_idx=0):
            # Return trajectory with some randomness
            return {
                'mock_process': {
                    'inputs': torch.tensor([[1.0, 2.0]]),
                    'outputs_mean': torch.tensor([[0.5]]),
                    'outputs_var': torch.tensor([[0.01]]),
                    'outputs_sampled': torch.tensor([[0.5 + 0.1 * np.random.randn()]])
                }
            }

    class MockSurrogate:
        use_deterministic_sampling = False

        def compute_reliability(self, trajectory):
            # Return reliability with some variance
            output = trajectory['mock_process']['outputs_sampled']
            if isinstance(output, torch.Tensor):
                output = output.item()
            # Quality function: exp(-(o - 0.5)^2 / 0.1)
            return torch.tensor(np.exp(-((output - 0.5)**2) / 0.1))

    # Test basic computation
    mock_chain = MockProcessChain()
    mock_surrogate = MockSurrogate()

    result = compute_empirical_L_min(
        process_chain=mock_chain,
        surrogate=mock_surrogate,
        F_star=0.9,  # Target reliability
        n_samples=100,
        verbose=True
    )

    print(f"\nTest Result:")
    print(f"  L_min: {result.L_min:.6f}")
    print(f"  E[F]:  {result.E_F:.6f}")
    print(f"  Var[F]: {result.Var_F:.6f}")
    print(f"  Bias^2: {result.Bias2:.6f}")

    # Verify L_min = Var[F] + Bias^2
    computed = result.Var_F + result.Bias2
    assert abs(result.L_min - computed) < 1e-10, f"L_min mismatch: {result.L_min} vs {computed}"
    print(f"\n  L_min = Var[F] + Bias^2 verified!")
    print("\nTest passed!")
