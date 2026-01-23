#!/usr/bin/env python
"""
Diagnostic script for L_min calculation issues.

This script helps diagnose why observed loss < L_min (theoretical minimum).
Such violations indicate an error in the theoretical parameter calculation.

The most common issues:
1. Using FIXED tau instead of ADAPTIVE tau in delta calculation
2. Using single-process L_min formula for multi-process system
3. Incorrect sigma2 or scale parameter values

Usage:
    python diagnostic_L_min.py [--results-dir PATH]
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add controller_optimization to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.src.analysis.theoretical_loss_analysis import (
    compute_theoretical_E_F,
    compute_theoretical_E_F2,
    compute_theoretical_L_min,
    compute_multi_process_L_min,
    compute_per_process_Q_stats
)
from controller_optimization.src.models.surrogate import ProTSurrogate


def diagnose_process(name: str, F_star: float, delta: float, sigma2: float, s: float,
                     loss_scale: float = 100.0) -> dict:
    """
    Diagnose theoretical parameters for a single process.

    Args:
        name: Process name
        F_star: Target quality (Q_i*)
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance
        s: Scale parameter of quality function
        loss_scale: Loss scale factor

    Returns:
        Dict with diagnostic information
    """
    print(f"\n{'='*60}")
    print(f"PROCESS: {name}")
    print(f"{'='*60}")
    print(f"  F*:        {F_star:.6f}")
    print(f"  delta:     {delta:.6f}")
    print(f"  sigma2:    {sigma2:.8f}")
    print(f"  s:         {s:.4f}")

    # Key ratios
    sigma2_over_s = sigma2 / s if s > 0 else 0
    delta2_over_s = delta**2 / s if s > 0 else 0

    print(f"  sigma2/s:  {sigma2_over_s:.6f}  {'WARNING: HIGH!' if sigma2_over_s > 0.1 else 'OK'}")
    print(f"  delta2/s:  {delta2_over_s:.6f}  {'WARNING: HIGH!' if delta2_over_s > 1 else 'OK'}")

    # Compute theoretical components
    E_Q, E_Q2, Var_Q = compute_per_process_Q_stats(F_star, delta, sigma2, s)
    Bias2 = (E_Q - F_star)**2
    L_min_unscaled = Var_Q + Bias2
    L_min = L_min_unscaled * loss_scale

    print(f"\n  Theoretical components:")
    print(f"    E[Q]:    {E_Q:.6f}")
    print(f"    E[Q2]:   {E_Q2:.6f}")
    print(f"    Var[Q]:  {Var_Q:.8f}")
    print(f"    Bias2:   {Bias2:.8f}")
    print(f"    L_min (unscaled): {L_min_unscaled:.8f}")
    print(f"    L_min (scaled):   {L_min:.6f}")

    # Check for issues
    if F_star > 1.0:
        print(f"    ERROR: F* > 1.0 (invalid quality value)")
    if E_Q > F_star * 1.001:  # Small tolerance for numerical error
        print(f"    WARNING: E[Q] > F* (unexpected, check formulas)")
    if delta**2 / s > 10:
        print(f"    WARNING: Very large delta2/s - this process dominates L_min")

    return {
        'E_Q': E_Q,
        'E_Q2': E_Q2,
        'Var_Q': Var_Q,
        'Bias2': Bias2,
        'L_min': L_min,
        'L_min_unscaled': L_min_unscaled
    }


def compare_delta_calculation(process_name: str, mu_target: float,
                              tau_fixed: float, tau_adaptive: float, s: float):
    """
    Compare delta calculations using fixed vs adaptive tau.

    This is often the root cause of L_min overestimation.
    """
    delta_fixed = mu_target - tau_fixed
    delta_adaptive = mu_target - tau_adaptive

    F_star_fixed = np.exp(-delta_fixed**2 / s)
    F_star_adaptive = np.exp(-delta_adaptive**2 / s)

    print(f"\n  Delta comparison for {process_name}:")
    print(f"    mu_target:      {mu_target:.6f}")
    print(f"    tau_fixed:      {tau_fixed:.6f}")
    print(f"    tau_adaptive:   {tau_adaptive:.6f}")
    print(f"    delta_fixed:    {delta_fixed:.6f}")
    print(f"    delta_adaptive: {delta_adaptive:.6f}")
    print(f"    F*_fixed:       {F_star_fixed:.6f}")
    print(f"    F*_adaptive:    {F_star_adaptive:.6f}")

    # Only warn if there's a meaningful difference
    if abs(delta_adaptive) > 1e-6:
        ratio = abs(delta_fixed / delta_adaptive)
        if ratio > 2:
            print(f"    >>> delta_fixed >> delta_adaptive by {ratio:.1f}x <<<")
            print(f"    >>> THIS IS LIKELY THE CAUSE OF L_min OVERESTIMATION <<<")
    elif abs(delta_fixed) > 1e-6:
        print(f"    >>> delta_adaptive ≈ 0 but delta_fixed = {delta_fixed:.4f} <<<")
        print(f"    >>> Adaptive target matches output well <<<")

    return delta_fixed, delta_adaptive


def compute_adaptive_targets(outputs: dict) -> dict:
    """
    Compute adaptive targets for all processes.

    Uses the same logic as ProTSurrogate.compute_reliability().
    """
    adaptive_targets = {}

    # LASER: First process, fixed target
    if 'laser' in outputs:
        adaptive_targets['laser'] = 0.8

    # PLASMA: Target depends on Laser
    if 'plasma' in outputs:
        plasma_target = 3.0
        if 'laser' in outputs:
            plasma_target = plasma_target + 0.2 * (outputs['laser'] - 0.8)
        adaptive_targets['plasma'] = plasma_target

    # GALVANIC: Target depends on Laser AND Plasma
    if 'galvanic' in outputs:
        galvanic_target = 10.0
        if 'plasma' in outputs:
            galvanic_target = galvanic_target + 0.5 * (outputs['plasma'] - 5.0)
        if 'laser' in outputs:
            galvanic_target = galvanic_target + 0.4 * (outputs['laser'] - 0.5)
        adaptive_targets['galvanic'] = galvanic_target

    # MICROETCH: Target depends on ALL previous processes
    if 'microetch' in outputs:
        microetch_target = 20.0
        if 'laser' in outputs:
            microetch_target = microetch_target + 1.5 * (outputs['laser'] - 0.5)
        if 'plasma' in outputs:
            microetch_target = microetch_target + 0.3 * (outputs['plasma'] - 5.0)
        if 'galvanic' in outputs:
            microetch_target = microetch_target - 0.15 * (outputs['galvanic'] - 10.0)
        adaptive_targets['microetch'] = microetch_target

    return adaptive_targets


def diagnose_multi_process_L_min(process_params: dict, process_weights: dict,
                                  loss_scale: float = 100.0,
                                  correlation_matrix: dict = None):
    """
    Diagnose multi-process L_min calculation.
    """
    print(f"\n{'='*60}")
    print("MULTI-PROCESS L_min CALCULATION")
    print(f"{'='*60}")

    # Compute using multi-process formula
    combined, per_process = compute_multi_process_L_min(
        process_params=process_params,
        process_weights=process_weights,
        loss_scale=loss_scale,
        correlation_matrix=correlation_matrix
    )

    print(f"\n  Per-process L_min contributions:")
    total_weighted_L_min = 0
    W = sum(process_weights.values())

    for name, comp in per_process.items():
        weight = process_weights.get(name, 1.0)
        contribution = comp.L_min * (weight / W)**2
        total_weighted_L_min += contribution
        print(f"    {name}: L_min={comp.L_min:.6f}, weight={weight:.2f}, contribution={contribution:.6f}")

    print(f"\n  Combined L_min (correct multi-process formula):")
    print(f"    F*:    {combined.F_star:.6f}")
    print(f"    E[F]:  {combined.E_F:.6f}")
    print(f"    Var[F]:{combined.Var_F:.6f}")
    print(f"    Bias2: {combined.Bias2:.6f}")
    print(f"    L_min: {combined.L_min:.6f}")

    # Compare with naive sum
    naive_sum = sum(comp.L_min for comp in per_process.values())
    print(f"\n  Comparison:")
    print(f"    Naive sum of L_min:     {naive_sum:.6f}")
    print(f"    Correct multi-process:  {combined.L_min:.6f}")
    print(f"    Ratio:                  {naive_sum / combined.L_min if combined.L_min > 0 else 0:.2f}x")

    return combined, per_process


def load_and_diagnose_results(results_dir: Path):
    """
    Load training results and diagnose L_min issues.
    """
    print("\n" + "="*70)
    print("LOADING TRAINING RESULTS")
    print("="*70)

    # Load theoretical analysis data
    theoretical_path = results_dir / 'theoretical_analysis_data.json'
    if not theoretical_path.exists():
        print(f"  ERROR: {theoretical_path} not found")
        return

    with open(theoretical_path) as f:
        theoretical_data = json.load(f)

    # Load final results
    final_results_path = results_dir / 'final_results.json'
    if final_results_path.exists():
        with open(final_results_path) as f:
            final_results = json.load(f)
    else:
        final_results = {}

    # Extract key values
    summary = theoretical_data.get('summary', {})
    final_loss = summary.get('final_loss', 0)
    final_L_min = summary.get('final_L_min', 0)
    final_efficiency = summary.get('final_efficiency', 0)
    n_violations = summary.get('n_violations', 0)
    total_epochs = summary.get('total_epochs', 0)

    print(f"\n  Summary from training:")
    print(f"    Final Loss:      {final_loss:.6f}")
    print(f"    Tracker L_min:   {final_L_min:.6f}")
    print(f"    Efficiency:      {final_efficiency*100:.1f}%")
    print(f"    Violations:      {n_violations}/{total_epochs}")

    # Check for combined L_min (correct calculation)
    combined_L_min = theoretical_data.get('combined_L_min', {})
    if combined_L_min:
        correct_L_min = combined_L_min.get('L_min', 0)
        print(f"\n  Combined L_min (correct):")
        print(f"    L_min:           {correct_L_min:.6f}")
        print(f"    F*:              {combined_L_min.get('F_star', 0):.6f}")
        print(f"    E[F]:            {combined_L_min.get('E_F', 0):.6f}")
        print(f"    Var[F]:          {combined_L_min.get('Var_F', 0):.6f}")
        print(f"    Bias2:           {combined_L_min.get('Bias2', 0):.6f}")

        if final_L_min > 0 and correct_L_min > 0:
            ratio = final_L_min / correct_L_min
            print(f"\n  Comparison:")
            print(f"    Tracker L_min / Combined L_min: {ratio:.2f}x")

            if ratio > 1.5:
                print(f"    >>> TRACKER OVERESTIMATES L_min by {ratio:.1f}x <<<")
                print(f"    >>> This is the bug: tracker uses single-process formula <<<")

    # Check per-process data
    per_process_L_min = theoretical_data.get('per_process_L_min', {})
    if per_process_L_min:
        print(f"\n  Per-process L_min:")
        for name, data in per_process_L_min.items():
            print(f"    {name}: L_min={data.get('L_min', 0):.6f}, "
                  f"delta={data.get('delta', 0):.4f}, "
                  f"sigma2={data.get('sigma2', 0):.6f}")

    return theoretical_data, final_results


def run_full_diagnosis(target_outputs: dict = None, sigma2_per_process: dict = None):
    """
    Run full diagnosis with example or provided values.
    """
    print("\n" + "="*70)
    print("FULL DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Default values based on typical AZIMUTH training
    if target_outputs is None:
        target_outputs = {
            'laser': 0.8,
            'plasma': 3.0,
            'galvanic': 10.0,
            'microetch': 20.0
        }

    if sigma2_per_process is None:
        sigma2_per_process = {
            'laser': 0.001,
            'plasma': 0.05,
            'galvanic': 0.2,
            'microetch': 0.5
        }

    # Get scale and weight from PROCESS_CONFIGS
    process_configs = ProTSurrogate.PROCESS_CONFIGS

    # Compute adaptive targets
    adaptive_targets = compute_adaptive_targets(target_outputs)

    print("\n  Target outputs and adaptive targets:")
    for name in target_outputs:
        mu = target_outputs[name]
        tau_fixed = process_configs[name]['target']
        tau_adaptive = adaptive_targets.get(name, tau_fixed)
        print(f"    {name}: mu={mu:.2f}, tau_fixed={tau_fixed:.2f}, tau_adaptive={tau_adaptive:.4f}")

    # Compute per-process parameters with ADAPTIVE targets
    process_params = {}

    for name in target_outputs:
        mu_target = target_outputs[name]
        tau_adaptive = adaptive_targets[name]
        s = process_configs[name]['scale']
        sigma2 = sigma2_per_process.get(name, 0.01)

        # Delta with ADAPTIVE tau
        delta = mu_target - tau_adaptive
        F_star = np.exp(-delta**2 / s)

        process_params[name] = {
            'F_star': F_star,
            'delta': delta,
            'sigma2': sigma2,
            's': s
        }

        # Also compare with fixed tau
        tau_fixed = process_configs[name]['target']
        compare_delta_calculation(name, mu_target, tau_fixed, tau_adaptive, s)

    # Diagnose each process
    print("\n" + "-"*60)
    print("PER-PROCESS DIAGNOSIS (using adaptive tau)")
    print("-"*60)

    loss_scale = 100.0
    for name, params in process_params.items():
        diagnose_process(name, params['F_star'], params['delta'],
                        params['sigma2'], params['s'], loss_scale)

    # Multi-process L_min
    print("\n" + "-"*60)
    print("MULTI-PROCESS DIAGNOSIS")
    print("-"*60)

    process_weights = {name: process_configs[name].get('weight', 1.0)
                       for name in process_params}

    combined, per_process = diagnose_multi_process_L_min(
        process_params, process_weights, loss_scale
    )

    return combined, per_process


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose L_min calculation issues')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to training results directory')
    parser.add_argument('--example', action='store_true',
                       help='Run with example values')

    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            load_and_diagnose_results(results_dir)
        else:
            print(f"ERROR: Results directory not found: {results_dir}")
            return 1

    if args.example or not args.results_dir:
        run_full_diagnosis()

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nKey findings to check:")
    print("  1. Is delta computed with ADAPTIVE tau (not FIXED)?")
    print("  2. Is multi-process L_min formula used (not single-process)?")
    print("  3. Is sigma2 reasonable (sigma2/s should be < 0.1)?")
    print("  4. Are process correlations accounted for?")

    return 0


if __name__ == '__main__':
    sys.exit(main())
