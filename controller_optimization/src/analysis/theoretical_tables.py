"""
Summary Tables for Theoretical Loss Analysis.

Generates formatted tables comparing observed vs theoretical metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass


def format_value(val: float, precision: int = 6) -> str:
    """Format a float value with consistent precision."""
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}"


def format_pct(val: float, precision: int = 1) -> str:
    """Format a percentage value."""
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}%"


def compute_diff_pct(observed: float, theoretical: float) -> float:
    """Compute percentage difference."""
    if theoretical == 0:
        return 0.0 if observed == 0 else float('inf')
    return 100 * (observed - theoretical) / abs(theoretical)


def get_status_symbol(diff_pct: float, good_thresh: float = 5.0, warn_thresh: float = 20.0) -> str:
    """Get status symbol based on percentage difference."""
    if abs(diff_pct) < good_thresh:
        return "OK"
    elif abs(diff_pct) < warn_thresh:
        return "WARN"
    else:
        return "MISMATCH"


def generate_main_results_table(
    tracker_summary: Dict[str, Any],
    loss_scale: float = 100.0
) -> str:
    """
    Generate main results table comparing observed vs theoretical.

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA PRINCIPALE - Risultati vs Limiti Teorici")
    lines.append("=" * 95)
    lines.append(f"{'Metrica':<25} {'Osservato':>12} {'Teorico':>12} {'Diff':>10} {'Diff %':>10} {'Status':>10}")
    lines.append("-" * 95)

    # Loss finale
    final_loss = tracker_summary.get('final_loss', 0)
    final_L_min = tracker_summary.get('final_L_min', 0)
    final_L_min_perfect = tracker_summary.get('final_L_min_perfect', 0)

    # vs L_min_target
    diff = final_loss - final_L_min
    diff_pct = compute_diff_pct(final_loss, final_L_min) if final_L_min > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Loss vs L_min(target)':<25} {format_value(final_loss):>12} {format_value(final_L_min):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    # vs L_min_perfect
    diff_perfect = final_loss - final_L_min_perfect
    diff_pct_perfect = compute_diff_pct(final_loss, final_L_min_perfect) if final_L_min_perfect > 0 else 0
    status_perfect = get_status_symbol(diff_pct_perfect)
    lines.append(f"{'Loss vs L_min(perfect)':<25} {format_value(final_loss):>12} {format_value(final_L_min_perfect):>12} {format_value(diff_perfect):>10} {format_pct(diff_pct_perfect):>10} {status_perfect:>10}")

    # Best loss vs L_min_perfect
    best_loss = tracker_summary.get('best_loss', 0)
    diff = best_loss - final_L_min_perfect
    diff_pct = compute_diff_pct(best_loss, final_L_min_perfect) if final_L_min_perfect > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Best vs L_min(perfect)':<25} {format_value(best_loss):>12} {format_value(final_L_min_perfect):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    lines.append("-" * 95)

    # E[F]
    emp_E_F = tracker_summary.get('empirical_E_F_final', 0)
    theo_E_F = tracker_summary.get('theoretical_E_F_final', 0)
    diff = emp_E_F - theo_E_F
    diff_pct = compute_diff_pct(emp_E_F, theo_E_F) if theo_E_F > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'E[F]':<25} {format_value(emp_E_F):>12} {format_value(theo_E_F):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    # Var[F]
    emp_Var_F = tracker_summary.get('empirical_Var_F_final', 0)
    theo_Var_F = tracker_summary.get('theoretical_Var_F_final', 0)
    diff = emp_Var_F - theo_Var_F
    diff_pct = compute_diff_pct(emp_Var_F, theo_Var_F) if theo_Var_F > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Var[F]':<25} {format_value(emp_Var_F):>12} {format_value(theo_Var_F):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    lines.append("=" * 95)
    lines.append("")
    lines.append(f"L_min(target) = Var[F] + Bias² (lower bound dato il target attuale)")
    lines.append(f"L_min(perfect) = Var[F] solo (lower bound assoluto, δ=0)")
    lines.append(f"Status: OK = match (<5%), WARN = warning (5-20%), MISMATCH = >20%")
    lines.append("")

    return "\n".join(lines)


def generate_process_params_table(
    process_params: Dict[str, Dict[str, float]]
) -> str:
    """
    Generate table of process parameters used for calculations.

    Args:
        process_params: Dict mapping process_name to {'F_star', 'sigma2', 'delta', 's'}

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA PARAMETRI - Valori usati per i calcoli")
    lines.append("=" * 75)
    lines.append(f"{'Processo':<15} {'F*':>12} {'sigma2':>12} {'delta':>12} {'s':>12}")
    lines.append("-" * 75)

    for process_name, params in process_params.items():
        F_star = params.get('F_star', 0)
        sigma2 = params.get('sigma2', 0)
        delta = params.get('delta', 0)
        s = params.get('s', 0)
        lines.append(f"{process_name:<15} {format_value(F_star):>12} {format_value(sigma2):>12} {format_value(delta):>12} {format_value(s):>12}")

    lines.append("=" * 75)
    lines.append("")

    return "\n".join(lines)


def generate_decomposition_table(
    Var_F: float,
    Bias2: float,
    gap: float,
    L_min: float,
    total_loss: float,
    L_min_perfect: Optional[float] = None
) -> str:
    """
    Generate table showing loss decomposition.

    Args:
        Var_F: Variance component
        Bias2: Bias squared component
        gap: Reducible gap (vs L_min_target)
        L_min: Theoretical minimum with current δ (Var_F + Bias2)
        total_loss: Total observed loss
        L_min_perfect: Theoretical minimum with δ=0 (Var_F only)

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA DECOMPOSIZIONE - Contributi alla Loss")
    lines.append("=" * 75)
    lines.append(f"{'Componente':<25} {'Valore':>12} {'% di L_min':>12} {'% di Loss':>12}")
    lines.append("-" * 75)

    # Calculate percentages
    pct_L_min_var = 100 * Var_F / L_min if L_min > 0 else 0
    pct_L_min_bias = 100 * Bias2 / L_min if L_min > 0 else 0
    pct_loss_var = 100 * Var_F / total_loss if total_loss > 0 else 0
    pct_loss_bias = 100 * Bias2 / total_loss if total_loss > 0 else 0
    pct_loss_gap = 100 * gap / total_loss if total_loss > 0 else 0

    lines.append(f"{'Var(F) [irreducibile]':<25} {format_value(Var_F):>12} {format_pct(pct_L_min_var):>12} {format_pct(pct_loss_var):>12}")
    lines.append(f"{'Bias² [da target]':<25} {format_value(Bias2):>12} {format_pct(pct_L_min_bias):>12} {format_pct(pct_loss_bias):>12}")
    lines.append(f"{'Gap [riducibile]':<25} {format_value(gap):>12} {'-':>12} {format_pct(pct_loss_gap):>12}")
    lines.append("-" * 75)

    # L_min perfect (δ=0)
    if L_min_perfect is not None:
        lines.append(f"{'L_min(perfect, δ=0)':<25} {format_value(L_min_perfect):>12} {'-':>12} {format_pct(100*L_min_perfect/total_loss if total_loss > 0 else 0):>12}")

    # L_min target
    lines.append(f"{'L_min(target, δ attuale)':<25} {format_value(L_min):>12} {'100.0%':>12} {format_pct(100*L_min/total_loss if total_loss > 0 else 0):>12}")
    lines.append(f"{'TOTALE LOSS':<25} {format_value(total_loss):>12} {'-':>12} {'100.0%':>12}")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Note: L_min(perfect) = Var(F) = lower bound assoluto (policy perfetta)")
    lines.append("      L_min(target) = Var(F) + Bias² = lower bound dato il target attuale")
    lines.append("")

    return "\n".join(lines)


def generate_efficiency_table(
    tracker_summary: Dict[str, Any]
) -> str:
    """
    Generate table of efficiency and convergence metrics.

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA EFFICIENZA - Metriche di convergenza")
    lines.append("=" * 60)
    lines.append(f"{'Metrica':<45} {'Valore':>12}")
    lines.append("-" * 60)

    # Efficiency vs L_min_target
    final_eff = tracker_summary.get('final_efficiency', 0)
    lines.append(f"{'Efficienza vs L_min(target)':<45} {format_pct(100*final_eff):>12}")

    # Efficiency vs L_min_perfect
    final_eff_perfect = tracker_summary.get('final_efficiency_perfect', 0)
    lines.append(f"{'Efficienza vs L_min(perfect)':<45} {format_pct(100*final_eff_perfect):>12}")

    lines.append("-" * 60)

    # Gap vs L_min_target
    final_loss = tracker_summary.get('final_loss', 0)
    final_L_min = tracker_summary.get('final_L_min', 0)
    final_L_min_perfect = tracker_summary.get('final_L_min_perfect', 0)

    gap_target = final_loss - final_L_min
    gap_rel_target = 100 * gap_target / final_loss if final_loss > 0 else 0
    lines.append(f"{'Gap vs L_min(target)':<45} {format_value(gap_target):>12}")
    lines.append(f"{'Gap relativo vs L_min(target)':<45} {format_pct(gap_rel_target):>12}")

    # Gap vs L_min_perfect
    gap_perfect = final_loss - final_L_min_perfect
    gap_rel_perfect = 100 * gap_perfect / final_loss if final_loss > 0 else 0
    lines.append(f"{'Gap vs L_min(perfect)':<45} {format_value(gap_perfect):>12}")
    lines.append(f"{'Gap relativo vs L_min(perfect)':<45} {format_pct(gap_rel_perfect):>12}")

    lines.append("-" * 60)

    # Best efficiency
    best_eff = tracker_summary.get('best_efficiency', 0)
    best_eff_perfect = tracker_summary.get('best_efficiency_perfect', 0)
    lines.append(f"{'Migliore efficienza vs L_min(target)':<45} {format_pct(100*best_eff):>12}")
    lines.append(f"{'Migliore efficienza vs L_min(perfect)':<45} {format_pct(100*best_eff_perfect):>12}")

    # Epochs to efficiency thresholds (vs target)
    epoch_90 = tracker_summary.get('epoch_90_efficiency', None)
    epoch_95 = tracker_summary.get('epoch_95_efficiency', None)
    lines.append(f"{'Epochs al 90% efficienza (target)':<45} {str(epoch_90) if epoch_90 else 'N/A':>12}")
    lines.append(f"{'Epochs al 95% efficienza (target)':<45} {str(epoch_95) if epoch_95 else 'N/A':>12}")

    # Total epochs
    total_epochs = tracker_summary.get('total_epochs', 0)
    lines.append(f"{'Epochs totali':<45} {str(total_epochs):>12}")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def generate_validation_table(
    tracker_summary: Dict[str, Any],
    z_score_E_F: float = 0.0,
    z_score_Var_F: float = 0.0
) -> str:
    """
    Generate validation table checking theory predictions.

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("VALIDAZIONE TEORIA")
    lines.append("=" * 60)
    lines.append(f"{'Check':<35} {'Risultato':>12} {'Status':>10}")
    lines.append("-" * 60)

    # Loss >= L_min check
    n_violations = tracker_summary.get('n_violations', 0)
    total_epochs = tracker_summary.get('total_epochs', 0)
    n_valid = total_epochs - n_violations
    valid_ratio = f"{n_valid}/{total_epochs}"
    status = "OK" if n_violations == 0 else ("WARN" if n_violations < total_epochs * 0.05 else "FAIL")
    lines.append(f"{'Loss >= L_min (tutte epoch)':<35} {valid_ratio:>12} {status:>10}")

    # E[F] comparison
    status_ef = "OK" if abs(z_score_E_F) < 2 else ("WARN" if abs(z_score_E_F) < 3 else "FAIL")
    lines.append(f"{'E[F] emp. ~ E[F] teo.':<35} {f'z={z_score_E_F:.2f}':>12} {status_ef:>10}")

    # Var[F] comparison
    status_var = "OK" if abs(z_score_Var_F) < 2 else ("WARN" if abs(z_score_Var_F) < 3 else "FAIL")
    lines.append(f"{'Var[F] emp. ~ Var[F] teo.':<35} {f'z={z_score_Var_F:.2f}':>12} {status_var:>10}")

    # Overall validation
    all_ok = (n_violations == 0) and (abs(z_score_E_F) < 2) and (abs(z_score_Var_F) < 2)
    overall_status = "OK" if all_ok else "FAIL"
    lines.append("-" * 60)
    lines.append(f"{'Teoria validata':<35} {'':>12} {overall_status:>10}")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def generate_full_report(
    tracker_data: Dict[str, Any],
    process_params: Optional[Dict[str, Dict[str, float]]] = None,
    z_score_E_F: float = 0.0,
    z_score_Var_F: float = 0.0
) -> str:
    """
    Generate complete report with all tables.

    Args:
        tracker_data: Dictionary from TheoreticalLossTracker.to_dict()
        process_params: Process parameters (optional)
        z_score_E_F: Z-score for E[F] validation
        z_score_Var_F: Z-score for Var[F] validation

    Returns:
        Complete report as string
    """
    lines = []

    # Header
    lines.append("=" * 85)
    lines.append("REPORT ANALISI TEORICA - Reliability Loss Function")
    lines.append("=" * 85)
    lines.append("")
    lines.append("Questo report confronta i risultati osservati durante il training")
    lines.append("con i limiti teorici derivati dall'analisi della loss function.")
    lines.append("")
    lines.append("Loss function: L = scale * (F - F*)^2")
    lines.append("Minimo teorico: L_min = Var[F] + Bias^2")
    lines.append("")

    summary = tracker_data.get('summary', {})

    # Main results table
    lines.append(generate_main_results_table(summary))

    # Process parameters table (if available)
    if process_params:
        lines.append(generate_process_params_table(process_params))

    # Decomposition table
    if len(tracker_data.get('theoretical_Var_F', [])) > 0:
        final_Var_F = tracker_data['theoretical_Var_F'][-1]
        final_Bias2 = tracker_data['theoretical_Bias2'][-1]
        final_gap = tracker_data['gap'][-1]
        final_L_min = tracker_data['theoretical_L_min'][-1]
        final_loss = tracker_data['observed_loss'][-1]
        final_L_min_perfect = tracker_data['theoretical_L_min_perfect'][-1] if tracker_data.get('theoretical_L_min_perfect') else None
        lines.append(generate_decomposition_table(final_Var_F, final_Bias2, final_gap, final_L_min, final_loss, final_L_min_perfect))

    # Efficiency table
    lines.append(generate_efficiency_table(summary))

    # Validation table
    lines.append(generate_validation_table(summary, z_score_E_F, z_score_Var_F))

    # Footer
    lines.append("")
    lines.append("=" * 95)
    lines.append("LEGENDA:")
    lines.append("- F* = reliability della traiettoria target (deterministico)")
    lines.append("- F = reliability del controller (sampling stocastico)")
    lines.append("- δ = μ_target - τ (distanza del target dall'ottimo del processo)")
    lines.append("")
    lines.append("DUE TIPI DI L_min:")
    lines.append("- L_min(perfect) = Var[F] = lower bound assoluto (δ=0, policy perfetta)")
    lines.append("                   Rappresenta il minimo irriducibile se si potesse scegliere")
    lines.append("                   τ (target ottimale) come output desiderato.")
    lines.append("")
    lines.append("- L_min(target)  = Var[F] + Bias² = lower bound dato il target attuale")
    lines.append("                   Rappresenta il minimo irriducibile dato che il target")
    lines.append("                   è fissato a un valore diverso dall'ottimo (δ ≠ 0).")
    lines.append("")
    lines.append("- Gap = differenza tra loss osservata e L_min (riducibile con training)")
    lines.append("- Efficienza = L_min / Loss (quanto della loss è spiegata dal limite teorico)")
    lines.append("=" * 95)
    lines.append("")

    return "\n".join(lines)


def save_report_txt(report: str, path: Path):
    """Save report to text file."""
    path = Path(path)
    with open(path, 'w') as f:
        f.write(report)
    print(f"  Saved report: {path}")


def save_report_json(tracker_data: Dict[str, Any], path: Path):
    """Save tracker data to JSON file."""
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    print(f"  Saved JSON data: {path}")


if __name__ == '__main__':
    # Test table generation
    print("Testing Table Generation")
    print("="*60)

    # Create dummy data
    summary = {
        'final_loss': 0.1234,
        'best_loss': 0.1100,
        'final_L_min': 0.0850,
        'final_gap': 0.0384,
        'final_efficiency': 0.689,
        'best_efficiency': 0.773,
        'mean_efficiency': 0.65,
        'empirical_E_F_final': 0.82,
        'theoretical_E_F_final': 0.84,
        'empirical_Var_F_final': 0.005,
        'theoretical_Var_F_final': 0.0048,
        'n_violations': 0,
        'total_epochs': 100,
        'violation_rate': 0.0,
        'epoch_90_efficiency': 45,
        'epoch_95_efficiency': None
    }

    tracker_data = {
        'summary': summary,
        'theoretical_Var_F': [0.005],
        'theoretical_Bias2': [0.08],
        'gap': [0.0384],
        'theoretical_L_min': [0.085],
        'observed_loss': [0.1234]
    }

    process_params = {
        'laser': {'F_star': 0.95, 'sigma2': 0.01, 'delta': 0.05, 's': 0.1},
        'plasma': {'F_star': 0.88, 'sigma2': 0.02, 'delta': 0.10, 's': 2.0}
    }

    # Generate full report
    report = generate_full_report(tracker_data, process_params, z_score_E_F=0.5, z_score_Var_F=0.3)
    print(report)

    print("\nTest passed!")
