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
    lines.append("=" * 85)
    lines.append(f"{'Metrica':<20} {'Osservato':>12} {'Teorico':>12} {'Diff':>10} {'Diff %':>10} {'Status':>10}")
    lines.append("-" * 85)

    # Loss finale vs L_min
    final_loss = tracker_summary.get('final_loss', 0)
    final_L_min = tracker_summary.get('final_L_min', 0)
    diff = final_loss - final_L_min
    diff_pct = compute_diff_pct(final_loss, final_L_min) if final_L_min > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Loss finale':<20} {format_value(final_loss):>12} {format_value(final_L_min):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    # Best loss
    best_loss = tracker_summary.get('best_loss', 0)
    diff = best_loss - final_L_min
    diff_pct = compute_diff_pct(best_loss, final_L_min) if final_L_min > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Best loss':<20} {format_value(best_loss):>12} {format_value(final_L_min):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    lines.append("-" * 85)

    # E[F]
    emp_E_F = tracker_summary.get('empirical_E_F_final', 0)
    theo_E_F = tracker_summary.get('theoretical_E_F_final', 0)
    diff = emp_E_F - theo_E_F
    diff_pct = compute_diff_pct(emp_E_F, theo_E_F) if theo_E_F > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'E[F]':<20} {format_value(emp_E_F):>12} {format_value(theo_E_F):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    # Var[F]
    emp_Var_F = tracker_summary.get('empirical_Var_F_final', 0)
    theo_Var_F = tracker_summary.get('theoretical_Var_F_final', 0)
    diff = emp_Var_F - theo_Var_F
    diff_pct = compute_diff_pct(emp_Var_F, theo_Var_F) if theo_Var_F > 0 else 0
    status = get_status_symbol(diff_pct)
    lines.append(f"{'Var[F]':<20} {format_value(emp_Var_F):>12} {format_value(theo_Var_F):>12} {format_value(diff):>10} {format_pct(diff_pct):>10} {status:>10}")

    lines.append("=" * 85)
    lines.append("")
    lines.append(f"L_min = Var[F] (minimo teorico irriducibile)")
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
) -> str:
    """
    Generate table showing loss decomposition.

    Args:
        Var_F: Variance component
        Bias2: Bias squared component
        gap: Reducible gap
        L_min: Theoretical minimum (Var_F + Bias2)
        total_loss: Total observed loss

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA DECOMPOSIZIONE - Contributi alla Loss")
    lines.append("=" * 65)
    lines.append(f"{'Componente':<20} {'Valore':>12} {'% di L_min':>12} {'% di Loss':>12}")
    lines.append("-" * 65)

    # Calculate percentages
    pct_L_min_var = 100 * Var_F / L_min if L_min > 0 else 0
    pct_L_min_bias = 100 * Bias2 / L_min if L_min > 0 else 0
    pct_loss_var = 100 * Var_F / total_loss if total_loss > 0 else 0
    pct_loss_bias = 100 * Bias2 / total_loss if total_loss > 0 else 0
    pct_loss_gap = 100 * gap / total_loss if total_loss > 0 else 0

    lines.append(f"{'Var(F)':<20} {format_value(Var_F):>12} {format_pct(pct_L_min_var):>12} {format_pct(pct_loss_var):>12}")
    lines.append(f"{'Bias²':<20} {format_value(Bias2):>12} {format_pct(pct_L_min_bias):>12} {format_pct(pct_loss_bias):>12}")
    lines.append(f"{'Gap (riducibile)':<20} {format_value(gap):>12} {'-':>12} {format_pct(pct_loss_gap):>12}")
    lines.append("-" * 65)
    lines.append(f"{'L_min':<20} {format_value(L_min):>12} {'100.0%':>12} {format_pct(100*L_min/total_loss if total_loss > 0 else 0):>12}")
    lines.append(f"{'TOTALE':<20} {format_value(total_loss):>12} {'-':>12} {'100.0%':>12}")
    lines.append("=" * 65)
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
    lines.append("=" * 50)
    lines.append(f"{'Metrica':<35} {'Valore':>12}")
    lines.append("-" * 50)

    # Efficiency (L_min / Loss)
    final_eff = tracker_summary.get('final_efficiency', 0)
    lines.append(f"{'Efficienza (L_min/Loss)':<35} {format_pct(100*final_eff):>12}")

    # Gap
    final_loss = tracker_summary.get('final_loss', 0)
    final_L_min = tracker_summary.get('final_L_min', 0)
    gap = final_loss - final_L_min
    gap_rel = 100 * gap / final_loss if final_loss > 0 else 0
    lines.append(f"{'Gap assoluto':<35} {format_value(gap):>12}")
    lines.append(f"{'Gap relativo':<35} {format_pct(gap_rel):>12}")

    lines.append("-" * 50)

    # Best efficiency
    best_eff = tracker_summary.get('best_efficiency', 0)
    lines.append(f"{'Migliore efficienza raggiunta':<35} {format_pct(100*best_eff):>12}")

    # Epochs to efficiency thresholds
    epoch_90 = tracker_summary.get('epoch_90_efficiency', None)
    epoch_95 = tracker_summary.get('epoch_95_efficiency', None)
    lines.append(f"{'Epochs al 90% efficienza':<35} {str(epoch_90) if epoch_90 else 'N/A':>12}")
    lines.append(f"{'Epochs al 95% efficienza':<35} {str(epoch_95) if epoch_95 else 'N/A':>12}")

    # Total epochs
    total_epochs = tracker_summary.get('total_epochs', 0)
    lines.append(f"{'Epochs totali':<35} {str(total_epochs):>12}")

    lines.append("=" * 50)
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


def generate_lambda_grad_table(
    lambda_grad_data: Dict[str, Any],
) -> str:
    """
    Generate table showing Λ_grad per-stage decomposition.

    Args:
        lambda_grad_data: Dict from LambdaGradResult.to_dict()

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA Λ_grad — Approssimazione Delta Method di L_min")
    lines.append("=" * 75)

    lg_val = lambda_grad_data.get('lambda_grad', 0)
    n_traj = lambda_grad_data.get('n_trajectories', 0)
    lines.append(f"Λ_grad(D) = {format_value(lg_val)}  (N = {n_traj} traiettorie)")
    lines.append("")

    lines.append(f"{'Stage':<15} {'Contributo':>14} {'(∂F̂/∂o)²':>14} {'σ²_ψt':>14}")
    lines.append("-" * 75)

    per_stage = lambda_grad_data.get('per_stage', {})
    per_stage_grad_sq = lambda_grad_data.get('per_stage_grad_sq', {})
    per_stage_sigma_sq = lambda_grad_data.get('per_stage_sigma_sq', {})
    process_names = lambda_grad_data.get('process_names', list(per_stage.keys()))

    for name in process_names:
        contrib = per_stage.get(name, 0)
        g2 = per_stage_grad_sq.get(name, 0)
        s2 = per_stage_sigma_sq.get(name, 0)
        lines.append(
            f"{name:<15} {format_value(contrib):>14} "
            f"{format_value(g2):>14} {format_value(s2):>14}"
        )

    lines.append("-" * 75)
    lines.append(f"{'TOTALE':<15} {format_value(lg_val):>14}")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Λ_grad ≈ L_min (approssimazione al primo ordine via Delta Method)")
    lines.append("Contributo_t = (∂F̂/∂o_t)² · σ²_ψt  — amplificazione del rumore allo stage t")
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
    lines.append("Minimo teorico: L_min = Var[F]")
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
        lines.append(generate_decomposition_table(final_Var_F, final_Bias2, final_gap, final_L_min, final_loss))

    # Efficiency table
    lines.append(generate_efficiency_table(summary))

    # Validation table
    lines.append(generate_validation_table(summary, z_score_E_F, z_score_Var_F))

    # Lambda_grad table (if available)
    lambda_grad_data = tracker_data.get('lambda_grad')
    if lambda_grad_data is not None:
        lines.append(generate_lambda_grad_table(lambda_grad_data))

    # Footer
    lines.append("")
    lines.append("=" * 85)
    lines.append("LEGENDA:")
    lines.append("- F* = reliability della traiettoria target (FISSO)")
    lines.append("- F = reliability del controller (sampling stocastico)")
    lines.append("- δ = μ_target - τ (distanza del target dall'ottimo del processo)")
    lines.append("")
    lines.append("CALCOLO L_min:")
    lines.append("- L_min = Var[F] = minimo teorico irriducibile")
    lines.append("- Var[F] = varianza irriducibile (dovuta al sampling stocastico)")
    lines.append("- Bias² = (E[F] - F*)² = bias sistematico")
    lines.append("")
    lines.append("- Gap = Loss - L_min (riducibile con training)")
    lines.append("- Efficienza = L_min / Loss")
    lines.append("=" * 85)
    lines.append("")

    return "\n".join(lines)


def save_report_txt(report: str, path: Path):
    """Save report to text file."""
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved report: {path}")


def save_report_json(tracker_data: Dict[str, Any], path: Path):
    """Save tracker data to JSON file."""
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tracker_data, f, indent=2)
    print(f"  Saved JSON data: {path}")


