"""
Summary Tables for Empirical Loss Analysis.

Generates formatted tables with empirical statistics.
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
    Generate main results table with observed metrics.

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA PRINCIPALE - Risultati Training")
    lines.append("=" * 65)
    lines.append(f"{'Metrica':<25} {'Valore':>12}")
    lines.append("-" * 65)

    # Loss finale
    final_loss = tracker_summary.get('final_loss', 0)
    lines.append(f"{'Loss finale':<25} {format_value(final_loss):>12}")

    # Best loss
    best_loss = tracker_summary.get('best_loss', 0)
    lines.append(f"{'Best loss':<25} {format_value(best_loss):>12}")

    lines.append("-" * 65)

    # Empirical E[F]
    emp_E_F = tracker_summary.get('empirical_E_F_final', 0)
    lines.append(f"{'E[F] (empirico)':<25} {format_value(emp_E_F):>12}")

    # Empirical Var[F]
    emp_Var_F = tracker_summary.get('empirical_Var_F_final', 0)
    lines.append(f"{'Var[F] (empirico)':<25} {format_value(emp_Var_F):>12}")

    # Total epochs
    total_epochs = tracker_summary.get('total_epochs', 0)
    lines.append(f"{'Epochs totali':<25} {str(total_epochs):>12}")

    lines.append("=" * 65)
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
    total_loss: float,
) -> str:
    """
    Generate table showing empirical statistics decomposition.

    Args:
        Var_F: Variance component
        Bias2: Bias squared component
        total_loss: Total observed loss

    Returns formatted ASCII table string.
    """
    lines = []
    lines.append("")
    lines.append("TABELLA DECOMPOSIZIONE - Statistiche Empiriche")
    lines.append("=" * 50)
    lines.append(f"{'Componente':<20} {'Valore':>12} {'%':>12}")
    lines.append("-" * 50)

    total_components = Var_F + Bias2
    pct_var = 100 * Var_F / total_components if total_components > 0 else 0
    pct_bias = 100 * Bias2 / total_components if total_components > 0 else 0

    lines.append(f"{'Var(F)':<20} {format_value(Var_F):>12} {format_pct(pct_var):>12}")
    lines.append(f"{'Bias²':<20} {format_value(Bias2):>12} {format_pct(pct_bias):>12}")
    lines.append("-" * 50)
    lines.append(f"{'Var[F] + Bias²':<20} {format_value(total_components):>12} {'100.0%':>12}")
    lines.append(f"{'Loss osservata':<20} {format_value(total_loss):>12}")
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

    # E[F] comparison
    status_ef = "OK" if abs(z_score_E_F) < 2 else ("WARN" if abs(z_score_E_F) < 3 else "FAIL")
    lines.append(f"{'E[F] emp. ~ E[F] teo.':<35} {f'z={z_score_E_F:.2f}':>12} {status_ef:>10}")

    # Var[F] comparison
    status_var = "OK" if abs(z_score_Var_F) < 2 else ("WARN" if abs(z_score_Var_F) < 3 else "FAIL")
    lines.append(f"{'Var[F] emp. ~ Var[F] teo.':<35} {f'z={z_score_Var_F:.2f}':>12} {status_var:>10}")

    # Overall validation
    all_ok = (abs(z_score_E_F) < 2) and (abs(z_score_Var_F) < 2)
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
    lines.append("REPORT ANALISI EMPIRICA - Reliability Loss Function")
    lines.append("=" * 85)
    lines.append("")
    lines.append("Questo report riporta le statistiche empiriche raccolte durante il training:")
    lines.append("E[F], Var[F], Bias² = (E[F] - F*)²")
    lines.append("")
    lines.append("Loss function: L = scale * (F - F*)^2")
    lines.append("")

    summary = tracker_data.get('summary', {})

    # Main results table
    lines.append(generate_main_results_table(summary))

    # Process parameters table (if available)
    if process_params:
        lines.append(generate_process_params_table(process_params))

    # Decomposition table
    if 'empirical_Var_F' in tracker_data and len(tracker_data.get('empirical_Var_F', [])) > 0:
        final_Var_F = tracker_data['empirical_Var_F'][-1]
        final_Bias2 = tracker_data['empirical_Bias2'][-1]
        final_loss = tracker_data['observed_loss'][-1]
        lines.append(generate_decomposition_table(final_Var_F, final_Bias2, final_loss))

    # Validation table
    lines.append(generate_validation_table(summary, z_score_E_F, z_score_Var_F))

    # Footer
    lines.append("")
    lines.append("=" * 85)
    lines.append("LEGENDA:")
    lines.append("- F* = reliability della traiettoria target (FISSO)")
    lines.append("- F = reliability del controller (sampling stocastico)")
    lines.append("- E[F] = valore atteso empirico di F")
    lines.append("- Var[F] = varianza empirica di F (dovuta al sampling stocastico)")
    lines.append("- Bias² = (E[F] - F*)² = bias sistematico")
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


if __name__ == '__main__':
    # Test table generation
    print("Testing Table Generation")
    print("="*60)

    # Create dummy data
    summary = {
        'final_loss': 0.1234,
        'best_loss': 0.1100,
        'empirical_E_F_final': 0.82,
        'empirical_Var_F_final': 0.005,
        'total_epochs': 100,
    }

    tracker_data = {
        'summary': summary,
        'empirical_Var_F': [0.005],
        'empirical_Bias2': [0.08],
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
