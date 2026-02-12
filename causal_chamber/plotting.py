"""
Plotting module — Nature Machine Intelligence style.

All figures follow the style of the Causal Chamber paper
(Gamella, Peters, Bühlmann — Nature Machine Intelligence, 2025):
serif fonts, white background, thin lines, sober colors, LaTeX labels,
light grid, compact multi-panel figures.

Every figure is saved as both PNG (300 dpi, for the PDF report) and
PDF (vector, for paper usage).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Global style (Nature Machine Intelligence)
# ---------------------------------------------------------------------------

STYLE_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.5,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
}

# Color palette (colorblind-friendly, matching paper style)
C_BLUE = '#4575b4'
C_RED = '#d73027'
C_GREY = '#808080'
C_LIGHT_GREY = '#c0c0c0'
C_BLACK = '#2a2a2a'
C_GREEN = '#1b7837'
C_ORANGE = '#fc8d59'

# Method colors for bar charts
METHOD_COLORS = {
    'Attention': C_BLUE,
    'PC': C_ORANGE,
    'GES': C_GREEN,
}


def apply_style():
    """Apply the Nature Machine Intelligence style globally."""
    plt.rcParams.update(STYLE_PARAMS)


def _save_figure(fig, output_dir: Path, name: str):
    """Save figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# 1. CAUSAL DISCOVERY FIGURES
# ===================================================================

def _build_dag_graph(edges, nodes):
    """Build a networkx DiGraph from edge list."""
    if not NX_AVAILABLE:
        warnings.warn("networkx not installed. DAG plots unavailable.")
        return None
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def _hierarchical_layout(nodes: List[str], process_order=None):
    """
    Create a hierarchical top-down layout respecting process flow.

    laser inputs -> laser output -> plasma inputs -> plasma output -> ... -> F
    """
    from causal_chamber.ground_truth import PROCESS_ORDER, PROCESS_OBSERVABLE_VARS

    if process_order is None:
        process_order = PROCESS_ORDER

    pos = {}
    y = 0
    for proc in process_order:
        info = PROCESS_OBSERVABLE_VARS[proc]
        inputs = [v for v in info['inputs'] if v in nodes]
        outputs = [v for v in info['outputs'] if v in nodes]

        # Inputs row
        n_inp = len(inputs)
        for i, v in enumerate(inputs):
            x = (i - (n_inp - 1) / 2) * 1.5
            pos[v] = (x, -y)
        y += 1

        # Outputs row
        n_out = len(outputs)
        for i, v in enumerate(outputs):
            x = (i - (n_out - 1) / 2) * 1.5
            pos[v] = (x, -y)
        y += 1

    # F at the bottom
    if 'F' in nodes:
        pos['F'] = (0, -y)

    return pos


def plot_dag_comparison(
    truth_edges: List[Tuple[str, str]],
    estimated_edges: List[Tuple[str, str]],
    nodes: List[str],
    output_dir: Path,
    name: str = 'dag_comparison',
):
    """
    Side-by-side DAG plot: ground truth G* and estimated graph G-hat.

    True positive edges in black, false positives in red,
    false negatives in grey dashed.
    """
    if not NX_AVAILABLE:
        warnings.warn("networkx not available. Skipping DAG comparison plot.")
        return

    apply_style()

    truth_set = set(truth_edges)
    est_set = set(estimated_edges)

    tp = truth_set & est_set
    fp = est_set - truth_set
    fn = truth_set - est_set

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    pos = _hierarchical_layout(nodes)

    # --- Ground truth ---
    ax = axes[0]
    G_true = _build_dag_graph(truth_edges, nodes)
    nx.draw_networkx_nodes(G_true, pos, ax=ax, node_size=400, node_color='white',
                           edgecolors=C_BLACK, linewidths=0.8)
    nx.draw_networkx_labels(G_true, pos, ax=ax, font_size=7, font_family='serif')
    nx.draw_networkx_edges(G_true, pos, ax=ax, edge_color=C_BLACK,
                           arrows=True, arrowstyle='->', arrowsize=10,
                           width=0.8, connectionstyle='arc3,rad=0.1')
    ax.set_title(r'Ground Truth $G^*$', fontsize=11, fontfamily='serif')
    ax.axis('off')

    # --- Estimated with colored edges ---
    ax = axes[1]
    G_est = nx.DiGraph()
    G_est.add_nodes_from(nodes)
    # Add all edges (TP + FP + FN for display)
    all_display_edges = tp | fp | fn

    nx.draw_networkx_nodes(G_est, pos, ax=ax, node_size=400, node_color='white',
                           edgecolors=C_BLACK, linewidths=0.8)
    nx.draw_networkx_labels(G_est, pos, ax=ax, font_size=7, font_family='serif')

    # Draw TP edges (black)
    if tp:
        nx.draw_networkx_edges(G_est, pos, edgelist=list(tp), ax=ax,
                               edge_color=C_BLACK, arrows=True,
                               arrowstyle='->', arrowsize=10, width=0.8,
                               connectionstyle='arc3,rad=0.1')
    # Draw FP edges (red)
    if fp:
        nx.draw_networkx_edges(G_est, pos, edgelist=list(fp), ax=ax,
                               edge_color=C_RED, arrows=True,
                               arrowstyle='->', arrowsize=10, width=1.0,
                               connectionstyle='arc3,rad=0.1')
    # Draw FN edges (grey dashed)
    if fn:
        nx.draw_networkx_edges(G_est, pos, edgelist=list(fn), ax=ax,
                               edge_color=C_LIGHT_GREY, arrows=True,
                               arrowstyle='->', arrowsize=10, width=0.8,
                               style='dashed',
                               connectionstyle='arc3,rad=0.1')

    ax.set_title(r'Estimated $\hat{G}$', fontsize=11, fontfamily='serif')
    ax.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BLACK, lw=1, label='True Positive'),
        Line2D([0], [0], color=C_RED, lw=1, label='False Positive'),
        Line2D([0], [0], color=C_LIGHT_GREY, lw=1, linestyle='--', label='False Negative'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_figure(fig, output_dir, name)


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    var_names: List[str],
    output_dir: Path,
    name: str = 'attention_heatmap',
    title: str = 'Aggregated Attention Weights',
):
    """
    Heatmap of the aggregated attention matrix (variables x variables).
    Blues colormap, annotated with 2-decimal values, axes with variable names
    rotated 45 degrees, colorbar on the side.
    """
    apply_style()

    n = attention_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))

    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')

    # Annotate cells
    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            val = attention_matrix[i, j]
            color = 'white' if val > attention_matrix.max() * 0.7 else C_BLACK
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=max(5, 8 - n // 5), color=color)

    # Labels
    row_names = var_names[:attention_matrix.shape[0]]
    col_names = var_names[:attention_matrix.shape[1]]
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=10, fontfamily='serif')
    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_metrics_bar_chart(
    metrics_dict: Dict[str, Dict[str, float]],
    output_dir: Path,
    name: str = 'metrics_bar_chart',
):
    """
    Grouped bar chart of precision/recall/F1 per method.

    Parameters
    ----------
    metrics_dict : dict
        Keys: method names (e.g., 'Attention', 'GES', 'PC').
        Values: dict with keys 'precision', 'recall', 'f1'.
    """
    apply_style()

    methods = list(metrics_dict.keys())
    metric_names = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', '$F_1$']

    n_methods = len(methods)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        color = METHOD_COLORS.get(method, C_GREY)
        bars = ax.bar(x + i * width - (n_methods - 1) * width / 2, values,
                      width, label=method, color=color, edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 2. INTERVENTIONAL VALIDATION FIGURES
# ===================================================================

def plot_intervention_violins(
    obs_values: np.ndarray,
    int_values: np.ndarray,
    variable_name: str,
    intervention_str: str,
    p_value: float,
    output_dir: Path,
    name: str = 'intervention_violin',
):
    """
    Violin/box plot: observational vs interventional distribution.
    Blue for observational, red for interventional.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(4, 4))

    data = [obs_values, int_values]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True,
                          showmedians=True, showextrema=False)

    # Color violins
    colors = [C_BLUE, C_RED]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color(C_BLACK)
    parts['cmedians'].set_color(C_BLACK)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Observational', 'Interventional'], fontsize=8)
    ax.set_ylabel(variable_name, fontsize=9)

    # Annotate p-value
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
    ax.text(0.5, 0.95, f'p={p_value:.2e} ({sig})',
            transform=ax.transAxes, ha='center', fontsize=7, style='italic')

    ax.set_title(intervention_str, fontsize=9, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_f_scatter(
    F_formula: np.ndarray,
    F_predicted: np.ndarray,
    output_dir: Path,
    name: str = 'f_scatter',
    title: str = '',
    labels: Optional[List[str]] = None,
    colors_list: Optional[List[str]] = None,
):
    """
    Scatter plot: F_formula (x) vs F_CausaliT (y) under intervention.
    Dashed y=x line, R² annotated.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(5, 5))

    F_f = np.asarray(F_formula).ravel()
    F_p = np.asarray(F_predicted).ravel()

    if colors_list is not None:
        ax.scatter(F_f, F_p, c=colors_list, s=15, alpha=0.7, edgecolor='none')
    else:
        ax.scatter(F_f, F_p, c=C_BLUE, s=15, alpha=0.5, edgecolor='none')

    # y=x line
    lims = [min(F_f.min(), F_p.min()), max(F_f.max(), F_p.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, '--', color=C_GREY, linewidth=0.8, zorder=0)

    # R²
    from sklearn.metrics import r2_score
    r2 = r2_score(F_f, F_p) if len(F_f) > 1 else float('nan')
    ax.text(0.05, 0.92, f'$R^2 = {r2:.3f}$',
            transform=ax.transAxes, fontsize=9, fontfamily='serif')

    ax.set_xlabel(r'$F$ (formula)', fontsize=9)
    ax.set_ylabel(r'$F$ (CausaliT)', fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontfamily='serif')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_intervention_summary_violins(
    results_list: List[Dict],
    output_dir: Path,
    name: str = 'intervention_summary',
):
    """
    Multi-panel violin plot for all interventions.

    Each (intervention, variable) pair gets its own panel, so both
    process outputs and F are shown when available.
    """
    apply_style()

    # Flatten: one panel per (result, variable) pair
    panels = []
    for result in results_list:
        proc = result['process']
        intv = result['intervention']
        intv_str = ', '.join(f"do({k}={v})" for k, v in intv.items())
        obs_df = result['obs_data']
        int_df = result['int_data']
        for var, comp in result['comparisons'].items():
            if var in obs_df.columns and var in int_df.columns:
                panels.append((proc, intv_str, var, comp, obs_df, int_df))

    n_panels = len(panels)
    if n_panels == 0:
        return

    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for idx, (proc, intv_str, var, comp, obs_df, int_df) in enumerate(panels):
        ax = axes[idx]

        data = [obs_df[var].values, int_df[var].values]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True,
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([C_BLUE, C_RED][i])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color(C_BLACK)
        parts['cmedians'].set_color(C_BLACK)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Obs', 'Int'], fontsize=7)
        ax.set_ylabel(var, fontsize=7)

        pval = comp['p_value']
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        ax.set_title(f'{proc}: {intv_str}\n{var}: p={pval:.1e} ({sig})',
                     fontsize=7, fontfamily='serif')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 3. OOD ANALYSIS FIGURES
# ===================================================================

def plot_ood_bar_chart(
    id_metrics: Dict[str, Dict[str, float]],
    ood_metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
    name: str = 'ood_bar_chart',
    id_F_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ood_F_metrics: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Grouped bar chart: metrics (mean, std, etc.) in-distribution vs OOD per process.
    Blue = ID, Red = OOD.

    If F metrics are provided, a second subplot shows F under each OOD shift.
    """
    apply_style()

    processes = list(id_metrics.keys())
    n_proc = len(processes)
    has_F = id_F_metrics is not None and ood_F_metrics is not None

    if has_F:
        fig, (ax, ax_f) = plt.subplots(1, 2, figsize=(max(6, n_proc * 2) + 4, 4),
                                        gridspec_kw={'width_ratios': [n_proc, n_proc]})
    else:
        fig, ax = plt.subplots(figsize=(max(6, n_proc * 2), 4))

    x = np.arange(n_proc)
    width = 0.35

    id_vals = [id_metrics[p].get('mean', 0) for p in processes]
    ood_vals = [ood_metrics[p].get('mean', 0) for p in processes]

    bars1 = ax.bar(x - width / 2, id_vals, width, label='In-distribution',
                   color=C_BLUE, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ood_vals, width, label='OOD',
                   color=C_RED, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars1, id_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, ood_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(processes, fontsize=8)
    ax.set_ylabel('Output Mean', fontsize=9)
    ax.set_title('Per-process outputs', fontsize=10, fontfamily='serif')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # F subplot
    if has_F:
        id_F_vals = [id_F_metrics[p].get('mean', 0) for p in processes]
        ood_F_vals = [ood_F_metrics[p].get('mean', 0) for p in processes]

        bars_f1 = ax_f.bar(x - width / 2, id_F_vals, width, label='ID',
                           color=C_BLUE, edgecolor='white', linewidth=0.5)
        bars_f2 = ax_f.bar(x + width / 2, ood_F_vals, width, label='OOD',
                           color=C_RED, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars_f1, id_F_vals):
            ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                      f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars_f2, ood_F_vals):
            ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                      f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax_f.set_xticks(x)
        ax_f.set_xticklabels([f'{p}\nshift' for p in processes], fontsize=8)
        ax_f.set_ylabel('Reliability $F$', fontsize=9)
        ax_f.set_title('Pipeline $F$ under OOD shift', fontsize=10, fontfamily='serif')
        ax_f.legend(frameon=False, fontsize=8)
        ax_f.spines['top'].set_visible(False)
        ax_f.spines['right'].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_ood_attention_comparison(
    id_attention: np.ndarray,
    ood_attention: np.ndarray,
    var_names: List[str],
    output_dir: Path,
    name: str = 'ood_attention_comparison',
):
    """
    Side-by-side attention heatmaps: in-distribution vs OOD.
    Same colormap and scale for fair comparison.
    """
    apply_style()

    vmin = min(id_attention.min(), ood_attention.min())
    vmax = max(id_attention.max(), ood_attention.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_rows_id = id_attention.shape[0]
    n_cols_id = id_attention.shape[1]
    n_rows_ood = ood_attention.shape[0]
    n_cols_ood = ood_attention.shape[1]

    for ax, att, title in [(ax1, id_attention, 'In-Distribution'),
                           (ax2, ood_attention, 'OOD')]:
        im = ax.imshow(att, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto')
        n_r, n_c = att.shape
        row_names = var_names[:n_r]
        col_names = var_names[:n_c]
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names, fontsize=6)
        ax.set_title(title, fontsize=10, fontfamily='serif')

    fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_ood_f_scatter(
    F_id: np.ndarray,
    F_ood: np.ndarray,
    output_dir: Path,
    name: str = 'ood_f_scatter',
):
    """
    Scatter plot of F predictions: ID (blue) and OOD (red).
    X-axis: sample index, Y-axis: predicted F.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(range(len(F_id)), F_id, c=C_BLUE, s=10, alpha=0.5,
               label='In-distribution', edgecolor='none')
    ax.scatter(range(len(F_id), len(F_id) + len(F_ood)), F_ood,
               c=C_RED, s=10, alpha=0.5, label='OOD', edgecolor='none')

    ax.axhline(y=np.mean(F_id), color=C_BLUE, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=np.mean(F_ood), color=C_RED, linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Sample', fontsize=9)
    ax.set_ylabel(r'$F$', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 4. SYMBOLIC REGRESSION FIGURES
# ===================================================================

def plot_symbolic_scatter(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_names: List[str],
    output_name: str,
    equation_str: str,
    true_equation_str: str,
    r2: float,
    output_dir: Path,
    name: str = 'symbolic_scatter',
):
    """
    Scatter plot with fit curve for symbolic regression.

    For 1D: scatter of data + fitted curve + true curve.
    For 2D+: scatter colored by first input variable.
    """
    apply_style()

    n_inputs = X.shape[1] if X.ndim > 1 else 1

    if n_inputs == 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        x_vals = X.ravel()
        sort_idx = np.argsort(x_vals)

        ax.scatter(x_vals, y_true, c=C_LIGHT_GREY, s=5, alpha=0.3,
                   label='Data', zorder=1)
        ax.plot(x_vals[sort_idx], y_pred[sort_idx], color=C_RED,
                linewidth=1.2, label='Discovered', zorder=2)
        ax.set_xlabel(input_names[0], fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        sort_idx = np.argsort(y_true)
        sc = ax.scatter(y_true[sort_idx], y_pred[sort_idx],
                        c=X[sort_idx, 0], cmap='viridis', s=10, alpha=0.5)
        fig.colorbar(sc, ax=ax, label=input_names[0], fraction=0.046, pad=0.04)

        # y=x line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, '--', color=C_GREY, linewidth=0.8)

        ax.set_xlabel(f'{output_name} (true)', fontsize=9)
        ax.set_ylabel(f'{output_name} (predicted)', fontsize=9)

    ax.text(0.05, 0.92, f'$R^2 = {r2:.4f}$',
            transform=ax.transAxes, fontsize=9, fontfamily='serif')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_symbolic_equations_comparison(
    process_results: Dict[str, Dict],
    output_dir: Path,
    name: str = 'symbolic_equations',
):
    """
    Multi-panel figure: for each process, scatter + discovered equation.
    """
    apply_style()

    processes = list(process_results.keys())
    n = len(processes)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (proc, result) in enumerate(process_results.items()):
        ax = axes[idx]
        y_true = result['y_true']
        best = result['best_fit']
        y_pred = best['predictions']

        sort_idx = np.argsort(y_true)
        ax.scatter(y_true[sort_idx], y_pred[sort_idx],
                   c=C_BLUE, s=8, alpha=0.4, edgecolor='none')

        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, '--', color=C_GREY, linewidth=0.8)

        r2 = best['r2']
        ax.set_title(f'{proc}\n$R^2 = {r2:.4f}$', fontsize=9, fontfamily='serif')
        ax.set_xlabel('True', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 5. GROUND TRUTH DAG PLOT (standalone)
# ===================================================================

def plot_ground_truth_dag(
    edges: List[Tuple[str, str]],
    nodes: List[str],
    output_dir: Path,
    name: str = 'ground_truth_dag',
):
    """Plot the ground truth DAG alone."""
    if not NX_AVAILABLE:
        warnings.warn("networkx not available. Skipping ground truth DAG plot.")
        return

    apply_style()

    fig, ax = plt.subplots(figsize=(8, 10))
    pos = _hierarchical_layout(nodes)

    G = _build_dag_graph(edges, nodes)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='white',
                           edgecolors=C_BLACK, linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_family='serif')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_BLACK,
                           arrows=True, arrowstyle='->', arrowsize=12,
                           width=0.8, connectionstyle='arc3,rad=0.05')

    ax.set_title(r'Ground Truth DAG $G^*$', fontsize=12, fontfamily='serif')
    ax.axis('off')
    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 6. CAUSAL VALIDATION FIGURES
# ===================================================================

def plot_validation_heatmap(
    results: Dict,
    output_dir: Path,
    name: str = 'causal_validation_summary',
):
    """
    Multi-panel figure: per-process p-value heatmaps + pipeline F row.

    Red boxes mark ground truth edges. Dark cells = significant (small p).
    """
    from causal_chamber.ground_truth import PROCESS_OBSERVABLE_VARS, get_ground_truth_edges

    apply_style()

    gt_edge_set = set(get_ground_truth_edges())
    pvalue_matrices = results['pvalue_matrices']
    pipeline_matrix = results['pipeline_pvalue_matrix']

    n_panels = len(pvalue_matrices) + 1  # +1 for pipeline F row
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5),
                              gridspec_kw={'width_ratios': [1] * len(pvalue_matrices) + [1.5]})

    # Per-process heatmaps
    for idx, (proc_name, pmat) in enumerate(pvalue_matrices.items()):
        ax = axes[idx]
        info = PROCESS_OBSERVABLE_VARS[proc_name]

        # Take -log10 for visualization (larger = more significant)
        log_p = -np.log10(pmat.values.astype(float) + 1e-300)

        im = ax.imshow(log_p, cmap='YlOrRd', aspect='auto', vmin=0, vmax=50)

        # Mark ground truth edges with red border
        for i, row_var in enumerate(pmat.index):
            for j, col_label in enumerate(pmat.columns):
                # Extract variable name from "do(Var=val)"
                intv_var = col_label.split('(')[1].split('=')[0]
                if (intv_var, row_var) in gt_edge_set:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         linewidth=2, edgecolor=C_RED,
                                         facecolor='none')
                    ax.add_patch(rect)

                # Annotate with p-value
                p = pmat.iloc[i, j]
                if p < 0.001:
                    txt = f'{log_p[i, j]:.0f}'
                else:
                    txt = f'{p:.2f}'
                color = 'white' if log_p[i, j] > 25 else C_BLACK
                ax.text(j, i, txt, ha='center', va='center', fontsize=6, color=color)

        ax.set_xticks(range(len(pmat.columns)))
        ax.set_xticklabels([c.split('(')[1].rstrip(')') for c in pmat.columns],
                           rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(pmat.index)))
        ax.set_yticklabels(pmat.index, fontsize=7)
        ax.set_title(proc_name, fontsize=10, fontfamily='serif')

    # Pipeline F heatmap (F row only)
    ax = axes[-1]
    if 'F' in pipeline_matrix.index:
        f_row = pipeline_matrix.loc[['F']].values.astype(float)
        log_p_f = -np.log10(f_row + 1e-300)

        im = ax.imshow(log_p_f, cmap='YlOrRd', aspect='auto', vmin=0, vmax=50)

        for j in range(f_row.shape[1]):
            p = f_row[0, j]
            if p < 0.001:
                txt = f'{log_p_f[0, j]:.0f}'
            else:
                txt = f'{p:.2f}'
            color = 'white' if log_p_f[0, j] > 25 else C_BLACK
            ax.text(j, 0, txt, ha='center', va='center', fontsize=6, color=color)

        ax.set_xticks(range(len(pipeline_matrix.columns)))
        ax.set_xticklabels([c.split('(')[1].rstrip(')') for c in pipeline_matrix.columns],
                           rotation=45, ha='right', fontsize=6)
        ax.set_yticks([0])
        ax.set_yticklabels(['F'], fontsize=7)
        ax.set_title('Pipeline -> F', fontsize=10, fontfamily='serif')

    fig.suptitle(r'Causal Validation: $-\log_{10}(p)$  (red box = ground truth edge)',
                 fontsize=11, fontfamily='serif', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, name)


# ===================================================================
# 7. OOD RADAR/SPIDER PLOTS (reproduces paper methodology)
# ===================================================================

# Colorblind-friendly palette for models (IBM design)
_RADAR_COLORS = [
    '#648FFF',  # blue
    '#DC267F',  # magenta
    '#FE6100',  # orange
    '#FFB000',  # yellow
    '#785EF0',  # purple
    '#009E73',  # green
]


def plot_ood_radar(
    model_results: Dict[str, Dict[str, Dict[str, float]]],
    env_names: List[str],
    output_dir: Path,
    name: str = 'ood_radar',
    title: str = '',
):
    """
    Radar/spider plot of MAE across environments for multiple models.

    Reproduces the Causal Chamber paper's ood_sensors.ipynb visualization:
    - Polar plot with one axis per environment
    - One closed polygon per model
    - Cube-root transformation for better readability
    - Dashed line for baseline 'mean' model

    Parameters
    ----------
    model_results : dict
        {model_name: {env_name: {'mean': float, 'std': float}}}.
    env_names : list of str
        Environment names (axes of the radar plot).
    output_dir : Path
        Output directory.
    name : str
        Figure filename (without extension).
    title : str
        Figure title.
    """
    apply_style()

    models = list(model_results.keys())
    N = len(env_names)
    if N == 0 or len(models) == 0:
        return

    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Cube root transform (as in the paper)
    def t(x):
        return np.power(np.maximum(x, 0), 1.0 / 3.0)

    for idx, model_name in enumerate(models):
        values = []
        for env in env_names:
            mae = model_results[model_name].get(env, {}).get('mean', 0)
            values.append(mae)
        values += values[:1]  # close
        values_t = [t(v) for v in values]

        color = _RADAR_COLORS[idx % len(_RADAR_COLORS)]
        if model_name == 'mean':
            label = 'Regr. to mean'
            linestyle = '--'
            color = C_BLACK
        else:
            label = model_name
            linestyle = '-'

        ax.plot(angles, values_t, linewidth=1.2, linestyle=linestyle,
                label=label, color=color)
        ax.fill(angles, values_t, alpha=0.08, color=color)

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(env_names, fontsize=7)

    ax.set_ylabel(r'MAE$^{1/3}$', fontsize=8, labelpad=20)
    if title:
        ax.set_title(title, fontsize=10, fontfamily='serif', pad=20)

    ax.legend(loc='lower left', bbox_to_anchor=(1.05, -0.05),
              frameon=False, fontsize=7, title='Model', title_fontsize=8)

    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def plot_ood_radar_multi(
    results: Dict,
    output_dir: Path,
    name: str = 'ood_radar_all',
):
    """
    Multi-panel radar plots: one per process + one for F.

    Parameters
    ----------
    results : dict
        Output of run_ood_analysis(), with 'per_process' and 'reliability' keys.
    """
    apply_style()

    processes = list(results.get('per_process', {}).keys())
    has_F = 'reliability' in results and results['reliability']
    n_panels = len(processes) + (1 if has_F else 0)

    if n_panels == 0:
        return

    for proc_name, proc_data in results['per_process'].items():
        env_names = proc_data.get('env_names', [])
        model_results = proc_data.get('model_results', {})
        plot_ood_radar(
            model_results, env_names, output_dir,
            name=f'ood_radar_{proc_name}',
            title=f'{proc_name}: {proc_data["output_var"]}',
        )

    if has_F:
        rel = results['reliability']
        plot_ood_radar(
            rel['model_results'], rel['env_names'], output_dir,
            name='ood_radar_F',
            title='Pipeline Reliability $F$',
        )
