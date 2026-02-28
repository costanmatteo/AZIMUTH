"""Analysis module for controller optimization."""

from .theoretical_loss_analysis import (
    EmpiricalStats,
    TheoreticalLossTracker,
    compute_empirical_stats,
    run_validation_sampling,
    compute_z_score,
    format_status
)

from .theoretical_visualization import (
    plot_observed_loss,
    plot_loss_decomposition,
    plot_empirical_vs_theoretical,
    create_summary_figure,
    generate_all_theoretical_plots
)

from .theoretical_tables import (
    generate_main_results_table,
    generate_process_params_table,
    generate_decomposition_table,
    generate_validation_table,
    generate_full_report,
    save_report_txt,
    save_report_json
)

from .bellman_lmin import (
    BellmanConfig,
    BellmanLminResult,
    compute_bellman_lmin,
    estimate_noise_covariance,
    compute_manifold,
    backward_induction,
    forward_simulation,
)

__all__ = [
    # Core analysis
    'EmpiricalStats',
    'TheoreticalLossTracker',
    'compute_empirical_stats',
    'run_validation_sampling',
    'compute_z_score',
    'format_status',
    # Visualization
    'plot_observed_loss',
    'plot_loss_decomposition',
    'plot_empirical_vs_theoretical',
    'create_summary_figure',
    'generate_all_theoretical_plots',
    # Tables
    'generate_main_results_table',
    'generate_process_params_table',
    'generate_decomposition_table',
    'generate_validation_table',
    'generate_full_report',
    'save_report_txt',
    'save_report_json',
    # Bellman L_min
    'BellmanConfig',
    'BellmanLminResult',
    'compute_bellman_lmin',
    'estimate_noise_covariance',
    'compute_manifold',
    'backward_induction',
    'forward_simulation',
]
