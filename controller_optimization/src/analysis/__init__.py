"""Analysis module for controller optimization."""

from .theoretical_loss_analysis import (
    TheoreticalLossComponents,
    TheoreticalLossTracker,
    compute_theoretical_L_min,
    compute_theoretical_E_F,
    compute_theoretical_E_F2,
    compute_multi_process_L_min,
    compute_effective_params_from_trajectory,
    estimate_effective_params_simple,
    run_validation_sampling,
    compute_z_score,
    format_status
)

from .theoretical_visualization import (
    plot_loss_vs_L_min,
    plot_efficiency_over_time,
    plot_loss_decomposition,
    plot_loss_scatter,
    plot_empirical_vs_theoretical,
    create_summary_figure,
    generate_all_theoretical_plots
)

from .theoretical_tables import (
    generate_main_results_table,
    generate_process_params_table,
    generate_decomposition_table,
    generate_efficiency_table,
    generate_validation_table,
    generate_full_report,
    save_report_txt,
    save_report_json
)

__all__ = [
    # Core analysis
    'TheoreticalLossComponents',
    'TheoreticalLossTracker',
    'compute_theoretical_L_min',
    'compute_theoretical_E_F',
    'compute_theoretical_E_F2',
    'compute_multi_process_L_min',
    'compute_effective_params_from_trajectory',
    'estimate_effective_params_simple',
    'run_validation_sampling',
    'compute_z_score',
    'format_status',
    # Visualization
    'plot_loss_vs_L_min',
    'plot_efficiency_over_time',
    'plot_loss_decomposition',
    'plot_loss_scatter',
    'plot_empirical_vs_theoretical',
    'create_summary_figure',
    'generate_all_theoretical_plots',
    # Tables
    'generate_main_results_table',
    'generate_process_params_table',
    'generate_decomposition_table',
    'generate_efficiency_table',
    'generate_validation_table',
    'generate_full_report',
    'save_report_txt',
    'save_report_json'
]
