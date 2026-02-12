"""
Causal Chamber — Entry Point.

Runs all causal analyses and produces a PDF report.

Usage:
    # All analyses
    python -m causal_chamber.run_analysis --all --output_dir reports/causal_analysis/

    # Selective analyses
    python -m causal_chamber.run_analysis --discovery --interventional --output_dir reports/

    # With CausaliT checkpoint
    python -m causal_chamber.run_analysis --all --checkpoint path/to/casualit.ckpt --output_dir reports/
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Causal Chamber Analysis — Azimuth Pipeline',
    )
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    parser.add_argument('--discovery', action='store_true',
                        help='Run causal discovery (ground truth + attention + metrics)')
    parser.add_argument('--interventional', action='store_true',
                        help='Run interventional validation')
    parser.add_argument('--ood', action='store_true',
                        help='Run out-of-distribution analysis')
    parser.add_argument('--symbolic', action='store_true',
                        help='Run symbolic regression')
    parser.add_argument('--validation', action='store_true',
                        help='Run causal validation (p-value matrices + F)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to CausaliT checkpoint (.ckpt)')
    parser.add_argument('--output_dir', type=str, default='reports/causal_analysis',
                        help='Output directory for figures and report')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Compute device (cpu/cuda)')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of samples for analyses')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Attention threshold for adjacency')
    return parser.parse_args()


def run_discovery_analysis(args, output_dir: Path) -> dict:
    """Run causal discovery analysis."""
    print('\n' + '=' * 60)
    print('SECTION 1: Causal Graph Analysis')
    print('=' * 60)

    from causal_chamber.ground_truth import (
        get_ground_truth_edges, get_ground_truth_adjacency_parent_convention,
        get_all_observable_vars,
    )
    from causal_chamber.generate_data import sample_joint_pipeline
    from causal_chamber.metrics import compute_all_metrics, run_classical_baselines
    from causal_chamber.plotting import (
        plot_ground_truth_dag, plot_dag_comparison, plot_attention_heatmap,
        plot_metrics_bar_chart,
    )

    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    nodes = get_all_observable_vars()
    gt_edges = get_ground_truth_edges()
    gt_adj = get_ground_truth_adjacency_parent_convention(nodes, as_dataframe=False)

    print(f'  Observable variables: {len(nodes)}')
    print(f'  Ground truth edges: {len(gt_edges)}')

    # Plot ground truth DAG
    plot_ground_truth_dag(gt_edges, nodes, figure_dir)
    print('  Ground truth DAG plotted.')

    results = {
        'nodes': nodes,
        'gt_edges': gt_edges,
        'gt_adjacency': gt_adj,
        'metrics': {},
    }

    # Attention-based discovery (requires checkpoint)
    if args.checkpoint is not None:
        print(f'  Loading CausaliT checkpoint: {args.checkpoint}')
        try:
            from causal_chamber.attention_discovery import run_attention_discovery
            import torch

            # Generate joint trajectory data (all processes + F)
            data = sample_joint_pipeline(n=args.n_samples, seed=args.seed)

            # For now, we work with the raw attention matrices.
            # A full implementation would format data into ProT input tensors.
            # This is a simplified version for the discovery pipeline.

            from causal_chamber.attention_discovery import (
                load_forecaster, extract_lie_dag_probabilities,
            )

            model = load_forecaster(args.checkpoint, device=args.device)
            lie_dag = extract_lie_dag_probabilities(model)

            if lie_dag.get('enc_phi') is not None:
                enc_phi = lie_dag['enc_phi']
                # For multi-head, average across heads
                if enc_phi.ndim == 3:
                    enc_phi = enc_phi.mean(axis=0)
                from causal_chamber.attention_discovery import attention_to_adjacency
                est_adj = attention_to_adjacency(enc_phi, threshold=args.threshold)
                plot_attention_heatmap(enc_phi, nodes[:enc_phi.shape[0]], figure_dir,
                                      name='attention_heatmap',
                                      title='Encoder Self-Attention (LieAttention DAG)')
                print('  LieAttention DAG extracted and plotted.')
            else:
                print('  No LieAttention DAG found. Attention extraction requires '
                      'formatted input data (not yet implemented in standalone mode).')
                est_adj = None

            if est_adj is not None:
                # Compute metrics
                # est_adj may be smaller than gt_adj (only encoder vars)
                n_enc = est_adj.shape[0]
                gt_sub = gt_adj[:n_enc, :n_enc]
                att_metrics = compute_all_metrics(est_adj, gt_sub)
                results['metrics']['Attention'] = att_metrics
                results['estimated_adjacency'] = est_adj
                print(f'  Attention metrics: P={att_metrics["precision"]:.3f}, '
                      f'R={att_metrics["recall"]:.3f}, F1={att_metrics["f1"]:.3f}, '
                      f'SHD={att_metrics["shd"]}')

                # Plot DAG comparison
                est_edges = []
                enc_nodes = nodes[:n_enc]
                for i in range(n_enc):
                    for j in range(n_enc):
                        if est_adj[i, j]:
                            est_edges.append((enc_nodes[i], enc_nodes[j]))
                gt_edges_sub = [(u, v) for u, v in gt_edges
                                if u in enc_nodes and v in enc_nodes]
                plot_dag_comparison(gt_edges_sub, est_edges, enc_nodes,
                                   figure_dir, name='dag_comparison')

        except Exception as e:
            warnings.warn(f'Attention-based discovery failed: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('  No checkpoint provided. Skipping attention-based discovery.')

    # Classical baselines (optional)
    try:
        data = sample_joint_pipeline(n=args.n_samples, seed=args.seed)
        data_np = data[nodes[:len(data.columns)]].values if len(data.columns) >= len(nodes) else data.values
        baselines = run_classical_baselines(
            data_np, node_names=list(data.columns),
        )
        for method, adj in baselines.items():
            if adj is not None:
                n_min = min(adj.shape[0], gt_adj.shape[0])
                m = compute_all_metrics(adj[:n_min, :n_min], gt_adj[:n_min, :n_min])
                results['metrics'][method] = m
                print(f'  {method} metrics: P={m["precision"]:.3f}, '
                      f'R={m["recall"]:.3f}, F1={m["f1"]:.3f}, SHD={m["shd"]}')
    except Exception as e:
        print(f'  Classical baselines skipped: {e}')

    # Plot metrics bar chart
    if results['metrics']:
        plot_metrics_bar_chart(results['metrics'], figure_dir)

    return results


def run_interventional_validation(args, output_dir: Path) -> dict:
    """Run interventional validation analysis."""
    print('\n' + '=' * 60)
    print('SECTION 2: Interventional Validation')
    print('=' * 60)

    from causal_chamber.interventional_analysis import run_interventional_analysis
    from causal_chamber.plotting import plot_intervention_summary_violins

    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = run_interventional_analysis(
        n=args.n_samples, seed=args.seed, device=args.device,
    )

    # Collect all single-intervention results for violin plots
    all_results = []
    for proc_results in results['per_process'].values():
        all_results.extend(proc_results)

    if all_results:
        plot_intervention_summary_violins(all_results, figure_dir)
        print(f'  {len(all_results)} interventions analyzed.')

    if not results['summary_table'].empty:
        n_sig = results['summary_table']['significant'].sum()
        n_total = len(results['summary_table'])
        print(f'  Significant effects: {n_sig}/{n_total}')

    return results


def run_ood(args, output_dir: Path) -> dict:
    """Run OOD analysis."""
    print('\n' + '=' * 60)
    print('SECTION 3: OOD Robustness')
    print('=' * 60)

    from causal_chamber.ood_analysis import run_ood_analysis
    from causal_chamber.plotting import plot_ood_bar_chart

    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = run_ood_analysis(n=args.n_samples, seed=args.seed)

    # Build bar chart data
    id_metrics = {}
    ood_metrics = {}
    id_F_metrics = {}
    ood_F_metrics = {}
    for proc, res in results['per_process'].items():
        id_metrics[proc] = res['id_output_stats']
        ood_metrics[proc] = res['ood_output_stats']
        if 'id_F_stats' in res and 'ood_F_stats' in res:
            id_F_metrics[proc] = res['id_F_stats']
            ood_F_metrics[proc] = res['ood_F_stats']

    plot_ood_bar_chart(
        id_metrics, ood_metrics, figure_dir,
        id_F_metrics=id_F_metrics if id_F_metrics else None,
        ood_F_metrics=ood_F_metrics if ood_F_metrics else None,
    )

    for proc, res in results['per_process'].items():
        print(f'  {proc}: ID mean={res["id_output_stats"]["mean"]:.3f}, '
              f'OOD mean={res["ood_output_stats"]["mean"]:.3f}, '
              f'KS p={res["ks_pvalue"]:.2e}, '
              f'F ID={res["id_F_stats"]["mean"]:.3f}, '
              f'F OOD={res["ood_F_stats"]["mean"]:.3f}, '
              f'F KS p={res["ks_pvalue_F"]:.2e}')

    return results


def run_symbolic(args, output_dir: Path) -> dict:
    """Run symbolic regression analysis."""
    print('\n' + '=' * 60)
    print('SECTION 4: Symbolic Regression')
    print('=' * 60)

    from causal_chamber.symbolic_analysis import run_symbolic_analysis, PYSR_AVAILABLE
    from causal_chamber.plotting import plot_symbolic_equations_comparison

    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    if PYSR_AVAILABLE:
        print('  PySR available — using symbolic regression.')
    else:
        print('  PySR not available — using polynomial fit as fallback.')

    results = run_symbolic_analysis(
        n=args.n_samples, seed=args.seed, use_pysr=PYSR_AVAILABLE,
    )

    # Plot equations comparison
    plot_symbolic_equations_comparison(results['per_process'], figure_dir)

    for proc, res in results['per_process'].items():
        r2 = res['best_fit']['r2']
        method = res['best_method']
        print(f'  {proc}: {method} R²={r2:.4f}')

    return results


def run_validation(args, output_dir: Path) -> dict:
    """Run causal validation (p-value matrices + pipeline F)."""
    print('\n' + '=' * 60)
    print('SECTION 5: Causal Validation')
    print('=' * 60)

    from causal_chamber.causal_validation import (
        compute_pvalue_matrix, print_validation_table,
    )
    from causal_chamber.plotting import plot_validation_heatmap

    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = compute_pvalue_matrix(n=args.n_samples, seed=args.seed)
    print_validation_table(results)

    # Plot p-value heatmaps
    plot_validation_heatmap(results, figure_dir)
    print('  Validation heatmap plotted.')

    return results


def generate_report(
    output_dir: Path,
    discovery_results: dict = None,
    interventional_results: dict = None,
    ood_results: dict = None,
    symbolic_results: dict = None,
    validation_results: dict = None,
    skipped_analyses: list = None,
):
    """Generate the final PDF report."""
    print('\n' + '=' * 60)
    print('Generating PDF Report')
    print('=' * 60)

    from causal_chamber.report_generator import CausalAnalysisReportGenerator

    report_path = output_dir / 'causal_analysis_report.pdf'
    figure_dir = output_dir / 'figures'

    gen = CausalAnalysisReportGenerator(str(report_path))
    gen.add_title_page()

    # Executive summary
    analyses_run = []
    key_findings = []

    if discovery_results is not None:
        analyses_run.append('Causal Discovery')
        if discovery_results.get('metrics'):
            for method, m in discovery_results['metrics'].items():
                key_findings.append(
                    f'{method}: F1={m["f1"]:.3f}, SHD={m["shd"]}'
                )

    if interventional_results is not None:
        analyses_run.append('Interventional Validation')
        if not interventional_results.get('summary_table', None) is None:
            st = interventional_results['summary_table']
            if not st.empty and 'significant' in st.columns:
                n_sig = st['significant'].sum()
                key_findings.append(f'{n_sig} significant interventional effects detected.')

    if ood_results is not None:
        analyses_run.append('OOD Robustness')

    if symbolic_results is not None:
        analyses_run.append('Symbolic Regression')
        if not symbolic_results.get('summary_table', None) is None:
            st = symbolic_results['summary_table']
            if not st.empty and 'best_r2' in st.columns:
                mean_r2 = st['best_r2'].mean()
                key_findings.append(f'Symbolic regression mean R²={mean_r2:.4f}.')

    if validation_results is not None:
        analyses_run.append('Causal Validation')
        s = validation_results.get('summary', {})
        key_findings.append(
            f'{s.get("n_validated", 0)}/{s.get("n_total_checked", 0)} '
            f'edges validated ({s.get("validation_rate", 0):.0%}).'
        )

    n_vars = len(discovery_results.get('nodes', [])) if discovery_results else 0
    gen.add_executive_summary({
        'n_processes': 4,
        'n_variables': n_vars,
        'analyses_run': analyses_run,
        'key_findings': key_findings,
    })

    # Sections
    if discovery_results is not None:
        gen.add_discovery_section(discovery_results, figure_dir)

    if interventional_results is not None:
        gen.add_interventional_section(interventional_results, figure_dir)

    if ood_results is not None:
        gen.add_ood_section(ood_results, figure_dir)

    if symbolic_results is not None:
        gen.add_symbolic_section(symbolic_results, figure_dir)

    if validation_results is not None:
        gen.add_validation_section(validation_results, figure_dir)

    # Conclusions
    conclusions = {
        'summary': 'This report presents a comprehensive causal analysis of the '
                   'Azimuth manufacturing pipeline using attention-based causal '
                   'discovery, interventional validation, OOD robustness testing, '
                   'and symbolic regression.',
        'strengths': [],
        'weaknesses': [],
        'recommendations': [],
        'skipped': skipped_analyses or [],
    }

    if discovery_results and discovery_results.get('metrics'):
        best_f1 = max(m['f1'] for m in discovery_results['metrics'].values())
        if best_f1 > 0.7:
            conclusions['strengths'].append(
                f'Causal discovery achieves good F1 score ({best_f1:.3f}).'
            )
        else:
            conclusions['weaknesses'].append(
                f'Causal discovery F1 is moderate ({best_f1:.3f}). '
                'Consider adjusting the attention threshold or using LieAttention.'
            )

    if symbolic_results and not symbolic_results.get('summary_table', None) is None:
        st = symbolic_results['summary_table']
        if not st.empty and 'best_r2' in st.columns:
            low_r2 = st[st['best_r2'] < 0.9]
            if len(low_r2) > 0:
                procs = ', '.join(low_r2['process'].values)
                conclusions['weaknesses'].append(
                    f'Symbolic regression has low R² for: {procs}. '
                    'These processes may require more complex expressions.'
                )

    conclusions['recommendations'].append(
        'Train CausaliT with LieAttention for interpretable DAG learning.'
    )
    conclusions['recommendations'].append(
        'Install PySR for better symbolic regression results.'
    )

    gen.add_conclusions(conclusions)
    gen.build()

    print(f'  Report saved to: {report_path}')
    return report_path


def main():
    args = parse_args()

    # Determine which analyses to run
    run_all = args.all
    if not (run_all or args.discovery or args.interventional or args.ood or args.symbolic or args.validation):
        print('No analysis specified. Use --all or specific flags. Use --help for options.')
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Causal Chamber Analysis — Azimuth Pipeline')
    print('=' * 60)
    print(f'  Output directory: {output_dir}')
    print(f'  Checkpoint: {args.checkpoint or "None (attention analyses skipped)"}')
    print(f'  Device: {args.device}')
    print(f'  Samples: {args.n_samples}')
    print(f'  Seed: {args.seed}')

    discovery_results = None
    interventional_results = None
    ood_results = None
    symbolic_results = None
    validation_results = None
    skipped = []

    # Run analyses
    if run_all or args.discovery:
        try:
            discovery_results = run_discovery_analysis(args, output_dir)
        except Exception as e:
            warnings.warn(f'Discovery analysis failed: {e}')
            import traceback
            traceback.print_exc()
            skipped.append(f'Causal Discovery (error: {e})')

    if run_all or args.interventional:
        try:
            interventional_results = run_interventional_validation(args, output_dir)
        except Exception as e:
            warnings.warn(f'Interventional analysis failed: {e}')
            import traceback
            traceback.print_exc()
            skipped.append(f'Interventional Validation (error: {e})')

    if run_all or args.ood:
        try:
            ood_results = run_ood(args, output_dir)
        except Exception as e:
            warnings.warn(f'OOD analysis failed: {e}')
            import traceback
            traceback.print_exc()
            skipped.append(f'OOD Analysis (error: {e})')

    if run_all or args.symbolic:
        try:
            symbolic_results = run_symbolic(args, output_dir)
        except Exception as e:
            warnings.warn(f'Symbolic analysis failed: {e}')
            import traceback
            traceback.print_exc()
            skipped.append(f'Symbolic Regression (error: {e})')

    if run_all or args.validation:
        try:
            validation_results = run_validation(args, output_dir)
        except Exception as e:
            warnings.warn(f'Causal validation failed: {e}')
            import traceback
            traceback.print_exc()
            skipped.append(f'Causal Validation (error: {e})')

    # Generate report
    try:
        report_path = generate_report(
            output_dir,
            discovery_results=discovery_results,
            interventional_results=interventional_results,
            ood_results=ood_results,
            symbolic_results=symbolic_results,
            validation_results=validation_results,
            skipped_analyses=skipped,
        )
    except Exception as e:
        warnings.warn(f'Report generation failed: {e}')
        import traceback
        traceback.print_exc()

    print('\n' + '=' * 60)
    print('Analysis complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
