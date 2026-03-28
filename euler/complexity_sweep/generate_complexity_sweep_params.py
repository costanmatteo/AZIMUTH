#!/usr/bin/env python3
"""
Generate sweep_params.txt for dataset complexity sensitivity analysis.

Uses Latin Hypercube Sampling (LHS) over ST complexity parameters
(n, m, rho, n_processes) combined with a reduced seed grid for each configuration.

Usage:
    python generate_complexity_sweep_params.py [options]

Output:
    complexity_sweep_params.txt with format:
        run_name st_n=X st_m=Y st_rho=Z st_n_processes=P seed_target=A seed_baseline=B
"""

import argparse
import numpy as np
from pathlib import Path


def latin_hypercube_sample(n_samples: int, ranges: dict, seed: int = 42) -> list:
    """
    Generate Latin Hypercube samples for mixed integer/continuous parameters.

    Args:
        n_samples: Number of LHS configurations to generate
        ranges: Dict of {param_name: (min, max, type)} where type is 'int' or 'float'
        seed: Random seed for reproducibility

    Returns:
        List of dicts, each mapping param_name -> sampled value
    """
    rng = np.random.default_rng(seed)
    n_params = len(ranges)
    param_names = list(ranges.keys())

    # Standard LHS: stratified sampling in [0,1]^d
    samples_unit = np.zeros((n_samples, n_params))
    for j in range(n_params):
        # Divide [0,1] into n_samples equal intervals, sample one point per interval
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            samples_unit[perm[i], j] = (i + rng.uniform()) / n_samples

    # Map to actual parameter ranges
    configs = []
    for i in range(n_samples):
        config = {}
        for j, name in enumerate(param_names):
            lo, hi, ptype = ranges[name]
            val = lo + samples_unit[i, j] * (hi - lo)
            if ptype == 'int':
                val = int(round(val))
                val = max(lo, min(hi, val))  # clamp
            else:
                val = round(val, 4)
            config[name] = val
        configs.append(config)

    return configs


def generate_complexity_sweep(
    n_lhs: int = 30,
    n_seeds: int = 5,
    output_file: Path = None,
    seed: int = 42,
    n_range: tuple = (2, 8),
    m_range: tuple = (1, 4),
    rho_range: tuple = (0.0, 0.5),
    nproc_range: tuple = (2, 5),
):
    """
    Generate complexity sweep parameter file.

    Each LHS configuration is tested with n_seeds^2 seed combinations.
    Total runs = n_lhs * n_seeds^2.

    Args:
        n_lhs: Number of LHS configurations for (n, m, rho, n_processes)
        n_seeds: Number of seed values (creates n_seeds^2 seed pairs per config)
        output_file: Output file path
        seed: Random seed for LHS
        n_range: Range for ST n parameter (input variables)
        m_range: Range for ST m parameter (cascaded stages)
        rho_range: Range for ST rho parameter (noise intensity)
        nproc_range: Range for number of processes in sequence
    """
    if output_file is None:
        output_file = Path(__file__).parent / "complexity_sweep_params.txt"

    # Generate LHS configurations
    ranges = {
        'n': (n_range[0], n_range[1], 'int'),
        'm': (m_range[0], m_range[1], 'int'),
        'rho': (rho_range[0], rho_range[1], 'float'),
        'n_processes': (nproc_range[0], nproc_range[1], 'int'),
    }
    configs = latin_hypercube_sample(n_lhs, ranges, seed=seed)

    # Enforce constraint: m <= n (required by ST SCM)
    for cfg in configs:
        if cfg['m'] > cfg['n']:
            cfg['m'] = cfg['n']

    # Generate reduced seed grid
    seed_step = max(1, 100 // n_seeds)
    seed_values = list(range(1, 100, seed_step))[:n_seeds]

    total_runs = len(configs) * len(seed_values) ** 2

    print(f"Complexity Sensitivity Sweep")
    print(f"{'='*50}")
    print(f"  LHS configurations:  {len(configs)}")
    print(f"  Seed pairs per config: {len(seed_values)}^2 = {len(seed_values)**2}")
    print(f"  Total runs:          {total_runs}")
    print(f"  Parameter ranges:")
    print(f"    n (inputs):       [{n_range[0]}, {n_range[1]}]")
    print(f"    m (stages):       [{m_range[0]}, {m_range[1]}]")
    print(f"    rho (noise):      [{rho_range[0]}, {rho_range[1]}]")
    print(f"    n_processes:      [{nproc_range[0]}, {nproc_range[1]}]")
    print(f"  Seed values:     {seed_values}")
    print()

    # Print sampled configurations
    print("LHS configurations:")
    for i, cfg in enumerate(configs):
        print(f"  [{i:2d}] n={cfg['n']}, m={cfg['m']}, rho={cfg['rho']:.4f}, n_proc={cfg['n_processes']}")

    lines = [
        "# Complexity Sensitivity Sweep Configuration",
        "# ============================================",
        "#",
        f"# {total_runs} total runs = {len(configs)} LHS configs x {len(seed_values)**2} seed pairs",
        f"# Parameters: n in [{n_range[0]},{n_range[1]}], m in [{m_range[0]},{m_range[1]}], "
        f"rho in [{rho_range[0]},{rho_range[1]}], n_processes in [{nproc_range[0]},{nproc_range[1]}]",
        f"# Seeds: {seed_values}",
        "#",
        "# Format: run_name st_n=X st_m=Y st_rho=Z st_n_processes=P seed_target=A seed_baseline=B",
        "#",
        f"# Update complexity_sweep.sh: --array=0-{total_runs - 1}",
        "# ============================================",
        "",
    ]

    for cfg_idx, cfg in enumerate(configs):
        for seed_t in seed_values:
            for seed_b in seed_values:
                run_name = (
                    f"cfg{cfg_idx:02d}_n{cfg['n']}_m{cfg['m']}"
                    f"_p{cfg['n_processes']}_r{cfg['rho']:.2f}"
                    f"_t{seed_t:02d}_b{seed_b:02d}"
                )
                line = (
                    f"{run_name} "
                    f"st_n={cfg['n']} st_m={cfg['m']} st_rho={cfg['rho']} "
                    f"st_n_processes={cfg['n_processes']} "
                    f"seed_target={seed_t} seed_baseline={seed_b}"
                )
                lines.append(line)

    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWritten to: {output_file}")
    print(f"\nRemember to update complexity_sweep.sh:")
    print(f"  #SBATCH --array=0-{total_runs - 1}")


def _load_config():
    """Load defaults from configs/complexity_sweep_config.py if available."""
    try:
        import sys, importlib.util
        cfg_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'complexity_sweep_config.py'
        if not cfg_path.exists():
            return {}
        spec = importlib.util.spec_from_file_location('complexity_sweep_config', cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.COMPLEXITY_SWEEP_CONFIG
    except Exception:
        return {}


def main():
    cfg = _load_config()
    sampling = cfg.get('sampling', {})
    pr = cfg.get('param_ranges', {})

    parser = argparse.ArgumentParser(
        description='Generate complexity sweep parameters (LHS + reduced seeds)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_lhs', type=int, default=sampling.get('n_lhs', 30),
                        help='Number of LHS configurations for (n, m, rho)')
    parser.add_argument('--n_seeds', type=int, default=sampling.get('n_seeds', 5),
                        help='Number of seed values per axis (total pairs = n_seeds^2)')
    parser.add_argument('--seed', type=int, default=sampling.get('seed', 42),
                        help='Random seed for LHS generation')
    parser.add_argument('--output', type=str,
                        default=cfg.get('output', {}).get('params_file'),
                        help='Output file path')

    # Parameter ranges
    n_r = pr.get('n', (2, 8))
    m_r = pr.get('m', (1, 4))
    rho_r = pr.get('rho', (0.0, 0.5))
    np_r = pr.get('n_processes', (2, 5))
    parser.add_argument('--n_min', type=int, default=n_r[0],
                        help='Min ST input variables (n)')
    parser.add_argument('--n_max', type=int, default=n_r[1],
                        help='Max ST input variables (n)')
    parser.add_argument('--m_min', type=int, default=m_r[0],
                        help='Min ST cascaded stages (m)')
    parser.add_argument('--m_max', type=int, default=m_r[1],
                        help='Max ST cascaded stages (m)')
    parser.add_argument('--rho_min', type=float, default=rho_r[0],
                        help='Min ST noise intensity (rho)')
    parser.add_argument('--rho_max', type=float, default=rho_r[1],
                        help='Max ST noise intensity (rho)')
    parser.add_argument('--nproc_min', type=int, default=np_r[0],
                        help='Min number of processes in sequence')
    parser.add_argument('--nproc_max', type=int, default=np_r[1],
                        help='Max number of processes in sequence')

    args = parser.parse_args()

    output_file = Path(args.output) if args.output else None
    generate_complexity_sweep(
        n_lhs=args.n_lhs,
        n_seeds=args.n_seeds,
        output_file=output_file,
        seed=args.seed,
        n_range=(args.n_min, args.n_max),
        m_range=(args.m_min, args.m_max),
        rho_range=(args.rho_min, args.rho_max),
        nproc_range=(args.nproc_min, args.nproc_max),
    )


if __name__ == '__main__':
    main()
