#!/usr/bin/env python3
"""
Automated sweep: runs the full pipeline with a single command.

Steps:
    1. Generate dataset              (generate_dataset.py)
    2. Train uncertainty predictors  (train_predictor.py)
    3. Train surrogate               (train_surrogate.py — only if surrogate.type='casualit')
    4. Run controller sweep           (train_controller.py × N combinations)

Usage:
    python run_sweep.py                          # full pipeline + sweep
    python run_sweep.py --skip-dataset           # skip step 1 (dataset already exists)
    python run_sweep.py --skip-predictor         # skip step 2
    python run_sweep.py --skip-surrogate         # skip step 3 (even if casualit)
    python run_sweep.py --controller-only        # skip steps 1-3, only run sweep
    python run_sweep.py --max-parallel 4         # run up to 4 controller jobs in parallel
    python run_sweep.py --dry-run                # show what would run, don't execute

Extra args after '--' are forwarded to ALL train_controller.py calls:
    python run_sweep.py -- --device cuda --no_pdf
"""

import sys
import os
import subprocess
import itertools
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


def _detect_surrogate_type():
    """Check controller_config to see if casualit surrogate is configured."""
    from configs.controller_config import CONTROLLER_CONFIG
    return CONTROLLER_CONFIG.get('surrogate', {}).get('type', 'reliability_function')


def _build_sweep_combinations():
    """Read sweep_config and return list of (run_name, param_dict) tuples."""
    from configs.sweep_config import SWEEP_CONFIG

    cfg = SWEEP_CONFIG
    params = cfg['params']
    output_cfg = cfg['output']
    fixed = cfg.get('fixed_params', {})
    run_name_template = output_cfg['run_name_template']
    output_dir = output_cfg['output_dir']

    param_names = list(params.keys())
    param_values = [params[k] for k in param_names]
    combinations = list(itertools.product(*param_values))

    runs = []
    for combo in combinations:
        values = dict(zip(param_names, combo))
        run_name = run_name_template.format(**values)
        # Merge swept params + fixed params
        all_params = dict(values)
        all_params.update(fixed)
        runs.append((run_name, all_params, output_dir))

    return runs


def _run_step(description, cmd, dry_run=False):
    """Run a pipeline step, abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [dry-run] Skipped.")
        return

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)

    print(f"\n  Done in {elapsed:.1f}s")


def _build_controller_cmd(run_name, params, output_dir, extra_args):
    """Build the train_controller.py command for one sweep run."""
    cmd = [sys.executable, 'train_controller.py',
           '--output_dir', output_dir,
           '--run_name', run_name]

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    cmd.extend(extra_args)
    return cmd


def _run_single_controller(args_tuple):
    """Run a single controller training (for use with ProcessPoolExecutor)."""
    run_name, cmd, idx, total = args_tuple
    print(f"\n  [{idx+1}/{total}] Starting: {run_name}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "OK" if success else "FAILED"
    print(f"  [{idx+1}/{total}] {run_name}: {status} ({elapsed:.1f}s)")
    return run_name, success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Run the full sweep pipeline with a single command.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset generation (step 1)')
    parser.add_argument('--skip-predictor', action='store_true',
                        help='Skip predictor training (step 2)')
    parser.add_argument('--skip-surrogate', action='store_true',
                        help='Skip surrogate training (step 3) even if casualit is configured')
    parser.add_argument('--controller-only', action='store_true',
                        help='Skip steps 1-3, only run the controller sweep')
    parser.add_argument('--max-parallel', type=int, default=1,
                        help='Max parallel controller jobs (default: 1 = sequential)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--clean', action='store_true',
                        help='Remove previous sweep results before starting')

    # Separate extra args after '--' to forward to train_controller.py
    argv = sys.argv[1:]
    extra_args = []
    if '--' in argv:
        split_idx = argv.index('--')
        extra_args = argv[split_idx + 1:]
        argv = argv[:split_idx]

    args = parser.parse_args(argv)

    if args.controller_only:
        args.skip_dataset = True
        args.skip_predictor = True
        args.skip_surrogate = True

    surrogate_type = _detect_surrogate_type()
    runs = _build_sweep_combinations()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  AZIMUTH — Automated Sweep Pipeline")
    print("=" * 60)
    print(f"  Surrogate type : {surrogate_type}")
    print(f"  Sweep runs     : {len(runs)}")
    print(f"  Parallelism    : {args.max_parallel}")
    if extra_args:
        print(f"  Extra args     : {' '.join(extra_args)}")
    print()
    print("  Steps:")
    print(f"    1. Generate dataset       {'[SKIP]' if args.skip_dataset else '[RUN]'}")
    print(f"    2. Train predictors       {'[SKIP]' if args.skip_predictor else '[RUN]'}")
    if surrogate_type == 'casualit':
        print(f"    3. Train surrogate        {'[SKIP]' if args.skip_surrogate else '[RUN]'}")
    else:
        print(f"    3. Train surrogate        [SKIP — using {surrogate_type}]")
    print(f"    4. Controller sweep       [RUN] ({len(runs)} combinations)")
    print()

    # ── Clean ────────────────────────────────────────────────────────
    if args.clean and not args.dry_run:
        from configs.sweep_config import SWEEP_CONFIG
        sweep_dir = REPO_ROOT / SWEEP_CONFIG['output']['output_dir']
        if sweep_dir.exists():
            import shutil
            print(f"[clean] Removing {sweep_dir}")
            shutil.rmtree(sweep_dir)

    # ── Step 1: Generate dataset ─────────────────────────────────────
    if not args.skip_dataset:
        _run_step(
            "[1/4] Generating dataset",
            [sys.executable, 'generate_dataset.py'],
            dry_run=args.dry_run,
        )

    # ── Step 2: Train uncertainty predictors ─────────────────────────
    if not args.skip_predictor:
        _run_step(
            "[2/4] Training uncertainty predictors",
            [sys.executable, 'train_predictor.py'],
            dry_run=args.dry_run,
        )

    # ── Step 3: Train surrogate (only if casualit) ───────────────────
    if surrogate_type == 'casualit' and not args.skip_surrogate:
        _run_step(
            "[3/4] Training CasualiT surrogate",
            [sys.executable, 'train_surrogate.py'],
            dry_run=args.dry_run,
        )

    # ── Step 4: Controller sweep ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [4/4] Running controller sweep ({len(runs)} combinations)")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, (run_name, params, output_dir) in enumerate(runs):
            cmd = _build_controller_cmd(run_name, params, output_dir, extra_args)
            print(f"  [{i+1}/{len(runs)}] {run_name}")
            print(f"         {' '.join(cmd)}")
        print(f"\n  [dry-run] {len(runs)} runs would be executed.")
        return

    total = len(runs)
    start_all = time.time()
    results = []

    if args.max_parallel <= 1:
        # Sequential execution
        for i, (run_name, params, output_dir) in enumerate(runs):
            cmd = _build_controller_cmd(run_name, params, output_dir, extra_args)
            _, success, elapsed = _run_single_controller(
                (run_name, cmd, i, total)
            )
            results.append((run_name, success, elapsed))
    else:
        # Parallel execution
        tasks = []
        for i, (run_name, params, output_dir) in enumerate(runs):
            cmd = _build_controller_cmd(run_name, params, output_dir, extra_args)
            tasks.append((run_name, cmd, i, total))

        with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = {executor.submit(_run_single_controller, t): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - start_all
    succeeded = sum(1 for _, s, _ in results if s)
    failed = total - succeeded

    print(f"\n{'='*60}")
    print(f"  Sweep Complete")
    print(f"{'='*60}")
    print(f"  Total runs : {total}")
    print(f"  Succeeded  : {succeeded}")
    print(f"  Failed     : {failed}")
    print(f"  Total time : {total_time:.1f}s ({total_time/60:.1f}m)")
    print()

    if failed > 0:
        print("  Failed runs:")
        for name, success, _ in results:
            if not success:
                print(f"    - {name}")
        print()
        sys.exit(1)

    print("  All runs completed successfully!")
    print(f"  Results in: checkpoints/sweep/")
    print(f"\n  Next: python euler/sweep/generate_sweep_report.py")


if __name__ == '__main__':
    main()
