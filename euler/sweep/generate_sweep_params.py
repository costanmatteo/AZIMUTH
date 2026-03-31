#!/usr/bin/env python3
"""
Generate sweep_params.txt and sweep.sh from configs/sweep_config.py.

Usage:
    cd ~/AZIMUTH
    python euler/sweep/generate_sweep_params.py
"""

import sys
import itertools
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.sweep_config import SWEEP_CONFIG


def generate_sweep():
    cfg = SWEEP_CONFIG
    params = cfg['params']
    output_cfg = cfg['output']
    slurm_cfg = cfg['slurm']
    env_cfg = cfg['environment']
    fixed = cfg.get('fixed_params', {})

    output_dir = output_cfg['output_dir']
    params_file = REPO_ROOT / output_cfg['params_file']
    sweep_script = REPO_ROOT / output_cfg['sweep_script']
    run_name_template = output_cfg['run_name_template']

    # Generate cartesian product of all parameter values
    param_names = list(params.keys())
    param_values = [params[k] for k in param_names]
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    print(f"Sweep Configuration")
    print(f"{'='*50}")
    for name in param_names:
        print(f"  {name}: {params[name]} ({len(params[name])} values)")
    print(f"  Total combinations: {total}")
    print()

    # ---- Generate sweep_params.txt ----
    header = [
        "# Controller Sweep Configuration (auto-generated)",
        "# ====================================",
        "#",
        f"# {total} combinations from: {' x '.join(f'{k}({len(params[k])})' for k in param_names)}",
        "#",
        f"# Format: run_name {' '.join(f'{k}=X' for k in param_names)}",
        "#",
        "# ====================================",
        "",
    ]

    data_lines = []
    for combo in combinations:
        values = dict(zip(param_names, combo))
        run_name = run_name_template.format(**values)
        param_str = " ".join(f"{k}={v}" for k, v in values.items())
        data_lines.append(f"{run_name} {param_str}")

    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        f.write("\n".join(header + data_lines) + "\n")
    print(f"Written: {params_file} ({total} combinations)")

    # ---- Generate sweep.sh ----
    modules_line = " ".join(env_cfg['modules'])

    # Build fixed args string
    fixed_args_parts = []
    for k, v in fixed.items():
        if isinstance(v, bool) and v:
            fixed_args_parts.append(f"--{k}")
        elif not isinstance(v, bool):
            fixed_args_parts.append(f"--{k} {v}")
    fixed_args_str = " ".join(fixed_args_parts)

    # Venv activation block
    venv = env_cfg.get('venv')
    if venv:
        venv_block = f'source {venv}/bin/activate'
    else:
        venv_block = (
            'if [ -d "venv" ]; then\n'
            '    source venv/bin/activate\n'
            'elif [ -d ".venv" ]; then\n'
            '    source .venv/bin/activate\n'
            'fi'
        )

    script = f"""#!/bin/bash
#SBATCH --job-name={slurm_cfg['job_name']}
#SBATCH --account={slurm_cfg['account']}
#SBATCH --time={slurm_cfg['time']}
#SBATCH --ntasks={slurm_cfg['ntasks']}
#SBATCH --cpus-per-task={slurm_cfg['cpus_per_task']}
#SBATCH --mem-per-cpu={slurm_cfg['mem_per_cpu']}
#SBATCH --output={slurm_cfg['output_log']}
#SBATCH --error={slurm_cfg['error_log']}
#SBATCH --array=0-{total - 1}

# ============================================================================
# Controller Sweep - Euler HPC (auto-generated from configs/sweep_config.py)
# ============================================================================
#
# {total} runs = {' x '.join(f'{k}({len(params[k])})' for k in param_names)}
#
# Usage:
#   1. Generate: python euler/sweep/generate_sweep_params.py
#   2. Submit:   sbatch euler/sweep/sweep.sh
#   3. Report:   python euler/sweep/generate_sweep_report.py
# ============================================================================

set -e

module load {modules_line}

cd $HOME/AZIMUTH

mkdir -p logs

{venv_block}

# Read parameter combination for this array task
PARAMS_FILE="{output_cfg['params_file']}"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file not found: $PARAMS_FILE"
    exit 1
fi

LINE=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$LINE" ]; then
    echo "ERROR: No parameters found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

RUN_NAME=$(echo "$LINE" | awk '{{print $1}}')
PARAMS=$(echo "$LINE" | cut -d' ' -f2-)

ARGS=""
for PARAM in $PARAMS; do
    KEY=$(echo "$PARAM" | cut -d'=' -f1)
    VALUE=$(echo "$PARAM" | cut -d'=' -f2)
    ARGS="$ARGS --$KEY $VALUE"
done

OUTPUT_DIR="{output_dir}"

echo "=============================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Node:             $SLURM_NODELIST"
echo "=============================================="
echo "Run name:         $RUN_NAME"
echo "Parameters:       $PARAMS"
echo "Output directory: $OUTPUT_DIR/$RUN_NAME"
echo "=============================================="
echo ""

if [ -d "$OUTPUT_DIR/$RUN_NAME" ]; then
    echo "Removing previous run data: $OUTPUT_DIR/$RUN_NAME"
    rm -rf "$OUTPUT_DIR/$RUN_NAME"
fi

echo "Starting training..."
echo "Command: python train_controller.py --output_dir $OUTPUT_DIR --run_name $RUN_NAME {fixed_args_str} $ARGS"
echo ""

python train_controller.py \\
    --output_dir "$OUTPUT_DIR" \\
    --run_name "$RUN_NAME" \\
    {fixed_args_str} \\
    $ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/$RUN_NAME/"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "ERROR: Training failed!"
    echo "=============================================="
    exit 1
fi
"""

    with open(sweep_script, 'w') as f:
        f.write(script)
    print(f"Written: {sweep_script} (--array=0-{total - 1})")


if __name__ == '__main__':
    generate_sweep()
