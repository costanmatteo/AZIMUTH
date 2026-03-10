#!/bin/bash
#SBATCH --job-name=complexity_sweep
#SBATCH --account=es_mohr
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/complexity_%A_%a.out
#SBATCH --error=logs/complexity_%A_%a.err
#SBATCH --array=0-749

# ============================================================================
# Dataset Complexity Sensitivity Sweep - Euler HPC (ETH Zurich)
# ============================================================================
#
# Tests how controller win rate varies with dataset complexity parameters
# (n, m, rho) using Latin Hypercube Sampling + reduced seed grid.
#
# Default: 30 LHS configs × 25 seed pairs (5×5) = 750 runs
#
# Resources: CPU only, 1 core, 4GB RAM, 15 min per job
# (more RAM and time than seed sweep because larger n/m = bigger models)
#
# Usage:
#   1. Generate params: python generate_complexity_sweep_params.py
#   2. Submit: sbatch complexity_sweep.sh
#   3. After completion: python generate_complexity_sweep_report.py
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/complexity_<jobid>_<taskid>.out
# ============================================================================

set -e

# Load required modules (Euler 2024 stack)
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

# Navigate to project root
cd $HOME/AZIMUTH

mkdir -p logs

# Read parameter combination for this array task
PARAMS_FILE="controller_optimization/complexity_sweep_params.txt"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file not found: $PARAMS_FILE"
    echo "Run: python controller_optimization/generate_complexity_sweep_params.py"
    exit 1
fi

# Get the line corresponding to this array task (skip comments and empty lines)
LINE=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$LINE" ]; then
    echo "ERROR: No parameters found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse parameters from line
RUN_NAME=$(echo "$LINE" | awk '{print $1}')
PARAMS=$(echo "$LINE" | cut -d' ' -f2-)

# Convert param=value format to --param value format
ARGS=""
for PARAM in $PARAMS; do
    KEY=$(echo "$PARAM" | cut -d'=' -f1)
    VALUE=$(echo "$PARAM" | cut -d'=' -f2)
    ARGS="$ARGS --$KEY $VALUE"
done

OUTPUT_DIR="controller_optimization/checkpoints/complexity_sweep"

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

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run training with ST complexity overrides + seed parameters
# --no_pdf to save time (report generated after sweep)
echo "Starting training..."
echo "Command: python controller_optimization/train_controller.py --output_dir $OUTPUT_DIR --run_name $RUN_NAME --no_pdf $ARGS"
echo ""

python controller_optimization/train_controller.py \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --no_pdf \
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
