#!/bin/bash
#SBATCH --job-name=ctrl_sweep
#SBATCH --account=es_mohr
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-99

# ============================================================================
# Controller Seed Sweep - Euler HPC (ETH Zurich)
# ============================================================================
#
# This sweep tests 100 combinations of seed_target and seed_baseline
# to evaluate controller robustness across different random initializations.
#
# Usage:
#   1. Submit: sbatch sweep.sh
#   2. After completion, generate report: python generate_sweep_report.py
#
# Monitor:
#   squeue -u $USER                    # Check job status
#   tail -f logs/sweep_<jobid>_<taskid>.out  # Follow output
#
# Results:
#   - Individual runs: controller_optimization/checkpoints/sweep/<run_name>/
#   - Aggregated report: controller_optimization/checkpoints/sweep/sweep_report.pdf
# ============================================================================

# Exit on error
set -e

# Load required modules (Euler 2024 stack)
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

# Navigate to project root
cd $HOME/AZIMUTH

# Create logs directory if it doesn't exist
mkdir -p logs

# Read parameter combination for this array task
PARAMS_FILE="controller_optimization/sweep_params.txt"

# Check if params file exists
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file not found: $PARAMS_FILE"
    exit 1
fi

# Get the line corresponding to this array task (0-indexed)
# Skip empty lines and comments (lines starting with #)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS_FILE" | grep -v '^#' | grep -v '^$')

if [ -z "$LINE" ]; then
    echo "ERROR: No parameters found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse parameters from line (format: run_name param1=value1 param2=value2 ...)
RUN_NAME=$(echo "$LINE" | awk '{print $1}')
PARAMS=$(echo "$LINE" | cut -d' ' -f2-)

# Convert param=value format to --param value format
ARGS=""
for PARAM in $PARAMS; do
    KEY=$(echo "$PARAM" | cut -d'=' -f1)
    VALUE=$(echo "$PARAM" | cut -d'=' -f2)
    ARGS="$ARGS --$KEY $VALUE"
done

# Define output directory
OUTPUT_DIR="controller_optimization/checkpoints/sweep"

# Print job info
echo "=============================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Node:             $SLURM_NODELIST"
echo "GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo "Run name:         $RUN_NAME"
echo "Parameters:       $PARAMS"
echo "Output directory: $OUTPUT_DIR/$RUN_NAME"
echo "=============================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
fi

# Run training with parameters
echo "Starting training..."
echo "Command: python controller_optimization/train_controller.py --output_dir $OUTPUT_DIR --run_name $RUN_NAME $ARGS"
echo ""

python controller_optimization/train_controller.py \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    $ARGS

# Check exit status
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
