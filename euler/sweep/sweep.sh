#!/bin/bash
#SBATCH --job-name=ctrl_sweep
#SBATCH --account=es_mohr
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-249

# ============================================================================
# Controller Sweep - Euler HPC (auto-generated from configs/sweep_config.py)
# ============================================================================
#
# 250 runs = seed_target(25) x seed_baseline(10)
#
# Usage:
#   1. Generate: python euler/sweep/generate_sweep_params.py
#   2. Submit:   sbatch euler/sweep/sweep.sh
#   3. Report:   python euler/sweep/generate_sweep_report.py
# ============================================================================

set -e

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

cd $HOME/AZIMUTH

mkdir -p logs

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Read parameter combination for this array task
PARAMS_FILE="euler/sweep/sweep_params.txt"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file not found: $PARAMS_FILE"
    exit 1
fi

LINE=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$LINE" ]; then
    echo "ERROR: No parameters found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

RUN_NAME=$(echo "$LINE" | awk '{print $1}')
PARAMS=$(echo "$LINE" | cut -d' ' -f2-)

ARGS=""
for PARAM in $PARAMS; do
    KEY=$(echo "$PARAM" | cut -d'=' -f1)
    VALUE=$(echo "$PARAM" | cut -d'=' -f2)
    ARGS="$ARGS --$KEY $VALUE"
done

OUTPUT_DIR="checkpoints/sweep"

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

echo "Starting training..."
echo "Command: python train_controller.py --output_dir $OUTPUT_DIR --run_name $RUN_NAME --quiet $ARGS"
echo ""

python train_controller.py \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --quiet \
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
