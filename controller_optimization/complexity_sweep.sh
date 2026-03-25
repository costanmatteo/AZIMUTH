#!/bin/bash
#SBATCH --job-name=complexity_sweep
#SBATCH --account=es_mohr
#SBATCH --time=00:20:00
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
# (n, m, rho, n_processes) using Latin Hypercube Sampling + reduced seed grid.
#
# Default: 30 LHS configs x 25 seed pairs (5x5) = 750 runs
#
# Each job:
#   1. Generates dataset for the ST config (skip if done)
#   2. Trains uncertainty predictors (UPs) for its ST config (skip if done)
#   3. Trains controller with the given seed pair
#
# Data and UPs are saved in config-specific directories to avoid conflicts
# between parallel jobs with different ST parameters.
#
# Resources: CPU only, 1 core, 4GB RAM, 20 min per job
#
# Usage:
#   1. Generate params: python controller_optimization/generate_complexity_sweep_params.py
#   2. Submit: sbatch controller_optimization/complexity_sweep.sh
#   3. After completion: python controller_optimization/generate_complexity_sweep_report.py
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

# Extract individual parameter values
ST_N=$(echo "$PARAMS" | grep -oP 'st_n=\K[^ ]+')
ST_M=$(echo "$PARAMS" | grep -oP 'st_m=\K[^ ]+')
ST_RHO=$(echo "$PARAMS" | grep -oP 'st_rho=\K[^ ]+')
ST_NPROC=$(echo "$PARAMS" | grep -oP 'st_n_processes=\K[^ ]+')
SEED_T=$(echo "$PARAMS" | grep -oP 'seed_target=\K[^ ]+')
SEED_B=$(echo "$PARAMS" | grep -oP 'seed_baseline=\K[^ ]+')

# Config-specific directories (shared across seed pairs with same ST config)
DATA_DIR="controller_optimization/checkpoints/complexity_sweep/data_n${ST_N}_m${ST_M}_p${ST_NPROC}_r${ST_RHO}"
UP_DIR="controller_optimization/checkpoints/complexity_sweep/up_n${ST_N}_m${ST_M}_p${ST_NPROC}_r${ST_RHO}"
OUTPUT_DIR="controller_optimization/checkpoints/complexity_sweep"

echo "=============================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Node:             $SLURM_NODELIST"
echo "=============================================="
echo "Run name:         $RUN_NAME"
echo "ST params:        n=$ST_N m=$ST_M rho=$ST_RHO n_processes=$ST_NPROC"
echo "Seeds:            target=$SEED_T baseline=$SEED_B"
echo "Data directory:   $DATA_DIR"
echo "UP checkpoint:    $UP_DIR"
echo "Output directory: $OUTPUT_DIR/$RUN_NAME"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# ---- Step 1: Generate dataset (skip if already done) ----
# Multiple jobs with the same ST config may race here; generate_dataset.py
# is idempotent so the first one generates and others find data already present.
echo "Step 1: Generating dataset..."
if [ -d "$DATA_DIR/per_process" ] && [ "$(ls -A $DATA_DIR/per_process/*.pt 2>/dev/null)" ]; then
    echo "  Dataset already exists in $DATA_DIR. Skipping."
else
    echo "Command: python generate_dataset.py --st_n $ST_N --st_m $ST_M --st_rho $ST_RHO --st_n_processes $ST_NPROC --output_dir $DATA_DIR"
    python generate_dataset.py \
        --st_n "$ST_N" \
        --st_m "$ST_M" \
        --st_rho "$ST_RHO" \
        --st_n_processes "$ST_NPROC" \
        --output_dir "$DATA_DIR"
fi
echo ""
echo "Step 1 completed."
echo ""

# ---- Step 2: Train uncertainty predictors (skip if already done) ----
echo "Step 2: Training uncertainty predictors..."
echo "Command: python train_predictor.py --st_n $ST_N --st_m $ST_M --st_rho $ST_RHO --st_n_processes $ST_NPROC --checkpoint_base_dir $UP_DIR --data_dir $DATA_DIR --skip-existing"
echo ""

python train_predictor.py \
    --st_n "$ST_N" \
    --st_m "$ST_M" \
    --st_rho "$ST_RHO" \
    --st_n_processes "$ST_NPROC" \
    --checkpoint_base_dir "$UP_DIR" \
    --data_dir "$DATA_DIR" \
    --skip-existing

echo ""
echo "Step 2 completed."
echo ""

# ---- Step 3: Train controller with seed pair ----
echo "Step 3: Training controller..."
echo "Command: python train_controller.py --output_dir $OUTPUT_DIR --run_name $RUN_NAME --no_pdf --st_n $ST_N --st_m $ST_M --st_rho $ST_RHO --st_n_processes $ST_NPROC --up_checkpoint_dir $UP_DIR --seed_target $SEED_T --seed_baseline $SEED_B"
echo ""

python train_controller.py \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --no_pdf \
    --st_n "$ST_N" \
    --st_m "$ST_M" \
    --st_rho "$ST_RHO" \
    --st_n_processes "$ST_NPROC" \
    --up_checkpoint_dir "$UP_DIR" \
    --seed_target "$SEED_T" \
    --seed_baseline "$SEED_B"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/$RUN_NAME/"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "ERROR: Controller training failed!"
    echo "=============================================="
    exit 1
fi
