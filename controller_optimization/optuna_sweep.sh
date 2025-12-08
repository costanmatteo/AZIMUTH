#!/bin/bash
#SBATCH --job-name=optuna_hpo
#SBATCH --account=es_mohr
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err
#SBATCH --array=0-49

# ============================================================================
# Optuna Distributed Hyperparameter Optimization - Euler HPC (ETH Zurich)
# ============================================================================
#
# This script runs Optuna hyperparameter optimization in distributed mode.
# Each SLURM array job runs ONE trial, and all jobs share the same SQLite
# database for coordination.
#
# Hyperparameters optimized:
#   - hidden_sizes: Architecture of the policy generator
#   - dropout: Dropout rate for regularization
#   - use_batchnorm: Whether to use batch normalization
#   - scenario_embedding_dim: Dimension of scenario embedding
#   - learning_rate: Learning rate for optimizer
#
# Resources: CPU only, 4 cores, 16GB RAM total, 30 min per job
#
# Usage:
#   1. Create study (once):
#      python optuna_tuning.py --create-study --study-name "controller_hpo"
#
#   2. Submit jobs:
#      sbatch optuna_sweep.sh
#
#   3. Monitor progress:
#      python optuna_tuning.py --study-name "controller_hpo" --status
#
#   4. Generate report after completion:
#      python optuna_tuning.py --study-name "controller_hpo" --report
#
# Monitor SLURM:
#   squeue -u $USER                    # Check job status
#   tail -f logs/optuna_<jobid>_<taskid>.out  # Follow output
#
# Results:
#   - Database: controller_optimization/optuna_results/controller_hpo/study.db
#   - Report: controller_optimization/optuna_results/controller_hpo/
# ============================================================================

# Exit on error
set -e

# Configuration
STUDY_NAME="controller_hpo"
N_TRIALS=50  # Total trials (should match --array size)
REDUCED_EPOCHS=500  # Fewer epochs for faster trials (set to empty for full training)

# Load required modules (Euler 2024 stack)
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

# Navigate to project root
cd $HOME/AZIMUTH

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=============================================="
echo "OPTUNA HYPERPARAMETER OPTIMIZATION"
echo "=============================================="
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "Array Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Node:             $SLURM_NODELIST"
echo "CPUs:             $SLURM_CPUS_PER_TASK"
echo "Memory:           $SLURM_MEM_PER_CPU MB per CPU"
echo "=============================================="
echo "Study name:       $STUDY_NAME"
echo "Reduced epochs:   ${REDUCED_EPOCHS:-'(full training)'}"
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

# Ensure Optuna is installed
python -c "import optuna" 2>/dev/null || {
    echo "Installing Optuna..."
    pip install optuna plotly kaleido --quiet
}

# Create study if it doesn't exist (only first job does this)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Task 0: Checking if study exists..."
    python controller_optimization/optuna_tuning.py \
        --study-name "$STUDY_NAME" \
        --status 2>/dev/null || {
        echo "Creating new study..."
        python controller_optimization/optuna_tuning.py \
            --create-study \
            --study-name "$STUDY_NAME"
    }
    # Small delay to ensure study is created before other jobs try to access it
    sleep 5
fi

# Wait a bit to avoid all jobs hitting the DB at once
# Stagger start times based on array task ID
sleep $((SLURM_ARRAY_TASK_ID % 10))

# Build command
CMD="python controller_optimization/optuna_tuning.py"
CMD="$CMD --study-name $STUDY_NAME"
CMD="$CMD --single-trial"
CMD="$CMD --device cpu"
CMD="$CMD --verbose"

if [ -n "$REDUCED_EPOCHS" ]; then
    CMD="$CMD --reduced-epochs $REDUCED_EPOCHS"
fi

# Run single trial
echo "Starting trial..."
echo "Command: $CMD"
echo ""

$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Trial completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "ERROR: Trial failed!"
    echo "=============================================="
    exit 1
fi

# If this is the last job, generate the report
# Note: This is a simple check; in practice, you may want to run report separately
if [ "$SLURM_ARRAY_TASK_ID" -eq $((N_TRIALS - 1)) ]; then
    echo ""
    echo "Last job - waiting for other jobs to complete before generating report..."
    echo "Run manually: python optuna_tuning.py --study-name $STUDY_NAME --report"
fi
