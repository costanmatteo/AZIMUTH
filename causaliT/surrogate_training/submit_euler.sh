#!/bin/bash
#SBATCH --job-name=casualit_train
#SBATCH --output=logs/casualit_%j.out
#SBATCH --error=logs/casualit_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your Euler setup)
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0

# Activate your virtual environment (adjust path as needed)
# Option 1: venv
# source /cluster/home/$USER/venvs/azimuth/bin/activate

# Option 2: conda
# source /cluster/apps/local/env2lmod.sh
# module load anaconda3
# conda activate azimuth

# Go to project directory
cd $SLURM_SUBMIT_DIR/../../..

# Print info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# Option 1: Generate data only
# python -m causaliT.surrogate_training.train_surrogate --data_only --device cuda

# Option 2: Full training (generate data + train)
python -m causaliT.surrogate_training.train_surrogate --generate_data --device cuda

# Option 3: Training with custom parameters
# python -m causaliT.surrogate_training.train_surrogate --generate_data --epochs 1000 --batch_size 128 --device cuda

echo "End time: $(date)"
