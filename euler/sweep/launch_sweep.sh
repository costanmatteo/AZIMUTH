#!/bin/bash
# ============================================================================
# Launch Controller Sweep — Full Pipeline
# ============================================================================
#
# This script runs the complete pipeline before submitting the sweep:
#   1. Generate dataset (generate_dataset.py)
#   2. Train uncertainty predictors (train_predictor.py)
#   3. Train surrogate — only if controller_config uses 'casualit'
#   4. Generate sweep params (generate_sweep_params.py)
#   5. Submit sweep array job (sbatch sweep.sh)
#
# Usage (from login node):
#   bash euler/sweep/launch_sweep.sh
#
# ============================================================================

set -e

cd $HOME/AZIMUTH

# ── Load environment ────────────────────────────────────────────────────────
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Fix Windows line endings
find euler/sweep/ configs/ -name '*.py' -exec sed -i 's/\r$//' {} +
find . -maxdepth 1 -name '*.py' -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' euler/sweep/sweep.sh

mkdir -p logs

echo "=============================================="
echo "  Controller Sweep — Full Pipeline"
echo "=============================================="
echo ""

# ── Step 1: Generate dataset ────────────────────────────────────────────────
echo "[1/5] Generating dataset..."
srun --cpus-per-task=2 --mem-per-cpu=4G --time=00:30:00 \
    python generate_dataset.py
echo "  Done."
echo ""

# ── Step 2: Train uncertainty predictors ────────────────────────────────────
echo "[2/5] Training uncertainty predictors..."
srun --cpus-per-task=2 --mem-per-cpu=4G --time=00:30:00 \
    python train_predictor.py
echo "  Done."
echo ""

# ── Step 3: Train surrogate (only if using CasualiT) ───────────────────────
SURROGATE_TYPE=$(python -c "
from configs.controller_config import CONTROLLER_CONFIG
print(CONTROLLER_CONFIG.get('surrogate', {}).get('type', 'reliability_function'))
" 2>/dev/null || echo "reliability_function")

if [ "$SURROGATE_TYPE" = "casualit" ]; then
    echo "[3/5] Training CasualiT surrogate..."
    srun --cpus-per-task=2 --mem-per-cpu=4G --time=01:00:00 \
        python train_surrogate.py
    echo "  Done."
else
    echo "[3/5] Skipped — using reliability_function (no surrogate needed)"
fi
echo ""

# ── Step 4: Generate sweep params ──────────────────────────────────────────
echo "[4/5] Generating sweep parameters..."
python euler/sweep/generate_sweep_params.py
echo "  Done."
echo ""

# ── Step 5: Submit sweep ────────────────────────────────────────────────────
echo "[5/5] Submitting sweep job..."
sbatch euler/sweep/sweep.sh
echo ""
echo "=============================================="
echo "  Pipeline complete — sweep submitted!"
echo "=============================================="
