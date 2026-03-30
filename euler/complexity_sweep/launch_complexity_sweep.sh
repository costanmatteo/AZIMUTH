#!/bin/bash
# ============================================================================
# Launch Complexity Sensitivity Sweep — Full Pipeline
# ============================================================================
#
# This script cleans up previous sweep results, generates parameters,
# and submits the complexity sweep array job:
#   1. Remove previous complexity sweep files
#   2. Generate sweep params (generate_complexity_sweep_params.py)
#   3. Submit sweep array job (sbatch complexity_sweep.sh)
#
# Usage (from login node):
#   bash euler/complexity_sweep/launch_complexity_sweep.sh
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
find euler/complexity_sweep/ configs/ -name '*.py' -exec sed -i 's/\r$//' {} +
find . -maxdepth 1 -name '*.py' -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' euler/complexity_sweep/complexity_sweep.sh

mkdir -p logs

echo "=============================================="
echo "  Complexity Sensitivity Sweep — Full Pipeline"
echo "=============================================="
echo ""

# ── Cleanup previous sweep ─────────────────────────────────────────────────
echo "[cleanup] Removing previous complexity sweep results..."
rm -rf controller_optimization/checkpoints/complexity_sweep
rm -f logs/complexity_*.out logs/complexity_*.err
rm -f euler/complexity_sweep/complexity_sweep_params.txt
echo "  Done."
echo ""

# ── Step 1: Generate sweep params ──────────────────────────────────────────
echo "[1/2] Generating complexity sweep parameters..."
python euler/complexity_sweep/generate_complexity_sweep_params.py
echo "  Done."
echo ""

# ── Step 2: Submit sweep ────────────────────────────────────────────────────
echo "[2/2] Submitting complexity sweep job..."
sbatch euler/complexity_sweep/complexity_sweep.sh
echo ""
echo "=============================================="
echo "  Pipeline complete — complexity sweep submitted!"
echo "=============================================="
