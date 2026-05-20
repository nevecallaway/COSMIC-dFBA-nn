#!/bin/bash
# =============================================================================
# SLURM job script — COSMIC-dFBA training via Apptainer container
# Usage: sbatch run_cosmic_apptainer.sh
# =============================================================================

#SBATCH --job-name=cosmic_dfba
#SBATCH --partition=gpu                  # check: sinfo -s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cosmic_%j.log
#SBATCH --error=logs/cosmic_%j.err

set -euo pipefail

# ── Edit these paths ──────────────────────────────────────────────────────────
WORK_DIR="$HOME/cosmic-dfba"       # directory containing your 4 .py files + data/
SIF="$HOME/cosmic-dfba/cosmic.sif" # path to the built .sif image
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} starting on $(hostname)"
log "GPUs: ${CUDA_VISIBLE_DEVICES:-none visible}"

mkdir -p "$WORK_DIR/logs"

# Verify the container and GPU are accessible
apptainer exec --nv "$SIF" python -c \
    "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Step 1: Generate synthetic training data
log "Generating synthetic data..."
apptainer exec --nv \
    --bind "${WORK_DIR}:/workspace" \
    "$SIF" \
    bash -c "cd /workspace && python generate_synthetic_training.py"
log "Synthetic data done."

# Step 2: Train the model
log "Starting training..."
apptainer exec --nv \
    --bind "${WORK_DIR}:/workspace" \
    "$SIF" \
    bash -c "cd /workspace && python train.py"
log "Training complete."

# Copy outputs to timestamped results directory
RESULTS_DIR="$WORK_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cp "$WORK_DIR/improved_model.pt" "$RESULTS_DIR/" 2>/dev/null || true
cp "logs/cosmic_${SLURM_JOB_ID}.log" "$RESULTS_DIR/" 2>/dev/null || true

log "Outputs saved to $RESULTS_DIR"
log "Job complete."

# =============================================================================
# FIRST-TIME SETUP
# =============================================================================
# 1. Copy the minimal files to the cluster:
#      scp train.py model.py utils.py generate_synthetic_training.py \
#          <user>@<cluster>:~/cosmic-dfba/
#      scp data/data_1.csv data/data_2.csv data/data_3.csv \
#          <user>@<cluster>:~/cosmic-dfba/data/
#      scp cosmic.sif <user>@<cluster>:~/cosmic-dfba/
#
# 2. Build the container locally (needs Docker or fakeroot):
#      apptainer build cosmic.sif cosmic.def
#    Or with fakeroot on supported clusters:
#      apptainer build --fakeroot cosmic.sif cosmic.def
#
# 3. Check partition and GPU availability:
#      sinfo -s
#      sinfo -o "%P %G %l"
#
# 4. Submit:
#      sbatch run_cosmic_apptainer.sh
#
# 5. Monitor:
#      squeue -u $USER
#      tail -f logs/cosmic_<job_id>.log
# =============================================================================
