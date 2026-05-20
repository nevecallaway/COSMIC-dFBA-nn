#!/bin/bash
# =============================================================================
# SLURM job script for COSMIC-dFBA surrogate model training
# Usage: sbatch run_cosmic_slurm.sh
# =============================================================================

# ── Edit these to match your cluster ─────────────────────────────────────────
#SBATCH --job-name=cosmic_dfba
#SBATCH --account=jth54
#SBATCH --partition=m9g                  # GPU partition (m8g also available)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Data loading workers
#SBATCH --gres=gpu:p100:1               # 1x P100 GPU
#SBATCH --mem=24G
#SBATCH --time=08:00:00                 # 8h — conservative for 500 pretrain + 400 finetune epochs
#SBATCH --output=logs/cosmic_%j.log     # stdout  (%j = job ID)
#SBATCH --error=logs/cosmic_%j.err      # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # exit on error, undefined variable, or pipe failure

REPO_DIR="$HOME/cosmic"
CONDA_ENV="cosmic"                      # conda env name  (see SETUP below)

# ── Logging helpers ───────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} starting on $(hostname)"
log "GPUs: ${CUDA_VISIBLE_DEVICES:-none visible}"

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/logs"

# ── Load modules (edit for your cluster) ─────────────────────────────────────
# Common options — uncomment whichever applies:
# module load cuda/12.1
# module load python/3.11
# module load anaconda3

# ── Python from conda env (bypasses activation issues in batch jobs) ──────────
PYTHON="/home/nevecc/.conda/envs/cosmic/bin/python"

cd "$REPO_DIR"

# ── Step 1: Generate synthetic training data ──────────────────────────────────
log "Generating synthetic training data..."
$PYTHON generate_synthetic_training.py
log "Synthetic data generation complete."

# ── Step 2: Train the model ───────────────────────────────────────────────────
log "Starting training..."
$PYTHON train.py
log "Training complete."

# ── Copy outputs to a timestamped results directory ──────────────────────────
RESULTS_DIR="$REPO_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cp nn/improved_model.pt "$RESULTS_DIR/" 2>/dev/null || true
cp logs/cosmic_${SLURM_JOB_ID}.log "$RESULTS_DIR/" 2>/dev/null || true

log "Outputs saved to $RESULTS_DIR"
log "Job complete."

# =============================================================================
# FIRST-TIME SETUP (run once manually before submitting)
# =============================================================================
# 1. Clone the repo:
#      git clone https://github.com/nevecallaway/COSMIC-dFBA-nn.git ~/COSMIC-dFBA-nn
#
# 2. Create the conda environment:
#      conda create -n cosmic python=3.11 -y
#      conda activate cosmic
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#      pip install numpy pandas scikit-learn
#
# 3. Check your partition and GPU type:
#      sinfo -s               # list partitions
#      sinfo -o "%P %G %l"    # partition / GPU type / time limit
#
# 4. Submit:
#      sbatch run_cosmic_slurm.sh
#
# 5. Monitor:
#      squeue -u $USER
#      tail -f logs/cosmic_<job_id>.log
# =============================================================================
