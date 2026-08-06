#!/bin/bash
# =============================================================================
# SLURM job: leave-one-reactor-out (LORO) generalization test for en Primeur.
# Runs the broad-sampling experiment (rate-mix 1.0, rate-scale 0.5) so the model
# cannot memorize donor rates. Submit: sbatch run_loro_slurm.sh
# =============================================================================
#SBATCH --job-name=primeur_loro
#SBATCH --account=jth54
#SBATCH --partition=m9g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/loro_%j.log
#SBATCH --error=logs/loro_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Edit to match your cluster ───────────────────────────────────────────────
REPO_DIR="$HOME/cosmic"                        # dir containing the nn scripts + data/
PYTHON="/home/nevecc/.conda/envs/cosmic/bin/python"

FOLDS="0 1 2 3 4"                              # reactor indices to hold out
N_EXTRA=3000                                    # lower (e.g. 1500) if generation is slow
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

# Step 0: ensure per-AA scales exist on the cluster (else generation falls back
# to the uniform 210x and the data is wrong).
log "Computing per-AA scales..."
$PYTHON compute_aa_scales.py

# Baseline: tight sampling (80% donor copies). Confirms the leakage/memorization
# problem is representative across folds, not just reactor 0 (which gave 83.8%).
log "LORO BASELINE (tight: rate-mix 0.2, rate-scale 0.1)  folds=[$FOLDS]"
$PYTHON loro_eval.py --folds $FOLDS --rate-mix 0.2 --rate-scale 0.1 --n-extra $N_EXTRA

# Experiment: broad sampling + productivity extension. Covers reactors more/less
# productive than any real one, so the LORO productivity extremes (R0004 low,
# R0005 high) become interpolation instead of extrapolation.
log "LORO EXPERIMENT (broad + extend-prod 0.5)  folds=[$FOLDS]"
$PYTHON loro_eval.py --folds $FOLDS --rate-mix 1.0 --rate-scale 0.5 \
    --extend-prod 0.5 --n-extra $N_EXTRA

log "Done. Two summaries (BASELINE then EXPERIMENT) at the end of the log."
