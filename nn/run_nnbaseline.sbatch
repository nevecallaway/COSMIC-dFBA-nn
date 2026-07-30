#!/bin/bash
# =============================================================================
# SLURM job: pure-NN accuracy baseline (nn_baseline.py) with the AUC column.
#
# Trains the physics-free NextDayPredictor on synthetic data, window-level
# holdout, and scores all 8 variables in absolute units (rho, normMAE, peak,
# AUC, range). AUC is the forecast-window integral ratio (pred/true) -- the
# integrated-output metric (total titer / biomass).
#
# Runs on standby QOS, which is PREEMPTIBLE, so --requeue lets SLURM restart the
# job from scratch if a node is reclaimed. Training is ~1 min once scheduled, so
# a restart is cheap; nn_baseline does not checkpoint mid-run.
#
# Submit from the nn/ directory: sbatch run_nnbaseline.sbatch
# =============================================================================
#SBATCH --job-name=nn_baseline
#SBATCH --account=jth54
#SBATCH --partition=m9g
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/nnbaseline_%j.log
#SBATCH --error=logs/nnbaseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY="/home/nevecc/.conda/envs/cosmic/bin/python"
DATA="${NN_DIR}/synthetic_ode.npz"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  requeue#=${SLURM_RESTART_COUNT:-0}"
mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

if [[ ! -f "$DATA" ]]; then
  log "synthetic_ode.npz not found -> generating (closed-form, 3000 extra reactors)"
  $PY generate_synthetic_ode.py --n-extra 3000 --fast --output "$DATA"
fi

log "=== nn_baseline (all 8 features, AUC column) ==="
$PY -u nn_baseline.py --data "$DATA" --all-features

log "Done. Quote the AUC column: ~1.00 means the integrated output (total titer / "
log "biomass) is right, not just the peak."
