#!/bin/bash
# =============================================================================
# SLURM job: warm-up sweep for the real-reactor LORO at the best window (seq 2).
#
# Sweeps how many real-anchored predictions the held-out reactor gets before it
# free-runs on its own predictions: warmup 0 = pure forecast, single-step = the
# ceiling (never free-runs). Answers "how many days of real data does a new
# reactor need before its forecast is trustworthy".
#
# Submit from the nn/ directory: sbatch run_warmup.sbatch
# =============================================================================
#SBATCH --job-name=warmup_sweep
#SBATCH --account=jth54
#SBATCH --partition=m9g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/warmup_%j.log
#SBATCH --error=logs/warmup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY="/home/nevecc/.conda/envs/cosmic/bin/python"

SEQ=2
BATCH=8
EPOCHS=60
SEED=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

run() {  # $1 = label, rest = args
  local label="$1"; shift
  echo -n "$label: "
  $PY loro_real.py --seq-len $SEQ --batch $BATCH --epochs $EPOCHS --seed $SEED "$@" \
      2>/dev/null | grep MEAN
}

log "Warm-up sweep at seq_len=$SEQ, batch=$BATCH, epochs=$EPOCHS, seed=$SEED"
echo "============================================================"
run "forecast   (warmup 0)" --warmup 0
run "warmup 1           "   --warmup 1
run "warmup 2           "   --warmup 2
run "warmup 3           "   --warmup 3
run "warmup 5           "   --warmup 5
run "single-step (ceil) "   --single-step
echo "============================================================"
log "Done. Rows should climb from forecast (worst) toward single-step (ceiling)."
