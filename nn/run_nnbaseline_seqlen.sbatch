#!/bin/bash
# =============================================================================
# SLURM job (GPU): pure-NN (no ODE) baseline swept over input window length 1-5.
#
# Does a shorter window help the physics-free NN like it did the hybrid on real
# data? Trains NextDayPredictor at seq_len 1..5 on synthetic, window-level
# holdout, and prints the all-feature accuracy table for each so you can compare.
#
# Needs synthetic_ode.npz (generated once). Submit from nn/:
#   sbatch --qos=standby --partition=dw run_nnbaseline_seqlen.sbatch
# (dw has idle A100s; standby QOS reaches them. Falls back below if that changes.)
# =============================================================================
#SBATCH --job-name=nnbase_seqlen
#SBATCH --account=jth54
#SBATCH --qos=standby
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/nnbase_seqlen_%j.log
#SBATCH --error=logs/nnbase_seqlen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY="/home/nevecc/.conda/envs/cosmic/bin/python"
DATA="$NN_DIR/synthetic_ode.npz"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

# make the synthetic data once if it is not already there
if [ ! -f "$DATA" ]; then
  log "generating synthetic_ode.npz (n-extra 3000)..."
  $PY generate_synthetic_ode.py --n-extra 3000 --fast --output "$DATA"
fi

for SL in 1 2 3 4 5; do
  log "=== pure NN baseline, seq_len=$SL ==="
  $PY nn_baseline.py --data "$DATA" --seq-len "$SL" --all-features
  echo
done

log "Done. Compare the all-feature tables across seq_len 1-5 (esp. CellDensity rho)."
