#!/bin/bash
# =============================================================================
# SLURM job: fed-batch data-scaling sweep.
#
# Trains the fed-batch hybrid model (small conv body -> fluxes -> fixed fed-batch
# ODE) on 10, 50, 200, 1000 runs against a FIXED held-out set of 200 runs, and
# forecasts each held-out run from its first 6 days.
#
# The question: does held-out accuracy improve with training-set size? If yes,
# the 10-reactor wall on the perfusion data was a DATA limit, not a method limit.
#
# Submit from the nn/ directory: sbatch train_fedbatch.sbatch
# =============================================================================
#SBATCH --job-name=fedbatch_sweep
#SBATCH --account=jth54
#SBATCH --partition=m9g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fedbatch_sweep_%j.log
#SBATCH --error=logs/fedbatch_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="/home/nevecc/.conda/envs/cosmic/bin/python"

SWEEP="10 50 200 1000"
EPOCHS=60

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"

mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

# Pass 1: fixed epoch budget, but now with a validation split and logged curves.
# The overtraining readout at the end says whether val bottomed out early.
log "PASS 1: fixed $EPOCHS epochs (no early stopping), curves -> curves_fixed/"
$PYTHON train_fedbatch.py --sweep $SWEEP --epochs $EPOCHS \
    --curve-dir curves_fixed

# Pass 2: same, with early stopping on validation loss and best-weight restore.
# If Kimberly is right, this beats pass 1, especially at the small n_train rows.
log "PASS 2: early stopping (patience 20), curves -> curves_earlystop/"
$PYTHON train_fedbatch.py --sweep $SWEEP --epochs $EPOCHS --patience 20 \
    --curve-dir curves_earlystop

log "Done. Compare PASS 1 vs PASS 2 tables; check the overtraining readout."
