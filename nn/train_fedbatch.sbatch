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

log "Data-scaling sweep: n_train = [$SWEEP], epochs=$EPOCHS"
$PYTHON train_fedbatch.py --sweep $SWEEP --epochs $EPOCHS

log "Done. Read the table: rising rho and falling MAE with n_train => data limit."
