#!/bin/bash
# =============================================================================
# SLURM job: one-time prep of the Golzarijalal 400k in-silico CHO fed-batch
# dataset (Figshare DOI 10.26188/28943096, CC BY 4.0) into a compact daily npz.
#
# CPU only: this is decompression + CSV downsampling, no GPU work, so it runs on
# m9 rather than m9g. Idempotent, re-running is a no-op once the npz exists.
#
# The 13 GB zip must already be downloaded into nn/data from a login/transfer
# node (compute nodes have no internet):
#     cd <repo>/nn/data
#     wget -O cho_fedbatch.zip "https://ndownloader.figshare.com/files/56779742"
#
# Submit from the nn/ directory: sbatch prep_fedbatch.sbatch
# =============================================================================
#SBATCH --job-name=fedbatch_prep
#SBATCH --account=jth54
#SBATCH --partition=m9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fedbatch_prep_%j.log
#SBATCH --error=logs/fedbatch_prep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Edit to match your cluster ───────────────────────────────────────────────
NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"             # submit from the nn/ directory
PYTHON="/home/nevecc/.conda/envs/cosmic/bin/python"

N_RUNS=2000                                     # runs to sample (raise for more data)
SEED=0
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)"

mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

# py7zr is only needed if the system 7z is unavailable; prep_fedbatch.py prefers 7z.
$PYTHON -c "import py7zr" 2>/dev/null || $PYTHON -m pip install --quiet py7zr

export FEDBATCH_ZIP="$NN_DIR/data/cho_fedbatch.zip"
export FEDBATCH_OUT="$NN_DIR/data/cho_fedbatch_daily.npz"
export FEDBATCH_N=$N_RUNS
export FEDBATCH_SEED=$SEED

log "Sampling $N_RUNS runs from $FEDBATCH_ZIP ..."
$PYTHON prep_fedbatch.py

log "Done. Reusable dataset: $FEDBATCH_OUT"
log "Cleanup to reclaim ~26 GB once verified:"
log "  rm -rf $NN_DIR/data/fedbatch_inner.7z $NN_DIR/data/fedbatch_runs"
