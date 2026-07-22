#!/bin/bash
# One-time extraction + build of the fed-batch daily npz on SLURM (CPU only).
# This job has NO GPU work (decompression + CSV downsampling), so it uses the
# CPU partition, not gpu. The 13 GB zip must already be downloaded:
#
#   On a login/transfer node (compute nodes have no internet):
#       cd <repo>/nn/data
#       wget -O cho_fedbatch.zip "https://ndownloader.figshare.com/files/56779742"
#
#   Then, from the repo's nn/ directory:
#       mkdir -p logs && sbatch prep_fedbatch.sbatch
#
#SBATCH --job-name=fedbatch_prep
#SBATCH --partition=m9               # CPU partition, lots of idle nodes. NOT a GPU partition.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fedbatch_prep_%j.log
#SBATCH --error=logs/fedbatch_prep_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

# --- environment (mirrors your interactive `conda activate cosmic`) ---
# conda activate needs initializing in a batch shell; this does that first.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosmic
python -c "import py7zr" 2>/dev/null || pip install --quiet py7zr

# --- paths: zip lives in nn/data (gitignored); npz written alongside it ---
export FEDBATCH_ZIP=$SLURM_SUBMIT_DIR/data/cho_fedbatch.zip
export FEDBATCH_OUT=$SLURM_SUBMIT_DIR/data/cho_fedbatch_daily.npz
export FEDBATCH_N=2000
export FEDBATCH_SEED=0

python prep_fedbatch.py

echo "Done. Reusable dataset at: $FEDBATCH_OUT"
