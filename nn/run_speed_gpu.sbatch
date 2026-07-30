#!/bin/bash
# =============================================================================
# SLURM job: speed benchmark on GPU.
#
# Times N reactors' worth of 13-day trajectories seven ways (numerical ODE,
# closed-form ODE numpy/torch, torchdiffeq, pure NN, hybrid closed-form, hybrid
# Euler-50) with matched network size, so differences isolate the ODE cost. Each
# method is warmed up then looped for --secs of wall time and amortized, so the
# per-reactor numbers are real inference, not kernel-launch overhead.
#
# Two same-device baselines: scipy (CPU incumbent) and torchdiffeq (a general
# solver on the NN's GPU -> isolates algorithm from hardware).
#
# PREREQUISITE (run ONCE on a login node, compute nodes have no internet):
#   /home/nevecc/.conda/envs/cosmic/bin/pip install torchdiffeq
# Without it the torchdiffeq row is skipped (the run still works).
#
# Submit from the nn/ directory: sbatch run_speed_gpu.sbatch
# =============================================================================
#SBATCH --job-name=speed_bench
#SBATCH --account=jth54
#SBATCH --partition=m9g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/speed_%j.log
#SBATCH --error=logs/speed_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

NN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY="/home/nevecc/.conda/envs/cosmic/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job ${SLURM_JOB_ID} on $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
mkdir -p "$NN_DIR/logs"
cd "$NN_DIR"

# Confirm torchdiffeq is importable so we know whether that row will appear.
$PY -c "import torchdiffeq; print('torchdiffeq', torchdiffeq.__version__)" \
  || log "WARNING: torchdiffeq not installed -> that baseline row will be skipped"

# Bigger N on GPU so per-pass launch overhead is spread over enough reactors, and
# a long --secs so most of the run is spent doing inference (not warmup/overhead).
for N in 5000 20000; do
  log "=== benchmark, n=$N, 45s/method (GPU for NN + torch/torchdiffeq ODE) ==="
  $PY speed_benchmark.py --n "$N" --secs 45 --device cuda
  echo
done

log "Done. Quote the largest-N table; read the 'vs torchdiffeq' column as the "
log "honest same-device surrogate-vs-solver number."
