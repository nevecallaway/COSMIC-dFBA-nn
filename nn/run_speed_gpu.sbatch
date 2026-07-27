#!/bin/bash
# =============================================================================
# SLURM job: speed benchmark on GPU.
#
# Times N reactors' worth of 13-day trajectories five ways (numerical ODE,
# closed-form ODE, pure NN, hybrid closed-form, hybrid Euler-50) with matched
# network size, so differences isolate the ODE cost. The NN runs on GPU; the
# mechanistic solver (scipy solve_ivp) is CPU regardless, which is the honest
# baseline either way.
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

# Bigger N on GPU so the tiny per-reactor NN times average over enough work and
# are not dominated by kernel-launch overhead.
for N in 5000 20000; do
  log "=== benchmark, n=$N (GPU for NN, CPU for solve_ivp) ==="
  $PY speed_benchmark.py --n "$N" --device cuda
  echo
done

log "Done. Quote the largest-N table (most stable per-reactor times)."
