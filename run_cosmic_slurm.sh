#!/bin/bash
#SBATCH --job-name=cosmic_nn
#SBATCH --output=cosmic_nn_%j.log
#SBATCH --error=cosmic_nn_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1          # Remove if no GPU available
#SBATCH --mem=16G
#SBATCH --partition=gpu            # Adjust to your cluster's partition

# Load Python module (varies by cluster)
module load python/3.10            # Check your cluster's module names
# OR use conda
# module load conda

# Create virtual environment (first time only)
# python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r nn/requirements.txt

# Run the demo
cd COSMIC-dFBA-nn
python nn/test_demo.py

# Optional: Save outputs
echo "Job completed at $(date)" >> results.log