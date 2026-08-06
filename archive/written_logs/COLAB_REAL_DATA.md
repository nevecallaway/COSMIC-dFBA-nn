#!/usr/bin/env python3
"""
Colab notebook to train COSMIC-dFBA on real experimental data.
Run this in Colab with GPU support.
"""

COLAB_NOTEBOOK = """
# COSMIC-dFBA: Training on Real Experimental Data
# ================================================

# Step 0: Setup and clone repo
GITHUB_URL = "https://github.com/nevecallaway/COSMIC-dFBA-nn.git"

!cd /content && git clone $GITHUB_URL 2>/dev/null || (cd /content/COSMIC-dFBA-nn && git pull)
%cd /content/COSMIC-dFBA-nn/nn

# Step 1: Upload real data or download from paper
print("="*70)
print("COSMIC-dFBA Real Data Training")
print("="*70)

# Option A: Upload your CSV files
from google.colab import files
print("\\nUpload data_2.csv (experimental measurements)")
uploaded = files.upload()

# Option B: Or copy paste this if you have it:
# data_2.csv should contain columns:
# Vessel, Time, Production phase fraction, Cell Density, Glucose, Lactate, Titer, ...

# Step 2: Load and analyze real data
!python load_real_data.py

# Step 3: Train on real data
!python train_real_data.py

# Step 4: Download visualizations
from google.colab import files
files.download('real_data_predictions.png')
files.download('phase_transitions_real.png')

print("\\n✓ Real data training complete!")
print("Download visualizations and check phase transition learning")

# Step 5: Compare with synthetic data
# The visualizations show:
# - Left plots: How well NN predicts real metabolite trajectories
# - Top right: Phase transition learning (real vs predicted)
# - Green shaded regions: Growth phase (p_m < 0.2)
# - Red shaded regions: Production phase (p_m > 0.8)

# Key metrics:
# - Phase classification accuracy: How well model learns bistability
# - Trajectory MSE: Prediction error on metabolites
# - Does model learn sharp transitions or smooth gradients?
"""

print(COLAB_NOTEBOOK)
