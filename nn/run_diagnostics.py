#!/usr/bin/env python3
"""
COSMIC-dFBA Diagnostic Suite
This script audits a trained surrogate model for biological consistency,
modality dominance, and predicts titer drop-off points.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cosmic_nn_surrogate import (
    CosmicNNSurrogateEnhanced,
    dFBADataset,
    PredictionInterface
)
from diagnostics import ModelDiagnostics

def run_comprehensive_audit(model_path, dataset, test_cases):
    """
    Runs a full biological and statistical audit on the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"COSMIC-dFBA MODEL AUDIT")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # 1. Load Model
    checkpoint = torch.load(model_path, map_location=device)

    # Handle full-checkpoint vs state_dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        # Use hyperparameters from checkpoint if available, else use defaults
        hparams = checkpoint.get('hyperparams', {'latent_dim': 64, 'n_heads': 4, 'n_components': dataset.n_components})
    else:
        state_dict = checkpoint
        hparams = {'latent_dim': 64, 'n_heads': 4, 'n_components': dataset.n_components}

    model = CosmicNNSurrogateEnhanced(
        n_components=hparams['n_components'],
        n_params=0,
        latent_dim=hparams['latent_dim'],
        n_heads=hparams['n_heads']
    ).to(device)

    model.load_state_dict(state_dict)

    predictor = PredictionInterface(model, dataset, device=device, model_type='enhanced')

    # 2. Audit each test case
    for i, case in enumerate(test_cases):
        ic = case['ic']
        time = case['time']
        params = case.get('params', None)
        case_name = case.get('name', f"Case {i+1}")

        print(f"\n--- Analyzing {case_name} ---")

        # Get Prediction
        results = predictor.predict(ic, time, params, return_rates=True)
        conc = results['concentrations']

        # A. Modality Dominance Check
        # Note: analyze_modality_dominance expects tensors
        dominance = ModelDiagnostics.analyze_modality_dominance(
            model, ic, time, params if params is not None else np.zeros(0), device=device
        )
        print(f"Saliency Audit: IC Importance={dominance['ic_importance']:.4f}, "
              f"Param Importance={dominance['param_importance']:.4f}")
        print(f"Dominance Ratio (IC/Param): {dominance['dominance_ratio']:.2f}")

        if dominance['dominance_ratio'] > 10:
            print("⚠️ WARNING: Model may be dominated by Initial Conditions (ignoring parameters).")
        elif dominance['dominance_ratio'] < 0.1:
            print("⚠️ WARNING: Model may be dominated by Parameters (ignoring ICs).")
        else:
            print("✅ Modality balance looks healthy.")

        # B. Drop-off RCA
        drop_off = ModelDiagnostics.detect_drop_off_rca(
            conc[np.newaxis, :], time, comp_idx=-1
        )[0]

        if drop_off['is_crashing']:
            print(f"🚨 CRASH DETECTED: Titer drop-off starts at day {drop_off['drop_start_time']:.2f}")
            print(f"📉 Avg Decay Rate: {drop_off['avg_decay_rate']:.4f} units/day")
        else:
            print("✅ Stability: No significant titer drop-off detected.")

    print(f"\n{'='*70}")
    print("Audit Complete.")

def main():
    # --- CONFIGURATION ---
    # Replace these with your actual file paths
    MODEL_PATH = 'improved_model.pt'
    # We need a dataset object to get the normalization constants (traj_min, traj_max)
    # In a real scenario, you'd load your training data here.
    # For this script, we'll create a dummy dataset with the correct dimensions.

    # Adjust these dimensions to match your actual la-data
    N_COMPONENTS = 5
    N_TIMEPOINTS = 100

    print("Initializing Mock Dataset for normalization...")
    # We create a dummy dataset just to satisfy the PredictionInterface
    # In production, you should load the actual dFBADataset used during training.
    dataset = dFBADataset(
        trajectories=np.random.rand(1, N_TIMEPOINTS, N_COMPONENTS),
        time_points=np.linspace(0, 10, N_TIMEPOINTS)[np.newaxis, :],
        initial_conditions=np.random.rand(1, N_COMPONENTS),
        parameters={},
        normalize=True
    )

    # --- DEFINE TEST CASES ---
    # Case 1: Standard run
    # Case 2: Low substrate (expected to crash)
    test_cases = [
        {
            "name": "Standard Condition",
            "ic": np.array([0.5, 1.0, 0.0, 0.0, 0.0]),
            "time": np.linspace(0, 10, 100),
            "params": np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        },
        {
            "name": "Substrate Limited (Crash Test)",
            "ic": np.array([0.5, 0.1, 0.0, 0.0, 0.0]), # Very low glucose
            "time": np.linspace(0, 10, 100),
            "params": np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        }
    ]

    try:
        run_comprehensive_audit(MODEL_PATH, dataset, test_cases)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please update MODEL_PATH in the script.")
    except Exception as e:
        print(f"An error occurred during audit: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
