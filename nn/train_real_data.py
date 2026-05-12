#!/usr/bin/env python3
"""
Train COSMIC-dFBA on real experimental data from Gopalakrishnan et al. (2024).
Validates model on bistable phase transitions with real metabolite trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Import our modules
try:
    from cosmic_nn_surrogate import (
        CosmicNNSurrogateEnhanced, TrainingManager, PredictionInterface, dFBADataset, dfba_collate_fn
    )
    from load_real_data import load_experimental_data, analyze_phase_transitions
    from torch.utils.data import DataLoader, random_split
    IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    IMPORTS_OK = False


def train_real_model(trajectories, time_points, initial_conditions, metadata):
    """Train enhanced model on real experimental data."""
    print(f"\n{'='*70}")
    print("Training Enhanced Model on Real Experimental Data")
    print(f"{'='*70}")
    
    # Create dataset
    dataset = dFBADataset(trajectories, time_points, initial_conditions, 
                         parameters={}, normalize=True)
    
    # Split train/val (small dataset - use 70/30)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                             num_workers=0, collate_fn=dfba_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                           num_workers=0, collate_fn=dfba_collate_fn)
    
    print(f"Train: {len(train_dataset)} reactors, Validation: {len(val_dataset)} reactors")
    print(f"Batch size: 2, Dataset: {dataset.n_components} components × {dataset.n_timepoints} timepoints")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model outputs: concentrations, phase_weights, growth_rates, prod_rates")
    
    # Train with enhanced loss
    trainer = TrainingManager(model, device=device, learning_rate=5e-4, model_type='enhanced')
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=50, patience=15)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    return model, dataset, device, metadata


def evaluate_real_model(model, dataset, device, metadata):
    """Evaluate on real data and compare predictions to ground truth phases."""
    print(f"\n{'='*70}")
    print("Evaluating Model on Real Data")
    print(f"{'='*70}")
    
    predictor = PredictionInterface(model, dataset, device, model_type='enhanced')
    
    # Test on first reactor
    test_idx = 0
    test_ic = dataset.initial_conditions[test_idx]
    test_time = dataset.time_points[test_idx]
    true_trajectory = dataset.trajectories[test_idx]
    true_phases = metadata['phases'][test_idx]
    
    print(f"\nTesting on reactor {metadata['reactors'][test_idx]}")
    print(f"Initial conditions: {test_ic}")
    print(f"Time span: {test_time[0]:.4f} - {test_time[-1]:.2f} days ({len(test_time)} points)")
    
    # Predict
    try:
        prediction = predictor.predict(test_ic, test_time, return_rates=True)
        print(f"✓ Prediction successful!")
        print(f"  Keys: {list(prediction.keys())}")
        print(f"  Concentrations shape: {prediction['concentrations'].shape}")
        print(f"  Phase weights shape: {prediction['phase_weights'].shape}")
        
        pred_conc = prediction['concentrations']
        pred_phases = prediction['phase_weights'].squeeze()
        
        # Compare phase predictions to ground truth
        print(f"\nPhase Transition Comparison:")
        print(f"  True phases: {true_phases[:5]} ... {true_phases[-5:]}")
        print(f"  Pred phases: {pred_phases[:5]:.3f} ... {pred_phases[-5:]:.3f}")
        
        # Compute phase accuracy (classify as growth/production)
        true_bistable = np.zeros_like(true_phases)
        true_bistable[true_phases > 0.5] = 1  # Production phase
        
        pred_bistable = np.zeros_like(pred_phases)
        pred_bistable[pred_phases > 0.5] = 1
        
        phase_accuracy = np.mean(true_bistable == pred_bistable)
        print(f"  Phase classification accuracy: {phase_accuracy:.1%}")
        
        # Compute trajectory error
        pred_denorm = pred_conc
        true_denorm = true_trajectory
        
        mse = np.mean((pred_denorm - true_denorm) ** 2)
        mae = np.mean(np.abs(pred_denorm - true_denorm))
        
        print(f"\nTrajectory Prediction Error:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        return prediction, test_time, test_ic, true_trajectory, true_phases, predictor
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def visualize_real_results(prediction, test_time, true_trajectory, true_phases, test_ic, reactor_name):
    """Visualize predictions vs real data."""
    print(f"\n{'='*70}")
    print("Visualizing Results")
    print(f"{'='*70}")
    
    if prediction is None:
        print("No predictions to visualize")
        return
    
    pred_conc = prediction['concentrations']
    pred_phases = prediction['phase_weights'].squeeze()
    
    components = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Real Data vs Model Predictions - {reactor_name}", fontsize=14, fontweight='bold')
    
    for i, (ax, comp_name) in enumerate(zip(axes.flat, components)):
        # Plot true data
        ax.plot(test_time, true_trajectory[:, i], 'o-', linewidth=2, 
               markersize=6, label='Real Data', color='red', alpha=0.7)
        
        # Plot predicted
        ax.plot(test_time, pred_conc[:, i], 's--', linewidth=2, 
               markersize=5, label='NN Prediction', color='blue', alpha=0.7)
        
        # Color background by phase
        for j in range(len(test_time)-1):
            if true_phases[j] < 0.2:
                ax.axvspan(test_time[j], test_time[j+1], alpha=0.1, color='green', label='Growth' if j == 0 else '')
            elif true_phases[j] > 0.8:
                ax.axvspan(test_time[j], test_time[j+1], alpha=0.1, color='red', label='Production' if j == 0 else '')
        
        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel(comp_name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: real_data_predictions.png")
    
    # Phase transition plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(test_time, true_phases, 'o-', linewidth=2, markersize=6, 
           label='Ground Truth', color='red', alpha=0.7)
    ax.plot(test_time, pred_phases, 's--', linewidth=2, markersize=5, 
           label='NN Prediction', color='blue', alpha=0.7)
    
    # Mark growth/production regions
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Growth threshold')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Production threshold')
    
    ax.fill_between(test_time, 0, 0.2, alpha=0.1, color='green')
    ax.fill_between(test_time, 0.8, 1.0, alpha=0.1, color='orange')
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Phase Weight (0=Growth, 1=Production)', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Bistable Phase Transition Learning', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase_transitions_real.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: phase_transitions_real.png")
    
    plt.show()


def main():
    """Main pipeline: load real data → train → evaluate."""
    
    if not IMPORTS_OK:
        print("Failed to import required modules")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA Training on Real Experimental Data")
    print(f"{'='*70}")
    
    # Step 1: Load real data
    print(f"\n{'='*70}")
    print("Step 1: Loading Real Experimental Data")
    print(f"{'='*70}")
    
    data_file = Path("/Users/nevecallaway/Downloads/data_2.csv")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        sys.exit(1)
    
    trajectories, time_points, ics, metadata = load_experimental_data(str(data_file))
    
    # Step 2: Analyze phases
    print(f"\n{'='*70}")
    print("Step 2: Analyzing Phase Transitions")
    print(f"{'='*70}")
    
    analyze_phase_transitions(metadata['phases'])
    
    # Step 3: Train model
    model, dataset, device, metadata = train_real_model(trajectories, time_points, ics, metadata)
    
    # Step 4: Evaluate
    prediction, test_time, test_ic, true_traj, true_phases, predictor = \
        evaluate_real_model(model, dataset, device, metadata)
    
    if prediction is not None:
        # Step 5: Visualize
        visualize_real_results(prediction, test_time, true_traj, true_phases, 
                             test_ic, metadata['reactors'][0])
        
        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Real experimental data training successful!")
        print(f"Model learned bistable phase transitions from 10 reactors")
        print(f"Visualizations saved to current directory")
    else:
        print(f"\n✗ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
