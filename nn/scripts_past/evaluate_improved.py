#!/usr/bin/env python3
"""
Evaluate improved model on real data and compare predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import (
        CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
    )
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def evaluate_improved_model():
    """Evaluate improved model and visualize predictions."""
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA: Evaluate Improved Model")
    print(f"{'='*70}")
    
    # Load real data
    print(f"\nLoading real experimental data...")
    possible_paths = [
        Path("data_2.csv"),
        Path("/content/COSMIC-dFBA-nn/nn/data_2.csv"),
        Path("/Users/nevecallaway/Downloads/data_2.csv"),
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = str(p)
            break
    
    if data_file is None:
        print(f"Error: data_2.csv not found")
        sys.exit(1)
    
    trajectories, time_points, ics, metadata = load_experimental_data(data_file)
    phases = metadata['phases']
    
    # Create dataset
    dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True, phases=phases)
    
    # Use all data for evaluation
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    # Try to load improved model
    if Path('improved_model.pt').exists():
        model.load_state_dict(torch.load('improved_model.pt', map_location=device))
        print("✓ Loaded improved_model.pt")
    else:
        print("✗ improved_model.pt not found")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} reactors...")
    all_predictions = []
    all_targets = []
    all_phases_true = []
    all_phases_pred = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            phases = batch.get('phases', None)
            if phases is not None:
                phases = phases.to(device)
            
            predictions = model(ic, time, params)
            
            conc_pred = predictions['concentrations'].cpu().numpy()
            phase_logits = predictions['phase_weights'].cpu().numpy()
            
            # Convert logits to probabilities
            phase_probs = 1.0 / (1.0 + np.exp(-phase_logits))  # Sigmoid for inference
            
            all_predictions.append(conc_pred[0])  # Shape: (n_time, n_components)
            all_targets.append(target.cpu().numpy()[0])
            all_phases_pred.append(phase_probs[0, :, 0])  # Production class probability
            if phases is not None:
                all_phases_true.append(phases.cpu().numpy()[0])
    
    # Visualize first 3 reactors
    print(f"\nGenerating visualizations...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('Improved Model: Predictions vs Real Data', fontsize=14, fontweight='bold')
    
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    for reactor_idx in range(min(3, len(all_predictions))):
        pred = all_predictions[reactor_idx]
        real = all_targets[reactor_idx]
        phases_true = all_phases_true[reactor_idx] if all_phases_true else None
        phases_pred = all_phases_pred[reactor_idx]
        
        # Phase prediction
        ax = axes[reactor_idx, 0]
        time_axis = np.arange(len(phases_pred))
        ax.plot(time_axis, all_phases_true[reactor_idx], 'o-', linewidth=2.5, markersize=6, label='Ground Truth', color='red')
        ax.plot(time_axis, phases_pred, 's--', linewidth=2, markersize=5, label='NN Prediction', color='blue')
        ax.axhline(0.2, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Growth threshold')
        ax.axhline(0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Production threshold')
        ax.fill_between(time_axis, 0, 0.2, alpha=0.2, color='green', label='Growth phase')
        ax.fill_between(time_axis, 0.8, 1.0, alpha=0.2, color='orange', label='Production phase')
        ax.set_ylabel('Phase Weight (p_m)')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.set_title(f'Reactor {reactor_idx}: Phase Prediction')
        ax.grid(True, alpha=0.3)
        
        # Metabolite predictions
        for comp_idx in range(3):
            ax = axes[reactor_idx, comp_idx + 1]
            time_axis = np.arange(len(pred))
            
            # Plot real data
            ax.plot(time_axis, real[:, comp_idx], 'o-', linewidth=2.5, markersize=6, 
                   label='Real', color='red', alpha=0.8)
            
            # Plot prediction
            ax.plot(time_axis, pred[:, comp_idx], 's--', linewidth=2, markersize=5, 
                   label='NN Pred', color='blue', alpha=0.8)
            
            # Shade phases
            if phases_true is not None:
                for t in range(len(phases_true)):
                    if t > 0:
                        if phases_true[t-1] < 0.2:
                            ax.axvspan(t-0.5, t+0.5, alpha=0.1, color='green')
                        elif phases_true[t-1] > 0.8:
                            ax.axvspan(t-0.5, t+0.5, alpha=0.1, color='orange')
            
            ax.set_ylabel(component_names[comp_idx])
            ax.legend(fontsize=8, loc='best')
            ax.set_title(f'{component_names[comp_idx]} (Reactor {reactor_idx})')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: improved_predictions.png")
    plt.close()
    
    # Phase accuracy analysis
    print(f"\n{'='*70}")
    print("Phase Prediction Analysis")
    print(f"{'='*70}")
    
    all_phases_true_flat = np.concatenate(all_phases_true)
    all_phases_pred_flat = np.concatenate(all_phases_pred)
    
    # Classify predictions
    pred_growth = all_phases_pred_flat < 0.4
    pred_prod = all_phases_pred_flat > 0.6
    true_growth = all_phases_true_flat < 0.2
    true_prod = all_phases_true_flat > 0.8
    
    growth_recall = (pred_growth & true_growth).sum() / (true_growth.sum() + 1e-6)
    prod_recall = (pred_prod & true_prod).sum() / (true_prod.sum() + 1e-6)
    
    print(f"Growth phase (true <0.2) recall: {growth_recall:.3f}")
    print(f"Production phase (true >0.8) recall: {prod_recall:.3f}")
    print(f"Mean predicted phase: {all_phases_pred_flat.mean():.3f}")
    print(f"Std predicted phase: {all_phases_pred_flat.std():.3f}")
    print(f"Min-max predicted phase: {all_phases_pred_flat.min():.3f} - {all_phases_pred_flat.max():.3f}")
    
    # Concentration MSE
    print(f"\n{'='*70}")
    print("Concentration Prediction Error")
    print(f"{'='*70}")
    
    for comp_idx, comp_name in enumerate(component_names):
        all_pred_comp = np.concatenate([p[:, comp_idx] for p in all_predictions])
        all_real_comp = np.concatenate([r[:, comp_idx] for r in all_targets])
        mse = np.mean((all_pred_comp - all_real_comp) ** 2)
        mae = np.mean(np.abs(all_pred_comp - all_real_comp))
        print(f"{comp_name}: MSE={mse:.5f}, MAE={mae:.5f}")


if __name__ == "__main__":
    evaluate_improved_model()
