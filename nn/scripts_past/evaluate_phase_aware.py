#!/usr/bin/env python3
"""
Evaluate phase-aware model predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from phase_aware_model import CosmicNNSurrogatePhaseAware
    from cosmic_nn_surrogate import dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def evaluate_phase_aware_model():
    """Evaluate phase-aware model."""
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA: Evaluate Phase-Aware Model")
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
    phases_true = metadata['phases']
    
    # Create dataset
    dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True)
    
    # Use all data for evaluation
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CosmicNNSurrogatePhaseAware(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    # Try to load trained model
    if Path('phase_aware_model.pt').exists():
        model.load_state_dict(torch.load('phase_aware_model.pt', map_location=device))
        print("✓ Loaded phase_aware_model.pt")
    else:
        print("✗ phase_aware_model.pt not found")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} reactors...")
    all_predictions = []
    all_targets = []
    all_phases_pred = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            
            predictions = model(ic, time, params)
            
            conc_pred = predictions['concentrations'].cpu().numpy()
            phase_weights = predictions['phase_weights'].cpu().numpy()
            
            all_predictions.append(conc_pred[0])
            all_targets.append(target.cpu().numpy()[0])
            all_phases_pred.append(phase_weights[0, :, 0])
    
    # Visualize first 3 reactors
    print(f"\nGenerating visualizations...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('Phase-Aware Model: Predictions vs Real Data', fontsize=14, fontweight='bold')
    
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    for reactor_idx in range(min(3, len(all_predictions))):
        pred = all_predictions[reactor_idx]
        real = all_targets[reactor_idx]
        phases_pred = all_phases_pred[reactor_idx]
        phases_real = phases_true[reactor_idx]
        
        # Phase prediction
        ax = axes[reactor_idx, 0]
        time_axis = np.arange(len(phases_pred))
        ax.plot(time_axis, phases_real, 'o-', linewidth=2.5, markersize=6, label='Ground Truth', color='red')
        ax.plot(time_axis, phases_pred, 's--', linewidth=2, markersize=5, label='NN Prediction', color='blue')
        ax.axhline(0.2, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.fill_between(time_axis, 0, 0.2, alpha=0.2, color='green')
        ax.fill_between(time_axis, 0.8, 1.0, alpha=0.2, color='orange')
        ax.set_ylabel('Phase Weight')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.set_title(f'Reactor {reactor_idx}: Phase')
        ax.grid(True, alpha=0.3)
        
        # Metabolite predictions
        for comp_idx in range(3):
            ax = axes[reactor_idx, comp_idx + 1]
            time_axis = np.arange(len(pred))
            
            ax.plot(time_axis, real[:, comp_idx], 'o-', linewidth=2.5, markersize=6, 
                   label='Real', color='red', alpha=0.8)
            ax.plot(time_axis, pred[:, comp_idx], 's--', linewidth=2, markersize=5, 
                   label='NN Pred', color='blue', alpha=0.8)
            
            ax.set_ylabel(component_names[comp_idx])
            ax.legend(fontsize=8, loc='best')
            ax.set_title(f'{component_names[comp_idx]} (Reactor {reactor_idx})')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_aware_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: phase_aware_predictions.png")
    plt.close()
    
    # Analysis
    print(f"\n{'='*70}")
    print("Prediction Analysis")
    print(f"{'='*70}")
    
    all_phases_pred_flat = np.concatenate(all_phases_pred)
    print(f"\nPhase predictions:")
    print(f"  Mean: {all_phases_pred_flat.mean():.3f}")
    print(f"  Std: {all_phases_pred_flat.std():.3f}")
    print(f"  Min-Max: {all_phases_pred_flat.min():.3f} - {all_phases_pred_flat.max():.3f}")
    
    # Phase distribution
    growth_pred = (all_phases_pred_flat < 0.4).sum()
    prod_pred = (all_phases_pred_flat > 0.6).sum()
    trans_pred = ((all_phases_pred_flat >= 0.4) & (all_phases_pred_flat <= 0.6)).sum()
    total = len(all_phases_pred_flat)
    
    print(f"\nPhase distribution (predicted):")
    print(f"  Growth (<0.4): {growth_pred} ({100*growth_pred/total:.1f}%)")
    print(f"  Transition: {trans_pred} ({100*trans_pred/total:.1f}%)")
    print(f"  Production (>0.6): {prod_pred} ({100*prod_pred/total:.1f}%)")
    
    # Concentration errors
    print(f"\n{'='*70}")
    print("Concentration Prediction Errors")
    print(f"{'='*70}")
    
    for comp_idx, comp_name in enumerate(component_names):
        all_pred_comp = np.concatenate([p[:, comp_idx] for p in all_predictions])
        all_real_comp = np.concatenate([r[:, comp_idx] for r in all_targets])
        mse = np.mean((all_pred_comp - all_real_comp) ** 2)
        mae = np.mean(np.abs(all_pred_comp - all_real_comp))
        print(f"{comp_name}: MSE={mse:.5f}, MAE={mae:.5f}")


if __name__ == "__main__":
    evaluate_phase_aware_model()
