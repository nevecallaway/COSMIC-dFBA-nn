#!/usr/bin/env python3
"""
Detailed error analysis: Find trends in where model fails.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"  Make sure all dependencies are installed")
    sys.exit(1)


def analyze_errors():
    """Detailed error analysis."""
    
    print(f"\n{'='*70}")
    print("Error Analysis: Where Does the Model Fail?")
    print(f"{'='*70}")
    
    # Load data
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
        print("Error: data_2.csv not found")
        sys.exit(1)
    
    trajectories, time_points, ics, metadata = load_experimental_data(data_file)
    phases = metadata['phases']
    
    dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True, phases=phases)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Load model (Simple Baseline - the winner)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleBaseline(n_components=dataset.n_components, n_params=0, latent_dim=32)
    
    model_paths = [
        'simple_baseline_model.pt',
        'simple-baseline_model.pt',
        'simple_baseline_enhanced_model.pt'
    ]
    
    model_found = False
    for path in model_paths:
        if Path(path).exists():
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"✓ Loaded {path}")
            model_found = True
            break
    
    if not model_found:
        print(f"\n⚠ Pre-trained model not found. Training from scratch...")
        print(f"  (Usually run: python compare_models.py first)")
        print(f"\nTraining Simple Baseline on 70% data...")
        
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dfba_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, 51):
            model.train()
            for batch in train_loader:
                ic = batch['initial_conditions'].to(device)
                time = batch['time'].to(device)
                params = batch['parameters'].to(device)
                target = batch['trajectory'].to(device)
                
                optimizer.zero_grad()
                predictions = model(ic, time, params)
                conc_pred = predictions['concentrations']
                loss = nn.functional.mse_loss(conc_pred, target)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in val_loader:
                    ic = batch['initial_conditions'].to(device)
                    time = batch['time'].to(device)
                    params = batch['parameters'].to(device)
                    target = batch['trajectory'].to(device)
                    predictions = model(ic, time, params)
                    val_loss += nn.functional.mse_loss(predictions['concentrations'], target).item()
                val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Val Loss = {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                break
        
        torch.save(model.state_dict(), 'simple_baseline_model.pt')
        print(f"✓ Model trained and saved")
    
    model.to(device)
    model.eval()
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    all_errors = []
    reactor_ids = list(metadata['reactors'])
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            
            predictions = model(ic, time, params)
            conc_pred = predictions['concentrations'].cpu().numpy()[0]
            target_np = target.cpu().numpy()[0]
            
            error = np.abs(conc_pred - target_np)
            
            all_predictions.append(conc_pred)
            all_targets.append(target_np)
            all_errors.append(error)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)
    
    print(f"\n{'='*70}")
    print("Error Statistics by Reactor")
    print(f"{'='*70}")
    
    reactor_errors = np.mean(all_errors, axis=(1, 2))
    sorted_indices = np.argsort(reactor_errors)[::-1]  # Worst first
    
    print(f"\n{'Reactor':<12} {'MAE':<10} {'Phase Range':<20}")
    for idx in sorted_indices:
        mae = reactor_errors[idx]
        phase_range = f"{phases[idx].min():.2f}-{phases[idx].max():.2f}"
        print(f"{reactor_ids[idx]:<12} {mae:<10.4f} {phase_range:<20}")
    
    print(f"\nWorst reactor: {reactor_ids[sorted_indices[0]]} (MAE={reactor_errors[sorted_indices[0]]:.4f})")
    print(f"Best reactor: {reactor_ids[sorted_indices[-1]]} (MAE={reactor_errors[sorted_indices[-1]]:.4f})")
    
    print(f"\n{'='*70}")
    print("Error Statistics by Component")
    print(f"{'='*70}")
    
    for comp_idx, comp_name in enumerate(component_names):
        comp_errors = all_errors[:, :, comp_idx]
        mean_error = np.mean(comp_errors)
        max_error = np.max(comp_errors)
        std_error = np.std(comp_errors)
        
        print(f"\n{comp_name}:")
        print(f"  Mean MAE: {mean_error:.4f}")
        print(f"  Std:      {std_error:.4f}")
        print(f"  Max:      {max_error:.4f}")
    
    print(f"\n{'='*70}")
    print("Error vs Phase Analysis")
    print(f"{'='*70}")
    
    # Correlate error with phase value
    # Expand phases to match error dimensions (repeat for each component)
    all_phases_expanded = np.repeat(np.concatenate(phases)[:, np.newaxis], 4, axis=1).flatten()
    all_errors_flat = all_errors.reshape(-1)
    
    phase_bins = np.linspace(0, 1, 11)
    for i in range(len(phase_bins) - 1):
        mask = (all_phases_expanded >= phase_bins[i]) & (all_phases_expanded < phase_bins[i+1])
        if mask.sum() > 0:
            bin_error = np.mean(all_errors_flat[mask])
            print(f"Phase [{phase_bins[i]:.1f}-{phase_bins[i+1]:.1f}): Mean Error = {bin_error:.4f} ({mask.sum()} points)")
    
    # Visualization
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Error Analysis: Simple Baseline Model', fontsize=14, fontweight='bold')
    
    # 1. Error by reactor
    ax = axes[0, 0]
    ax.barh(range(len(reactor_ids)), reactor_errors[sorted_indices])
    ax.set_yticks(range(len(reactor_ids)))
    ax.set_yticklabels([reactor_ids[i] for i in sorted_indices], fontsize=8)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Error by Reactor')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Error by component
    ax = axes[0, 1]
    comp_errors = [np.mean(all_errors[:, :, i]) for i in range(len(component_names))]
    ax.bar(range(len(component_names)), comp_errors)
    ax.set_xticks(range(len(component_names)))
    ax.set_xticklabels(component_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error by Component')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Error vs phase
    ax = axes[0, 2]
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    bin_errors = []
    for i in range(len(phase_bins) - 1):
        mask = (all_phases_expanded >= phase_bins[i]) & (all_phases_expanded < phase_bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(all_errors_flat[mask]))
        else:
            bin_errors.append(np.nan)
    
    ax.plot(phase_centers, bin_errors, 'o-', linewidth=2, markersize=8)
    ax.axvline(0.2, color='green', linestyle='--', alpha=0.5, label='Phase thresholds')
    ax.axvline(0.8, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error vs Phase')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Predicted vs Real (all points)
    ax = axes[1, 0]
    pred_flat = all_predictions.flatten()
    real_flat = all_targets.flatten()
    scatter = ax.scatter(real_flat, pred_flat, alpha=0.3, s=10, c=all_errors_flat, cmap='hot')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Real Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predictions vs Reality (colored by error)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Error')
    ax.grid(True, alpha=0.3)
    
    # 5. Error distribution per component
    ax = axes[1, 1]
    bp = ax.boxplot([all_errors[:, :, i].flatten() for i in range(len(component_names))],
                     labels=component_names)
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution per Component')
    ax.set_xticklabels(component_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Error timeline (first reactor)
    ax = axes[1, 2]
    time_axis = np.arange(all_errors[0].shape[0])
    for comp_idx, comp_name in enumerate(component_names):
        ax.plot(time_axis, all_errors[0, :, comp_idx], 'o-', label=comp_name, markersize=5)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Error Timeline: {reactor_ids[0]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: error_analysis.png")
    plt.close()
    
    # Insights
    print(f"\n{'='*70}")
    print("Key Insights")
    print(f"{'='*70}")
    
    worst_comp_idx = np.argmax(comp_errors)
    best_comp_idx = np.argmin(comp_errors)
    print(f"\nHardest component: {component_names[worst_comp_idx]} (MAE={comp_errors[worst_comp_idx]:.4f})")
    print(f"Easiest component: {component_names[best_comp_idx]} (MAE={comp_errors[best_comp_idx]:.4f})")
    
    worst_phase_idx = np.nanargmax(bin_errors)
    best_phase_idx = np.nanargmin(bin_errors)
    print(f"\nWorst phase range: [{phase_bins[worst_phase_idx]:.1f}-{phase_bins[worst_phase_idx+1]:.1f}] (MAE={bin_errors[worst_phase_idx]:.4f})")
    print(f"Best phase range: [{phase_bins[best_phase_idx]:.1f}-{phase_bins[best_phase_idx+1]:.1f}] (MAE={bin_errors[best_phase_idx]:.4f})")
    
    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    analyze_errors()
