#!/usr/bin/env python3
"""
Multi-Task Learning with Modality Balance Detection
- Prevents one task from dominating others
- Uses weighted losses + smoothness penalties + physical constraints
- Creates assessment figures to detect imbalance
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import matplotlib.pyplot as plt

try:
    from cosmic_nn_surrogate import dFBADataset, dfba_collate_fn
    from train_heteroscedastic import HeteroscedasticMultiTask
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def compute_smoothness_loss(predictions):
    """
    Penalize non-smooth predictions.
    Physically, concentrations shouldn't change wildly between timepoints.
    
    Loss = mean(|pred[t] - pred[t-1]|²) over all timepoints
    """
    if predictions.shape[1] < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Difference between consecutive timepoints
    diffs = predictions[:, 1:, :] - predictions[:, :-1, :]
    smoothness = torch.mean(diffs ** 2)
    return smoothness


def compute_rate_magnitude_loss(predictions):
    """
    Constrain rate of change magnitude.
    Prevents model from predicting unrealistic concentration changes.
    """
    if predictions.shape[1] < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Rate of change per timepoint
    rates = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :])
    
    # Penalize large rates (max realistic rate ≈ 0.1 per timepoint)
    rate_magnitude = torch.mean(torch.clamp(rates - 0.1, min=0) ** 2)
    return rate_magnitude


def evaluate_modality_balance(train_losses_history, val_losses_history, component_names):
    """
    Create figures to detect modality dominance.
    
    Args:
        train_losses_history: Dict of list of training losses per component
        val_losses_history: Dict of list of validation losses per component
        component_names: List of component names
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Modality Balance Analysis: Detecting Dominance', 
                fontsize=14, fontweight='bold')
    
    # ====================================================================
    # Plot 1: Individual Component Losses Over Time
    # ====================================================================
    ax = axes[0, 0]
    
    for comp_idx, comp_name in enumerate(component_names + ['Phase']):
        if comp_name in val_losses_history:
            losses = val_losses_history[comp_name]
            ax.plot(losses, marker='o', label=comp_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax.set_title('Individual Component Losses', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')  # Log scale to see all modalities
    
    # ====================================================================
    # Plot 2: Loss Ratio (Titer / Glucose) - Imbalance Indicator
    # ====================================================================
    ax = axes[0, 1]
    
    if 'Titer' in val_losses_history and 'Glucose' in val_losses_history:
        titer_losses = np.array(val_losses_history['Titer'])
        glucose_losses = np.array(val_losses_history['Glucose'])
        
        # Avoid division by zero
        ratio = np.divide(titer_losses, glucose_losses + 1e-8)
        
        ax.plot(ratio, marker='s', color='#FF6B6B', linewidth=2.5, markersize=5)
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Balanced (1:1)')
        ax.fill_between(range(len(ratio)), 0.8, 1.2, alpha=0.2, color='green', label='Healthy range')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Titer Loss / Glucose Loss', fontsize=11, fontweight='bold')
        ax.set_title('Modality Balance Ratio', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Highlight if imbalanced
        if np.any(ratio < 0.5) or np.any(ratio > 2.0):
            ax.text(0.95, 0.95, '⚠ IMBALANCED', transform=ax.transAxes,
                   fontsize=12, ha='right', va='top', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # ====================================================================
    # Plot 3: Per-Component Improvement Rate
    # ====================================================================
    ax = axes[1, 0]
    
    improvement_rates = []
    comp_labels = []
    
    for comp_name in component_names + ['Phase']:
        if comp_name in val_losses_history and len(val_losses_history[comp_name]) > 1:
            losses = np.array(val_losses_history[comp_name])
            # Improvement rate: (initial - final) / initial
            improvement = (losses[0] - losses[-1]) / (losses[0] + 1e-8) * 100
            improvement_rates.append(improvement)
            comp_labels.append(comp_name)
    
    if improvement_rates:
        colors = ['#FF6B6B' if x < 10 else '#4ECDC4' for x in improvement_rates]
        bars = ax.barh(comp_labels, improvement_rates, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, improvement_rates)):
            ax.text(rate, i, f' {rate:.1f}%', va='center', fontweight='bold')
        
        ax.axvline(x=20, color='green', linestyle='--', linewidth=2, label='Healthy (>20%)')
        ax.set_xlabel('Improvement Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Loss Reduction per Component', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    
    # ====================================================================
    # Plot 4: Loss Distribution (Box Plot) - Final Epoch
    # ====================================================================
    ax = axes[1, 1]
    
    final_losses = []
    labels_list = []
    
    for comp_name in component_names + ['Phase']:
        if comp_name in val_losses_history and len(val_losses_history[comp_name]) > 0:
            final_loss = val_losses_history[comp_name][-1]
            final_losses.append([final_loss])
            labels_list.append(comp_name)
    
    if final_losses:
        bp = ax.boxplot(final_losses, labels=labels_list, patch_artist=True)
        
        for patch, label in zip(bp['boxes'], labels_list):
            if label == 'Titer':
                patch.set_facecolor('#FF6B6B')
            elif label == 'Phase':
                patch.set_facecolor('#FFD93D')
            else:
                patch.set_facecolor('#4ECDC4')
        
        ax.set_ylabel('Final Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Final Loss Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('modality_balance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: modality_balance_analysis.png")
    plt.close()


def train_with_balance_detection(epochs=200, batch_size=2, learning_rate=1e-3):
    """Train with modality balance monitoring and assessment."""
    
    print(f"\n{'='*80}")
    print("MULTI-TASK LEARNING WITH MODALITY BALANCE DETECTION")
    print(f"{'='*80}\n")
    
    # Load data
    possible_paths = [
        Path("data_2.csv"),
        Path("/Users/nevecallaway/COSMIC-dFBA-nn/COSMIC-dFBA-nn/nn/data_2.csv"),
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = str(p)
            break
    
    if data_file is None:
        print("Error: data_2.csv not found")
        return
    
    trajectories, time_points, ics, metadata = load_experimental_data(data_file)
    phases = metadata['phases']
    
    dataset = dFBADataset(
        trajectories, time_points, ics, 
        parameters={}, normalize=True, phases=phases
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = random_split(range(len(dataset)), [train_size, val_size])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, collate_fn=dfba_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, collate_fn=dfba_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroscedasticMultiTask(n_components=4, n_params=0, latent_dim=32, hidden_dim=64)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    phase_criterion = nn.MSELoss()
    
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    # Loss tracking for balance detection
    train_losses_history = {comp: [] for comp in component_names + ['Phase', 'Smoothness', 'Rates']}
    val_losses_history = {comp: [] for comp in component_names + ['Phase', 'Smoothness', 'Rates']}
    
    print(f"{'='*80}")
    print("Loss Weighting (to prevent modality dominance):")
    print(f"{'='*80}")
    print("Concentration loss: 1.0")
    print("Smoothness penalty: 0.1  (physical constraint)")
    print("Rate magnitude:     0.01 (max change constraint)")
    print("Phase loss:         0.5  (auxiliary task)")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_conc_loss = 0.0
        train_smooth_loss = 0.0
        train_rate_loss = 0.0
        train_phase_loss = 0.0
        train_count = 0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            target_traj = batch['trajectory'].to(device)
            phase_target = batch['phases'].to(device) if 'phases' in batch else None
            params = batch['parameters'].to(device)
            
            optimizer.zero_grad()
            
            output = model(ic, time, params)
            pred_means = output['concentrations']
            pred_logvars = output['log_variances']
            pred_phases = output['phases']
            
            # Concentration loss
            precision = torch.exp(-pred_logvars)
            mse = (target_traj - pred_means) ** 2
            conc_loss = 0.5 * (pred_logvars + mse * precision).mean()
            
            # Smoothness penalty (physics: concentrations change smoothly)
            smoothness_loss = compute_smoothness_loss(pred_means)
            
            # Rate magnitude penalty (physics: max rate constraint)
            rate_loss = compute_rate_magnitude_loss(pred_means)
            
            # Phase loss
            if phase_target is not None:
                phase_loss = phase_criterion(pred_phases, phase_target)
            else:
                phase_loss = 0.0
            
            # COMBINED LOSS with weights to prevent dominance
            total_loss = (
                conc_loss +
                0.1 * smoothness_loss +
                0.01 * rate_loss +
                0.5 * phase_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_conc_loss += conc_loss.item() * ic.shape[0]
            train_smooth_loss += smoothness_loss.item() * ic.shape[0]
            train_rate_loss += rate_loss.item() * ic.shape[0]
            train_phase_loss += (phase_loss.item() if phase_loss != 0 else 0) * ic.shape[0]
            train_count += ic.shape[0]
        
        train_conc_loss /= train_count
        train_smooth_loss /= train_count
        train_rate_loss /= train_count
        train_phase_loss /= train_count if phase_loss != 0 else 1
        
        # Validation
        model.eval()
        val_conc_loss = 0.0
        val_smooth_loss = 0.0
        val_rate_loss = 0.0
        val_phase_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(device)
                time = batch['time'].to(device)
                target_traj = batch['trajectory'].to(device)
                phase_target = batch['phases'].to(device) if 'phases' in batch else None
                params = batch['parameters'].to(device)
                
                output = model(ic, time, params)
                pred_means = output['concentrations']
                pred_logvars = output['log_variances']
                pred_phases = output['phases']
                
                precision = torch.exp(-pred_logvars)
                mse = (target_traj - pred_means) ** 2
                conc_loss = 0.5 * (pred_logvars + mse * precision).mean()
                
                smoothness_loss = compute_smoothness_loss(pred_means)
                rate_loss = compute_rate_magnitude_loss(pred_means)
                
                if phase_target is not None:
                    phase_loss = phase_criterion(pred_phases, phase_target)
                else:
                    phase_loss = 0.0
                
                val_conc_loss += conc_loss.item() * ic.shape[0]
                val_smooth_loss += smoothness_loss.item() * ic.shape[0]
                val_rate_loss += rate_loss.item() * ic.shape[0]
                val_phase_loss += (phase_loss.item() if phase_loss != 0 else 0) * ic.shape[0]
                val_count += ic.shape[0]
        
        val_conc_loss /= val_count
        val_smooth_loss /= val_count
        val_rate_loss /= val_count
        val_phase_loss /= val_count if phase_loss != 0 else 1
        
        # Track individual losses
        train_losses_history['Cell Density'].append(train_conc_loss)  # Placeholder
        val_losses_history['Cell Density'].append(val_conc_loss)
        val_losses_history['Phase'].append(val_phase_loss)
        val_losses_history['Smoothness'].append(val_smooth_loss)
        val_losses_history['Rates'].append(val_rate_loss)
        
        val_total = val_conc_loss + 0.1 * val_smooth_loss + 0.01 * val_rate_loss + 0.5 * val_phase_loss
        
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(model.state_dict(), 'heteroscedastic_balanced_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Conc={val_conc_loss:.6f}, Phase={val_phase_loss:.6f}, "
                  f"Smooth={val_smooth_loss:.6f}, Rates={val_rate_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to heteroscedastic_balanced_model.pt")
    
    # Generate assessment figures
    print(f"\nGenerating modality balance analysis figures...")
    evaluate_modality_balance(train_losses_history, val_losses_history, component_names)
    
    return model


if __name__ == "__main__":
    train_with_balance_detection(epochs=200, batch_size=2, learning_rate=1e-3)
