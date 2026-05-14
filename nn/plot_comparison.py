#!/usr/bin/env python3
"""
Create comparison plots between Simple Baseline and Heteroscedastic models.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from train_heteroscedastic import HeteroscedasticMultiTask
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def plot_comparisons():
    """Create comprehensive comparison plots."""
    
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
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = random_split(range(len(dataset)), [train_size, val_size])
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=0, collate_fn=dfba_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    # Load models
    print("Loading models...")
    if not Path('simple_baseline_model.pt').exists():
        print("⚠ Simple baseline not found")
        return
    
    simple_model = SimpleBaseline(n_components=4, n_params=0, latent_dim=32)
    simple_model.load_state_dict(torch.load('simple_baseline_model.pt', map_location=device))
    simple_model.to(device)
    simple_model.eval()
    
    if not Path('heteroscedastic_model.pt').exists():
        print("⚠ Heteroscedastic model not found")
        return
    
    hetero_model = HeteroscedasticMultiTask(n_components=4, n_params=0)
    hetero_model.load_state_dict(torch.load('heteroscedastic_model.pt', map_location=device))
    hetero_model.to(device)
    hetero_model.eval()
    
    # Evaluate
    print("Evaluating models...")
    all_simple_pred = []
    all_hetero_pred = []
    all_hetero_var = []
    all_target = []
    all_target_phase = []
    
    with torch.no_grad():
        for batch in val_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            target = batch['trajectory'].to(device)
            phase_target = batch['phases'].to(device) if 'phases' in batch else None
            params = batch['parameters'].to(device)
            
            simple_out = simple_model(ic, time, params)
            simple_pred = simple_out['concentrations'].cpu().numpy()
            
            hetero_out = hetero_model(ic, time, params)
            hetero_pred = hetero_out['concentrations'].cpu().numpy()
            hetero_var = np.exp(hetero_out['log_variances'].cpu().numpy())
            
            all_simple_pred.append(simple_pred)
            all_hetero_pred.append(hetero_pred)
            all_hetero_var.append(hetero_var)
            all_target.append(target.cpu().numpy())
            if phase_target is not None:
                all_target_phase.append(phase_target.cpu().numpy())
    
    simple_all = np.concatenate(all_simple_pred, axis=0)
    hetero_all = np.concatenate(all_hetero_pred, axis=0)
    hetero_var_all = np.concatenate(all_hetero_var, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    target_phase_all = np.concatenate(all_target_phase, axis=0) if all_target_phase else None
    
    # ========================================================================
    # Plot 1: Per-Component MSE Comparison
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison: MSE per Component', fontsize=14, fontweight='bold')
    
    for comp_idx, comp_name in enumerate(component_names):
        ax = axes[comp_idx // 2, comp_idx % 2]
        
        simple_pred = simple_all[:, :, comp_idx].flatten()
        hetero_pred = hetero_all[:, :, comp_idx].flatten()
        target = target_all[:, :, comp_idx].flatten()
        
        simple_mse = np.mean((simple_pred - target) ** 2)
        hetero_mse = np.mean((hetero_pred - target) ** 2)
        improvement = (simple_mse - hetero_mse) / simple_mse * 100
        
        x = [0, 1]
        mses = [simple_mse, hetero_mse]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(x, mses, color=colors, width=0.6, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for i, (bar, mse) in enumerate(zip(bars, mses)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Simple\nBaseline', 'Heteroscedastic'], fontsize=11)
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp_name}\nImprovement: {improvement:+.1f}%', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(mses) * 1.3)
    
    plt.tight_layout()
    plt.savefig('comparison_mse_per_component.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_mse_per_component.png")
    plt.close()
    
    # ========================================================================
    # Plot 2: Predictions vs Actual (Titer - the problem component)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Titer: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    
    titer_idx = 3
    simple_titer = simple_all[:, :, titer_idx].flatten()
    hetero_titer = hetero_all[:, :, titer_idx].flatten()
    target_titer = target_all[:, :, titer_idx].flatten()
    
    # Plot 1: Simple Baseline
    ax = axes[0]
    ax.scatter(target_titer, simple_titer, alpha=0.6, s=50, color='#FF6B6B', edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Ground Truth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_title('Simple Baseline', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    mse_simple = np.mean((simple_titer - target_titer) ** 2)
    ax.text(0.05, 0.95, f'MSE: {mse_simple:.4f}', transform=ax.transAxes,
           fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Heteroscedastic
    ax = axes[1]
    ax.scatter(target_titer, hetero_titer, alpha=0.6, s=50, color='#4ECDC4', edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Ground Truth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_title('Heteroscedastic', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    mse_hetero = np.mean((hetero_titer - target_titer) ** 2)
    ax.text(0.05, 0.95, f'MSE: {mse_hetero:.4f}', transform=ax.transAxes,
           fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comparison_titer_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_titer_predictions.png")
    plt.close()
    
    # ========================================================================
    # Plot 3: Uncertainty Calibration (Titer)
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    hetero_titer_var = hetero_var_all[:, :, titer_idx].flatten()
    hetero_titer_error = np.abs(hetero_titer - target_titer)
    
    # Scatter plot with color gradient
    scatter = ax.scatter(hetero_titer_var, hetero_titer_error, c=target_titer, 
                        cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Predicted Uncertainty (Variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Titer: Uncertainty Calibration\n(Model learns to be uncertain when wrong)', 
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ground Truth Titer', fontsize=11, fontweight='bold')
    
    # Add correlation
    corr = np.corrcoef(hetero_titer_error, hetero_titer_var)[0, 1]
    ax.text(0.05, 0.95, f'Error-Variance Correlation: {corr:.3f}\n(Perfect calibration ≈ 0.95)',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('comparison_uncertainty_calibration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_uncertainty_calibration.png")
    plt.close()
    
    # ========================================================================
    # Plot 4: Early Phase Performance
    # ========================================================================
    if target_phase_all is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Early Phase Performance (phase < 0.1)', fontsize=14, fontweight='bold')
        
        early_mask = target_phase_all.flatten() < 0.1
        
        if early_mask.sum() > 0:
            # Per-component MSE in early phase
            ax = axes[0]
            early_mses_simple = []
            early_mses_hetero = []
            
            for comp_idx, comp_name in enumerate(component_names):
                simple_early = simple_all[:, :, comp_idx].flatten()[early_mask]
                hetero_early = hetero_all[:, :, comp_idx].flatten()[early_mask]
                target_early = target_all[:, :, comp_idx].flatten()[early_mask]
                
                mse_simple = np.mean((simple_early - target_early) ** 2)
                mse_hetero = np.mean((hetero_early - target_early) ** 2)
                
                early_mses_simple.append(mse_simple)
                early_mses_hetero.append(mse_hetero)
            
            x = np.arange(len(component_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, early_mses_simple, width, label='Simple', 
                          color='#FF6B6B', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, early_mses_hetero, width, label='Heteroscedastic',
                          color='#4ECDC4', edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
            ax.set_title('MSE per Component', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(component_names, fontsize=10)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Error values on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Overall comparison
            ax = axes[1]
            overall_simple = np.mean(early_mses_simple)
            overall_hetero = np.mean(early_mses_hetero)
            improvement = (overall_simple - overall_hetero) / overall_simple * 100
            
            bars = ax.bar([0, 1], [overall_simple, overall_hetero], 
                         color=['#FF6B6B', '#4ECDC4'], width=0.6, edgecolor='black', linewidth=2)
            
            for bar, val in zip(bars, [overall_simple, overall_hetero]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Simple\nBaseline', 'Heteroscedastic'], fontsize=11)
            ax.set_ylabel('Average MSE', fontsize=11, fontweight='bold')
            ax.set_title(f'Overall Early Phase\nImprovement: {improvement:+.1f}%', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_early_phase.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: comparison_early_phase.png")
        plt.close()
    
    print("\n✓ All plots saved!")
    print("\nGenerated files:")
    print("  - comparison_mse_per_component.png")
    print("  - comparison_titer_predictions.png")
    print("  - comparison_uncertainty_calibration.png")
    print("  - comparison_early_phase.png")


if __name__ == "__main__":
    plot_comparisons()
