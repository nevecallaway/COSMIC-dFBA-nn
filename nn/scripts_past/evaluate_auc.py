#!/usr/bin/env python3
"""
Add AUC metrics to model evaluation.
Treats predictions as binary classification (above/below threshold).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import roc_curve, auc, roc_auc_score

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from train_heteroscedastic import HeteroscedasticMultiTask
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def evaluate_with_auc():
    """Evaluate models with AUC metrics."""
    
    print(f"\n{'='*80}")
    print("AUC ANALYSIS: Binary Classification at Multiple Thresholds")
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
    print("Evaluating models...\n")
    all_simple_pred = []
    all_hetero_pred = []
    all_target = []
    
    with torch.no_grad():
        for batch in val_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            target = batch['trajectory'].to(device)
            params = batch['parameters'].to(device)
            
            simple_out = simple_model(ic, time, params)
            simple_pred = simple_out['concentrations'].cpu().numpy()
            
            hetero_out = hetero_model(ic, time, params)
            hetero_pred = hetero_out['concentrations'].cpu().numpy()
            
            all_simple_pred.append(simple_pred)
            all_hetero_pred.append(hetero_pred)
            all_target.append(target.cpu().numpy())
    
    simple_all = np.concatenate(all_simple_pred, axis=0).reshape(-1, 4)
    hetero_all = np.concatenate(all_hetero_pred, axis=0).reshape(-1, 4)
    target_all = np.concatenate(all_target, axis=0).reshape(-1, 4)
    
    # ========================================================================
    # AUC at Multiple Thresholds
    # ========================================================================
    thresholds = [0.25, 0.5, 0.75]
    
    print(f"{'='*80}")
    print("AUC SCORES (Binary Classification: Above/Below Threshold)")
    print(f"{'='*80}\n")
    
    auc_results = {}
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold:.2f}")
        print("-" * 60)
        
        auc_results[threshold] = {}
        
        for comp_idx, comp_name in enumerate(component_names):
            simple_pred = simple_all[:, comp_idx]
            hetero_pred = hetero_all[:, comp_idx]
            target = target_all[:, comp_idx]
            
            # Binary labels: 1 if >= threshold, 0 otherwise
            y_true = (target >= threshold).astype(int)
            
            # Skip if all same class
            if len(np.unique(y_true)) < 2:
                print(f"  {comp_name:15s}: Skipped (all same class)")
                continue
            
            try:
                auc_simple = roc_auc_score(y_true, simple_pred)
                auc_hetero = roc_auc_score(y_true, hetero_pred)
                improvement = (auc_hetero - auc_simple) / auc_simple * 100 if auc_simple > 0 else 0
                
                auc_results[threshold][comp_name] = {
                    'simple': auc_simple,
                    'hetero': auc_hetero,
                    'improvement': improvement
                }
                
                print(f"  {comp_name:15s}: Simple={auc_simple:.4f}, Hetero={auc_hetero:.4f}, "
                      f"Improvement: {improvement:+.1f}%")
            except Exception as e:
                print(f"  {comp_name:15s}: Error computing AUC ({e})")
    
    # ========================================================================
    # ROC Curve Plots
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ROC Curves: Simple Baseline vs Heteroscedastic (Threshold=0.5)', 
                 fontsize=14, fontweight='bold')
    
    threshold = 0.5
    
    for comp_idx, comp_name in enumerate(component_names):
        ax = axes[comp_idx // 2, comp_idx % 2]
        
        simple_pred = simple_all[:, comp_idx]
        hetero_pred = hetero_all[:, comp_idx]
        target = target_all[:, comp_idx]
        
        y_true = (target >= threshold).astype(int)
        
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, 'Not enough classes', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{comp_name}', fontsize=12, fontweight='bold')
            continue
        
        # ROC curves
        fpr_simple, tpr_simple, _ = roc_curve(y_true, simple_pred)
        fpr_hetero, tpr_hetero, _ = roc_curve(y_true, hetero_pred)
        
        auc_simple = auc(fpr_simple, tpr_simple)
        auc_hetero = auc(fpr_hetero, tpr_hetero)
        
        # Plot
        ax.plot(fpr_simple, tpr_simple, color='#FF6B6B', lw=2.5, 
               label=f'Simple (AUC={auc_simple:.3f})', marker='o', markersize=4, alpha=0.7)
        ax.plot(fpr_hetero, tpr_hetero, color='#4ECDC4', lw=2.5,
               label=f'Heteroscedastic (AUC={auc_hetero:.3f})', marker='s', markersize=4, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC=0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    plt.savefig('comparison_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: comparison_roc_curves.png")
    plt.close()
    
    # ========================================================================
    # AUC Comparison Bar Chart
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AUC Scores Across Multiple Thresholds', fontsize=14, fontweight='bold')
    
    for plot_idx, threshold in enumerate(thresholds):
        ax = axes[plot_idx]
        
        if threshold not in auc_results or not auc_results[threshold]:
            continue
        
        simple_aucs = []
        hetero_aucs = []
        comp_labels = []
        
        for comp_name in component_names:
            if comp_name in auc_results[threshold]:
                simple_aucs.append(auc_results[threshold][comp_name]['simple'])
                hetero_aucs.append(auc_results[threshold][comp_name]['hetero'])
                comp_labels.append(comp_name)
        
        if not simple_aucs:
            continue
        
        x = np.arange(len(comp_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, simple_aucs, width, label='Simple', 
                      color='#FF6B6B', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, hetero_aucs, width, label='Heteroscedastic',
                      color='#4ECDC4', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Threshold ≥ {threshold}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comp_labels, fontsize=10, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comparison_auc_thresholds.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_auc_thresholds.png")
    plt.close()
    
    print(f"\n{'='*80}\n")
    print("✓ AUC analysis complete!")
    print("\nGenerated files:")
    print("  - comparison_roc_curves.png (ROC curves for all components)")
    print("  - comparison_auc_thresholds.png (AUC across thresholds)")


if __name__ == "__main__":
    evaluate_with_auc()
