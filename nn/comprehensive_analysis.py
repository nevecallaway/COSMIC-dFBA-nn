#!/usr/bin/env python3
"""
Comprehensive Data Analysis and Performance Metrics.
Includes: data distribution, train/val/test splits, multiple metrics, prediction balance.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Correlation coefficient
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Correlation': corr,
    }


def main():
    """Comprehensive data and model analysis."""
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE DATA & PERFORMANCE ANALYSIS")
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
    reactor_ids = list(metadata['reactors'])
    
    dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True, phases=phases)
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    print(f"\n{'='*70}")
    print("1. DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"\nTotal reactors: {len(dataset)}")
    print(f"Timepoints per reactor: {trajectories.shape[1]}")
    print(f"Total data points: {len(dataset) * trajectories.shape[1]} ({len(dataset) * trajectories.shape[1] * 4} values)")
    print(f"Components: {', '.join(component_names)}")
    
    # Train/val split (70/30)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = random_split(range(len(dataset)), [train_size, val_size])
    
    print(f"\nTrain/Val Split (70/30):")
    print(f"  Training reactors:   {train_size} ({', '.join([reactor_ids[i] for i in sorted(train_indices.indices)])})")
    print(f"  Validation reactors: {val_size} ({', '.join([reactor_ids[i] for i in sorted(val_indices.indices)])})")
    
    print(f"\n{'='*70}")
    print("2. DATA DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    all_data = trajectories.reshape(-1, 4)
    
    for comp_idx, comp_name in enumerate(component_names):
        comp_data = all_data[:, comp_idx]
        train_comp = trajectories[sorted(train_indices.indices), :, comp_idx].flatten()
        val_comp = trajectories[sorted(val_indices.indices), :, comp_idx].flatten()
        
        print(f"{comp_name}:")
        print(f"  Overall:  mean={comp_data.mean():.4f}, std={comp_data.std():.4f}, "
              f"min={comp_data.min():.4f}, max={comp_data.max():.4f}")
        print(f"  Train:    mean={train_comp.mean():.4f}, std={train_comp.std():.4f}")
        print(f"  Val:      mean={val_comp.mean():.4f}, std={val_comp.std():.4f}")
        print()
    
    # Load model and get predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleBaseline(n_components=dataset.n_components, n_params=0, latent_dim=32)
    
    if not Path('simple_baseline_model.pt').exists():
        print("⚠ Model not found, skipping prediction analysis")
        return
    
    model.load_state_dict(torch.load('simple_baseline_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    all_pred = []
    all_real = []
    
    with torch.no_grad():
        for batch in eval_loader:
            ic = batch['initial_conditions'].to(device)
            time_points = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            
            predictions = model(ic, time_points, params)
            all_pred.append(predictions['concentrations'].cpu().numpy())
            all_real.append(target.cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_real = np.concatenate(all_real, axis=0)
    
    print(f"{'='*70}")
    print("3. COMPREHENSIVE PERFORMANCE METRICS")
    print(f"{'='*70}\n")
    
    # Overall metrics
    overall_metrics = calculate_metrics(all_real.flatten(), all_pred.flatten())
    print("Overall Performance:")
    for metric, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.6f}")
    
    # Per-component metrics
    print(f"\nPer-Component Metrics:\n")
    component_metrics = []
    for comp_idx, comp_name in enumerate(component_names):
        metrics = calculate_metrics(all_real[:, :, comp_idx].flatten(), 
                                   all_pred[:, :, comp_idx].flatten())
        metrics['Component'] = comp_name
        component_metrics.append(metrics)
        print(f"{comp_name}:")
        for metric, value in metrics.items():
            if metric != 'Component' and isinstance(value, float):
                print(f"  {metric}: {value:.6f}")
        print()
    
    # Per-reactor metrics
    print(f"\nPer-Reactor Metrics:\n")
    reactor_metrics = []
    for reactor_idx, reactor_id in enumerate(reactor_ids):
        metrics = calculate_metrics(all_real[reactor_idx].flatten(), 
                                   all_pred[reactor_idx].flatten())
        metrics['Reactor'] = reactor_id
        reactor_metrics.append(metrics)
    
    reactor_df = pd.DataFrame(reactor_metrics)
    print(reactor_df[['Reactor', 'MSE', 'MAE', 'R²']].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("4. PREDICTION BALANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    for comp_idx, comp_name in enumerate(component_names):
        real_comp = all_real[:, :, comp_idx].flatten()
        pred_comp = all_pred[:, :, comp_idx].flatten()
        
        print(f"{comp_name}:")
        print(f"  Real data:       mean={real_comp.mean():.4f}, std={real_comp.std():.4f}")
        print(f"  Predictions:     mean={pred_comp.mean():.4f}, std={pred_comp.std():.4f}")
        print(f"  Mean shift:      {(pred_comp.mean() - real_comp.mean()):.4f}")
        print(f"  Std ratio:       {(pred_comp.std() / real_comp.std()):.2f}x")
        print(f"  Range (real):    [{real_comp.min():.4f}, {real_comp.max():.4f}]")
        print(f"  Range (pred):    [{pred_comp.min():.4f}, {pred_comp.max():.4f}]")
        print()
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Data distribution by component
    ax = plt.subplot(3, 3, 1)
    for comp_idx, comp_name in enumerate(component_names):
        ax.hist(all_real[:, :, comp_idx].flatten(), alpha=0.5, bins=30, label=comp_name)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Data Distribution by Component (Full Dataset)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Train vs Val distribution (Cell Density example)
    ax = plt.subplot(3, 3, 2)
    train_cd = trajectories[sorted(train_indices.indices), :, 0].flatten()
    val_cd = trajectories[sorted(val_indices.indices), :, 0].flatten()
    ax.hist(train_cd, alpha=0.6, bins=20, label=f'Train (n={len(train_cd)})', color='blue')
    ax.hist(val_cd, alpha=0.6, bins=20, label=f'Val (n={len(val_cd)})', color='orange')
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Train/Val Split: Cell Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction balance - Real vs Predicted distributions
    ax = plt.subplot(3, 3, 3)
    ax.scatter(all_real.flatten(), all_pred.flatten(), alpha=0.3, s=10, c=all_real.flatten(), cmap='viridis')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Real Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Prediction Balance: All Components')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4-6. Per-component distributions
    for i, (comp_idx, comp_name) in enumerate([(0, 'Cell Density'), (1, 'Glucose'), (2, 'Lactate')]):
        ax = plt.subplot(3, 3, 4 + i)
        real_comp = all_real[:, :, comp_idx].flatten()
        pred_comp = all_pred[:, :, comp_idx].flatten()
        ax.hist(real_comp, alpha=0.6, bins=25, label='Real', color='blue')
        ax.hist(pred_comp, alpha=0.6, bins=25, label='Predicted', color='orange')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{comp_name}: Distribution Balance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Error distribution
    ax = plt.subplot(3, 3, 7)
    errors = np.abs(all_pred - all_real).flatten()
    ax.hist(errors, bins=30, color='red', alpha=0.7)
    ax.axvline(np.mean(errors), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Per-reactor performance
    ax = plt.subplot(3, 3, 8)
    reactor_mse = [calculate_metrics(all_real[i].flatten(), all_pred[i].flatten())['MSE'] 
                   for i in range(len(reactor_ids))]
    colors = ['green' if i in sorted(train_indices.indices) else 'orange' for i in range(len(reactor_ids))]
    ax.bar(range(len(reactor_ids)), reactor_mse, color=colors, alpha=0.7)
    ax.set_xticks(range(len(reactor_ids)))
    ax.set_xticklabels(reactor_ids, rotation=45, ha='right')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Reactor Performance (Green=Train, Orange=Val)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 9. Component performance
    ax = plt.subplot(3, 3, 9)
    comp_mse = [calculate_metrics(all_real[:, :, i].flatten(), all_pred[:, :, i].flatten())['MSE'] 
                for i in range(len(component_names))]
    comp_mae = [calculate_metrics(all_real[:, :, i].flatten(), all_pred[:, :, i].flatten())['MAE'] 
                for i in range(len(component_names))]
    x = np.arange(len(component_names))
    width = 0.35
    ax.bar(x - width/2, comp_mse, width, label='MSE', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, comp_mae, width, label='MAE', alpha=0.8, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(component_names, rotation=45, ha='right')
    ax.set_ylabel('MSE')
    ax2.set_ylabel('MAE')
    ax.set_title('Per-Component Performance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: comprehensive_analysis.png")
    plt.close()
    
    # Save detailed results
    df_components = pd.DataFrame(component_metrics)
    df_components.to_csv('component_metrics.csv', index=False)
    print(f"✓ Saved: component_metrics.csv")
    
    reactor_df.to_csv('reactor_metrics.csv', index=False)
    print(f"✓ Saved: reactor_metrics.csv")
    
    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
