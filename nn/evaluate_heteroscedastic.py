#!/usr/bin/env python3
"""
Evaluate Heteroscedastic Model vs Simple Baseline.
Compare on overall metrics + per-component metrics + uncertainty quality.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from train_heteroscedastic import HeteroscedasticMultiTask, gaussian_nll_loss
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


# Component weights (same as training)
COMPONENT_WEIGHTS = torch.tensor([
    1.0 / 0.0898,   # Cell Density
    1.0 / 0.0558,   # Glucose (protect this!)
    1.0 / 0.0592,   # Lactate
    1.0 / 0.1218,   # Titer
])
COMPONENT_WEIGHTS = COMPONENT_WEIGHTS / COMPONENT_WEIGHTS.sum() * 4


def calculate_metrics(y_true, y_pred):
    """Calculate standard metrics."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


def evaluate_models():
    """Compare heteroscedastic model vs simple baseline."""
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON: HETEROSCEDASTIC vs SIMPLE BASELINE")
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
    
    # Split data (same as training)
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
    
    # ========================================================================
    # Load Simple Baseline
    # ========================================================================
    print("Loading Simple Baseline...")
    if not Path('simple_baseline_model.pt').exists():
        print("⚠ Simple baseline not found, skipping comparison")
        return
    
    simple_model = SimpleBaseline(n_components=4, n_params=0, latent_dim=32)
    simple_model.load_state_dict(torch.load('simple_baseline_model.pt', map_location=device))
    simple_model.to(device)
    simple_model.eval()
    
    # ========================================================================
    # Load Heteroscedastic Model
    # ========================================================================
    print("Loading Heteroscedastic Model...")
    if not Path('heteroscedastic_model.pt').exists():
        print("⚠ Heteroscedastic model not found, skipping comparison")
        return
    
    hetero_model = HeteroscedasticMultiTask(n_components=4, n_params=0)
    hetero_model.load_state_dict(torch.load('heteroscedastic_model.pt', map_location=device))
    hetero_model.to(device)
    hetero_model.eval()
    
    # ========================================================================
    # Evaluate both models
    # ========================================================================
    print(f"\nEvaluating on {len(val_dataset)} validation reactors...\n")
    
    all_simple_pred = []
    all_hetero_pred = []
    all_hetero_logvar = []
    all_hetero_phase = []
    all_target = []
    all_target_phase = []
    
    with torch.no_grad():
        for batch in val_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            target = batch['trajectory'].to(device)
            phase_target = batch['phase'].to(device) if 'phase' in batch else None
            params = batch['parameters'].to(device)
            
            # Simple baseline prediction
            simple_out = simple_model(ic, time, params)
            simple_pred = simple_out['concentrations'].cpu().numpy()
            
            # Heteroscedastic prediction
            hetero_out = hetero_model(ic, time, params)
            hetero_pred = hetero_out['concentrations'].cpu().numpy()
            hetero_logvar = hetero_out['log_variances'].cpu().numpy()
            hetero_phase = hetero_out['phases'].cpu().numpy()
            
            all_simple_pred.append(simple_pred)
            all_hetero_pred.append(hetero_pred)
            all_hetero_logvar.append(hetero_logvar)
            all_hetero_phase.append(hetero_phase)
            all_target.append(target.cpu().numpy())
            if phase_target is not None:
                all_target_phase.append(phase_target.cpu().numpy())
    
    # Concatenate all predictions
    simple_pred_all = np.concatenate(all_simple_pred, axis=0)  # [reactors, time, components]
    hetero_pred_all = np.concatenate(all_hetero_pred, axis=0)
    hetero_logvar_all = np.concatenate(all_hetero_logvar, axis=0)
    hetero_phase_all = np.concatenate(all_hetero_phase, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    target_phase_all = np.concatenate(all_target_phase, axis=0) if all_target_phase else None
    
    # Flatten for per-component analysis
    simple_flat = simple_pred_all.reshape(-1, 4)
    hetero_flat = hetero_pred_all.reshape(-1, 4)
    target_flat = target_all.reshape(-1, 4)
    
    # ========================================================================
    # Overall Comparison
    # ========================================================================
    print(f"{'='*80}")
    print("OVERALL METRICS (all components, all time points)")
    print(f"{'='*80}\n")
    
    simple_mse_all = np.mean((simple_flat - target_flat) ** 2)
    hetero_mse_all = np.mean((hetero_flat - target_flat) ** 2)
    
    print(f"Simple Baseline: MSE = {simple_mse_all:.6f}")
    print(f"Heteroscedastic: MSE = {hetero_mse_all:.6f}")
    print(f"Improvement: {(simple_mse_all - hetero_mse_all) / simple_mse_all * 100:.1f}%\n")
    
    # ========================================================================
    # Per-Component Analysis
    # ========================================================================
    print(f"{'='*80}")
    print("PER-COMPONENT METRICS")
    print(f"{'='*80}\n")
    
    results = []
    for comp_idx, comp_name in enumerate(component_names):
        simple_y = simple_flat[:, comp_idx]
        hetero_y = hetero_flat[:, comp_idx]
        target_y = target_flat[:, comp_idx]
        
        simple_metrics = calculate_metrics(target_y, simple_y)
        hetero_metrics = calculate_metrics(target_y, hetero_y)
        
        improvement = (simple_metrics['MSE'] - hetero_metrics['MSE']) / simple_metrics['MSE'] * 100
        
        print(f"{comp_name}:")
        print(f"  Simple Baseline: MSE={simple_metrics['MSE']:.6f}, R²={simple_metrics['R²']:.4f}")
        print(f"  Heteroscedastic: MSE={hetero_metrics['MSE']:.6f}, R²={hetero_metrics['R²']:.4f}")
        print(f"  Improvement:     {improvement:+.1f}%")
        print()
        
        results.append({
            'Component': comp_name,
            'Simple_MSE': simple_metrics['MSE'],
            'Hetero_MSE': hetero_metrics['MSE'],
            'Improvement_%': improvement,
            'Simple_R2': simple_metrics['R²'],
            'Hetero_R2': hetero_metrics['R²'],
        })
    
    # ========================================================================
    # Uncertainty Analysis (Key to Heteroscedastic Model)
    # ========================================================================
    print(f"{'='*80}")
    print("UNCERTAINTY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Convert log-variance to actual variance
    hetero_var_all = np.exp(hetero_logvar_all)
    
    # Prediction error
    hetero_error = np.abs(hetero_flat - target_flat)
    
    print("Learned Uncertainty per Component (Avg log-variance):")
    for comp_idx, comp_name in enumerate(component_names):
        logvar = hetero_logvar_all.reshape(-1, 4)[:, comp_idx]
        var = hetero_var_all.reshape(-1, 4)[:, comp_idx]
        error = hetero_error[:, comp_idx]
        
        print(f"  {comp_name}:")
        print(f"    Avg log-variance: {logvar.mean():.4f} (range: {logvar.min():.4f} to {logvar.max():.4f})")
        print(f"    Avg variance:     {var.mean():.6f}")
        print(f"    Correlation(error, variance): {np.corrcoef(error, var)[0, 1]:.4f}")
        print()
    
    # ========================================================================
    # Phase Prediction Quality
    # ========================================================================
    if target_phase_all is not None:
        print(f"{'='*80}")
        print("PHASE PREDICTION QUALITY")
        print(f"{'='*80}\n")
        
        phase_pred_flat = hetero_phase_all.flatten()
        phase_target_flat = target_phase_all.flatten()
        
        phase_mse = np.mean((phase_pred_flat - phase_target_flat) ** 2)
        phase_mae = np.mean(np.abs(phase_pred_flat - phase_target_flat))
        phase_corr = np.corrcoef(phase_pred_flat, phase_target_flat)[0, 1]
        
        print(f"Phase Prediction (0-1 continuous):")
        print(f"  MSE: {phase_mse:.6f}")
        print(f"  MAE: {phase_mae:.6f}")
        print(f"  Correlation: {phase_corr:.4f}")
        print()
    
    # ========================================================================
    # Early Phase Analysis (where Titer is worst)
    # ========================================================================
    print(f"{'='*80}")
    print("EARLY PHASE ANALYSIS (phase < 0.1)")
    print(f"{'='*80}\n")
    
    if target_phase_all is not None:
        early_phase_mask = target_phase_all.flatten() < 0.1
        
        if early_phase_mask.sum() > 0:
            simple_early_mse = np.mean((simple_flat[early_phase_mask] - target_flat[early_phase_mask]) ** 2)
            hetero_early_mse = np.mean((hetero_flat[early_phase_mask] - target_flat[early_phase_mask]) ** 2)
            
            print(f"Early Phase MSE ({early_phase_mask.sum()} points):")
            print(f"  Simple Baseline: {simple_early_mse:.6f}")
            print(f"  Heteroscedastic: {hetero_early_mse:.6f}")
            print(f"  Improvement: {(simple_early_mse - hetero_early_mse) / simple_early_mse * 100:.1f}%\n")
            
            # Per-component early phase
            print("Per-component early phase MSE:")
            for comp_idx, comp_name in enumerate(component_names):
                simple_comp = simple_flat[early_phase_mask, comp_idx]
                hetero_comp = hetero_flat[early_phase_mask, comp_idx]
                target_comp = target_flat[early_phase_mask, comp_idx]
                
                simple_mse_comp = np.mean((simple_comp - target_comp) ** 2)
                hetero_mse_comp = np.mean((hetero_comp - target_comp) ** 2)
                
                print(f"  {comp_name:15s}: Simple={simple_mse_comp:.6f}, Hetero={hetero_mse_comp:.6f}")
    
    print(f"\n{'='*80}\n")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    evaluate_models()
