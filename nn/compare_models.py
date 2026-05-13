#!/usr/bin/env python3
"""
Model Comparison Framework: Train and evaluate multiple architectures.
Compare: Phase-Aware Regression, Simple Baseline, Original Enhanced.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys
import time

try:
    from phase_aware_model import CosmicNNSurrogatePhaseAware
    from cosmic_nn_surrogate import CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class SimpleBaseline(nn.Module):
    """Baseline: No phase, just concentration prediction."""
    def __init__(self, n_components, n_params=0, latent_dim=64):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        
        input_size = n_components + n_params
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
        )
        
        # Simple decoder
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        self.attention = nn.MultiheadAttention(latent_dim, 2, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Sigmoid()
        )
    
    def forward(self, initial_conditions, time_points, parameters=None):
        if parameters is not None and parameters.shape[1] > 0:
            encoder_input = torch.cat([initial_conditions, parameters], dim=-1)
        else:
            encoder_input = initial_conditions
        
        latent_state = self.encoder(encoder_input)
        
        batch_size = initial_conditions.shape[0]
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, time_points.shape[1], -1)
        
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        
        concentrations = self.decoder(combined)
        
        return {
            'concentrations': concentrations,
            'phase_weights': torch.ones(batch_size, time_points.shape[1], 1, device=initial_conditions.device) * 0.5
        }


class UnifiedTrainer:
    """Train any model with same loss function."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def compute_loss(self, predictions, targets, ics):
        """Standard loss for all models."""
        conc_pred = predictions['concentrations']
        conc_loss = nn.functional.mse_loss(conc_pred, targets)
        ic_loss = 0.2 * torch.mean((conc_pred[:, 0, :] - ics) ** 2)
        conc_variance = torch.var(conc_pred, dim=1)
        flatness_penalty = 0.05 * torch.mean(1.0 / (1.0 + conc_variance))
        return conc_loss + ic_loss + flatness_penalty
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(ic, time, params)
            loss = self.compute_loss(predictions, target, ic)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        self.scheduler.step(epoch_loss)
        return epoch_loss
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(self.device)
                time = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)
                
                predictions = self.model(ic, time, params)
                loss = self.compute_loss(predictions, target, ic)
                val_loss += loss.item()
                n_batches += 1
        
        return val_loss / n_batches
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            if patience_counter >= patience:
                break
        
        return best_val_loss


def evaluate_model(model, device, eval_loader, phases_true=None):
    """Compute detailed metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_phases_pred = []
    
    with torch.no_grad():
        for batch in eval_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            
            predictions = model(ic, time, params)
            all_predictions.append(predictions['concentrations'].cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_phases_pred.append(predictions['phase_weights'].cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_phases_pred = np.concatenate(all_phases_pred, axis=0).squeeze(-1)
    
    # Concentration metrics
    conc_mse = np.mean((all_predictions - all_targets) ** 2)
    conc_mae = np.mean(np.abs(all_predictions - all_targets))
    
    # Per-component metrics
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    metrics = {'overall_mse': conc_mse, 'overall_mae': conc_mae}
    
    for comp_idx, name in enumerate(component_names):
        pred_comp = all_predictions[:, :, comp_idx].flatten()
        real_comp = all_targets[:, :, comp_idx].flatten()
        mse = np.mean((pred_comp - real_comp) ** 2)
        mae = np.mean(np.abs(pred_comp - real_comp))
        metrics[f'{name}_mse'] = mse
        metrics[f'{name}_mae'] = mae
    
    # Phase metrics if available
    if phases_true is not None:
        phases_true_flat = np.concatenate([phases_true[i] for i in range(len(phases_true))])
        phase_mae = np.mean(np.abs(all_phases_pred.flatten() - phases_true_flat))
        phase_rmse = np.sqrt(np.mean((all_phases_pred.flatten() - phases_true_flat) ** 2))
        metrics['phase_mae'] = phase_mae
        metrics['phase_rmse'] = phase_rmse
    
    return metrics


def main():
    """Compare models."""
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA: Model Comparison")
    print(f"{'='*70}")
    
    # Load data
    print(f"\nLoading data...")
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
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dfba_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Evaluation on full dataset
    eval_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Create models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_comp = dataset.n_components
    
    models_config = [
        ('Phase-Aware Regression', CosmicNNSurrogatePhaseAware(n_comp, 0, 32, 2)),
        ('Original Enhanced', CosmicNNSurrogateEnhanced(n_comp, 0, 32, 2)),
        ('Simple Baseline', SimpleBaseline(n_comp, 0, 32)),
    ]
    
    results = []
    
    for model_name, model in models_config:
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        
        model.to(device)
        trainer = UnifiedTrainer(model, device, learning_rate=1e-3)
        
        start = time.time()
        best_val_loss = trainer.train(train_loader, val_loader, epochs=100, patience=20)
        elapsed = time.time() - start
        
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, device, eval_loader, phases_true=phases)
        
        metrics['model'] = model_name
        metrics['training_time'] = elapsed
        metrics['best_val_loss'] = best_val_loss
        metrics['n_params'] = sum(p.numel() for p in model.parameters())
        
        results.append(metrics)
        
        print(f"  ✓ Completed in {elapsed:.1f}s")
    
    # Summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(results)
    
    # Display key metrics
    display_cols = ['model', 'n_params', 'training_time', 'best_val_loss', 'overall_mse', 'overall_mae', 'phase_mae']
    print(df[display_cols].to_string(index=False))
    
    # Detailed per-component comparison
    print(f"\n{'='*70}")
    print("Per-Component Errors (MSE)")
    print(f"{'='*70}")
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    for comp in component_names:
        print(f"\n{comp}:")
        for _, row in df.iterrows():
            mse = row.get(f'{comp}_mse', 'N/A')
            print(f"  {row['model']}: {mse:.6f}" if isinstance(mse, float) else f"  {row['model']}: {mse}")
    
    # Best overall
    print(f"\n{'='*70}")
    print("Rankings")
    print(f"{'='*70}")
    print(f"\nBest by Validation Loss: {df.loc[df['best_val_loss'].idxmin(), 'model']}")
    print(f"Best by Overall MSE: {df.loc[df['overall_mse'].idxmin(), 'model']}")
    if 'phase_mae' in df.columns:
        print(f"Best by Phase MAE: {df.loc[df['phase_mae'].idxmin(), 'model']}")
    print(f"Fastest: {df.loc[df['training_time'].idxmin(), 'model']}")
    
    # Save results
    df.to_csv('model_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to model_comparison_results.csv")


if __name__ == "__main__":
    main()
