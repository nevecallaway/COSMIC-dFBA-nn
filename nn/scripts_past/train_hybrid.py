#!/usr/bin/env python3
"""
Train on synthetic + real data combined.
Tests if synthetic augmentation improves generalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time

try:
    from cosmic_nn_surrogate import SimpleBaseline, dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from generate_synthetic_training import load_synthetic_dataset, generate_dataset
    from torch.utils.data import DataLoader, ConcatDataset, random_split
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class HybridTrainer:
    """Train on synthetic + real data."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def compute_loss(self, predictions, targets, ics):
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
            time_points = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(ic, time_points, params)
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
                time_points = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)
                
                predictions = self.model(ic, time_points, params)
                loss = self.compute_loss(predictions, target, ic)
                val_loss += loss.item()
                n_batches += 1
        
        return val_loss / n_batches
    
    def train(self, train_loader, val_loader, epochs=150, patience=30):
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
                print(f"  Early stopping at epoch {epoch}")
                break
        
        return best_val_loss


def main():
    """Train on synthetic + real data."""
    
    print(f"\n{'='*70}")
    print("Hybrid Training: Synthetic + Real Data")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate or load synthetic data
    syn_file = 'synthetic_training.npz'
    if not Path(syn_file).exists():
        print(f"\n{'-'*70}")
        print("Generating synthetic training data...")
        print(f"{'-'*70}")
        generate_dataset(n_samples=100, n_timepoints=13, output_file=syn_file)
    else:
        print(f"\nLoading existing synthetic data...")
    
    syn_trajectories, syn_times, syn_ics, syn_phases = load_synthetic_dataset(syn_file)
    print(f"✓ Synthetic data: {syn_trajectories.shape[0]} trajectories")
    
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
        print("Error: data_2.csv not found")
        sys.exit(1)
    
    real_trajectories, real_times, real_ics, real_metadata = load_experimental_data(data_file)
    real_phases = real_metadata['phases']
    print(f"✓ Real data: {real_trajectories.shape[0]} reactors")
    
    # Create datasets
    syn_dataset = dFBADataset(syn_trajectories, syn_times, syn_ics, parameters={}, normalize=True, phases=syn_phases)
    real_dataset = dFBADataset(real_trajectories, real_times, real_ics, parameters={}, normalize=True, phases=real_phases)
    
    # Split real into train/val
    train_size = int(0.7 * len(real_dataset))
    val_size = len(real_dataset) - train_size
    real_train_dataset, real_val_dataset = random_split(real_dataset, [train_size, val_size])
    
    # Combine synthetic + real train
    combined_train_dataset = ConcatDataset([syn_dataset, real_train_dataset])
    
    print(f"\n{'='*70}")
    print("Dataset Composition")
    print(f"{'='*70}")
    print(f"Synthetic training:  {len(syn_dataset)}")
    print(f"Real training:       {len(real_train_dataset)}")
    print(f"Combined training:   {len(combined_train_dataset)}")
    print(f"Real validation:     {len(real_val_dataset)}")
    
    train_loader = DataLoader(combined_train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dfba_collate_fn)
    val_loader = DataLoader(real_val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dfba_collate_fn)
    
    # Train model
    print(f"\n{'='*70}")
    print("Training Simple Baseline (Synthetic + Real)")
    print(f"{'='*70}")
    
    model = SimpleBaseline(n_components=real_dataset.n_components, n_params=0, latent_dim=32)
    model.to(device)
    trainer = HybridTrainer(model, device, learning_rate=1e-3)
    
    start = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=150, patience=30)
    elapsed = time.time() - start
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'simple_baseline_hybrid_model.pt')
    print(f"\n✓ Saved: simple_baseline_hybrid_model.pt")
    
    # Evaluate on real validation set
    print(f"\n{'='*70}")
    print("Evaluating on Real Data")
    print(f"{'='*70}")
    
    model.eval()
    with torch.no_grad():
        all_pred = []
        all_real = []
        for batch in val_loader:
            ic = batch['initial_conditions'].to(device)
            time_points = batch['time'].to(device)
            params = batch['parameters'].to(device)
            target = batch['trajectory'].to(device)
            
            predictions = model(ic, time_points, params)
            all_pred.append(predictions['concentrations'].cpu().numpy())
            all_real.append(target.cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_real = np.concatenate(all_real, axis=0)
    
    mse = np.mean((all_pred - all_real) ** 2)
    mae = np.mean(np.abs(all_pred - all_real))
    
    print(f"\nValidation Metrics (on real data):")
    print(f"  Overall MSE: {mse:.6f}")
    print(f"  Overall MAE: {mae:.6f}")
    
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    print(f"\nPer-Component MSE:")
    for idx, name in enumerate(component_names):
        comp_mse = np.mean((all_pred[:, :, idx] - all_real[:, :, idx]) ** 2)
        print(f"  {name}: {comp_mse:.6f}")


if __name__ == "__main__":
    main()
