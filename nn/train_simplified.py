#!/usr/bin/env python3
"""
Simplified training focusing on concentration dynamics with IC constraints.
Removes phase classification - focuses on non-flatness and IC boundary conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import os

print(f"Current directory: {os.getcwd()}")

try:
    from cosmic_nn_surrogate import (
        CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
    )
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
    IMPORTS_OK = True
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class SimplifiedTrainer:
    """Trainer focused on dynamics learning with IC constraints."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.losses = []
    
    def compute_loss(self, predictions, targets, ics):
        """
        Simplified loss function:
        1. Concentration MSE (main)
        2. IC constraint (0.2× weight - strong)
        3. Non-flatness penalty (0.2× weight - strong to force dynamics)
        4. Temporal consistency (penalize abrupt changes)
        """
        conc_pred = predictions['concentrations']
        
        # Loss 1: Main concentration loss
        conc_loss = nn.functional.mse_loss(conc_pred, targets)
        
        # Loss 2: IC constraint - very strong (0.2)
        # First timepoint MUST be close to initial conditions
        ic_error = torch.mean((conc_pred[:, 0, :] - ics) ** 2)
        ic_loss = 0.2 * ic_error
        
        # Loss 3: Non-flatness penalty - very strong (0.2)
        # Penalize constant predictions heavily
        # Use coefficient of variation: std / mean
        conc_mean = torch.mean(conc_pred, dim=1, keepdim=True)  # (batch, 1, n_components)
        conc_std = torch.std(conc_pred, dim=1, keepdim=True)    # (batch, 1, n_components)
        
        # Coefficient of variation (std/mean, avoid division by zero)
        cv = conc_std / (torch.abs(conc_mean) + 1e-6)
        # Penalize LOW variance (low CV = flat)
        flatness_penalty = 0.2 * torch.mean(1.0 / (1.0 + cv))
        
        # Loss 4: Temporal consistency (small penalties for abrupt jumps)
        # Allow smooth trajectories but penalize unrealistic discontinuities
        temporal_diff = conc_pred[:, 1:, :] - conc_pred[:, :-1, :]
        temporal_penalty = 0.02 * torch.mean(temporal_diff ** 2)
        
        total_loss = conc_loss + ic_loss + flatness_penalty + temporal_penalty
        
        return total_loss, {
            'conc': conc_loss.item(),
            'ic': ic_loss.item(),
            'flatness': flatness_penalty.item(),
            'temporal': temporal_penalty.item(),
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        loss_components = {'conc': 0, 'ic': 0, 'flatness': 0, 'temporal': 0}
        n_batches = 0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(ic, time, params)
            
            # Compute loss
            loss, components = self.compute_loss(predictions, target, ic)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
            n_batches += 1
        
        epoch_loss /= n_batches
        for key in loss_components:
            loss_components[key] /= n_batches
        
        self.losses.append(epoch_loss)
        self.scheduler.step(epoch_loss)
        
        return epoch_loss, loss_components
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(self.device)
                time = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)
                
                predictions = self.model(ic, time, params)
                loss, _ = self.compute_loss(predictions, target, ic)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, components = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}: Loss={train_loss:.6f} | "
                      f"Val={val_loss:.6f} | "
                      f"IC={components['ic']:.4f} | "
                      f"Flat={components['flatness']:.4f} | "
                      f"Temp={components['temporal']:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return best_val_loss


def main():
    """Train simplified model on real data."""
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA: Simplified Training (Dynamics Focus)")
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
    
    # Create dataset (NO phases - just focus on concentration dynamics)
    dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True)
    
    # Split: 7 train, 3 val
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                             num_workers=0, collate_fn=dfba_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                           num_workers=0, collate_fn=dfba_collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print(f"\n{'='*70}")
    print("Training with Simplified Loss (IC + Non-flatness Focus)")
    print(f"{'='*70}")
    print("Loss components:")
    print("  - Concentration MSE (main)")
    print("  - IC constraint (0.2× weight) ← STRONG")
    print("  - Non-flatness penalty (0.2× weight) ← STRONG")
    print("  - Temporal consistency (0.02× weight)")
    
    trainer = SimplifiedTrainer(model, device, learning_rate=1e-3)
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=150, patience=30)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'simplified_model.pt')
    print(f"✓ Model saved: simplified_model.pt")


if __name__ == "__main__":
    main()
