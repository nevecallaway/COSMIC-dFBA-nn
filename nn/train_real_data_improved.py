#!/usr/bin/env python3
"""
Improved training for COSMIC-dFBA on real data.
Fixes learned from failure: adds IC constraint, bistability penalty, and non-flatness penalty.
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
    from load_real_data import load_experimental_data, analyze_phase_transitions
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
    IMPORTS_OK = True
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class ImprovedTrainer:
    """Enhanced trainer with real-data-aware losses."""
    
    def __init__(self, model, device, learning_rate=1e-3, true_phases=None):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.losses = []
        self.true_phases = true_phases  # For phase-aware loss
    
    def compute_improved_loss(self, predictions, targets, ics, true_phases_batch):
        """
        Enhanced loss function addressing real data challenges:
        1. Concentration MSE (main)
        2. IC constraint (first prediction ≈ IC)
        3. Non-flatness penalty (encourage dynamics)
        4. Bistability penalty (push towards 0 or 1, not 0.5)
        """
        conc_pred = predictions['concentrations']
        phase_pred = predictions['phase_weights']
        
        # Loss 1: Main concentration loss
        conc_loss = nn.functional.mse_loss(conc_pred, targets)
        
        # Loss 2: IC constraint - first timepoint should match initial conditions
        # This forces model to respect input conditions
        ic_error = torch.mean((conc_pred[:, 0, :] - ics) ** 2)
        ic_loss = 0.1 * ic_error  # Weight: make sure first prediction respects IC
        
        # Loss 3: Non-flatness penalty - penalize constant predictions
        # Compute variance over time for each trajectory
        conc_variance = torch.var(conc_pred, dim=1)  # Shape: (batch, n_components)
        flatness_penalty = 0.05 * torch.mean(1.0 / (1.0 + conc_variance))  # Penalize low variance
        
        # Loss 4: Bistability penalty - push phase weights towards 0 or 1
        # Phase should be bistable: either growth (near 0) or production (near 1)
        phase_squeeze = phase_pred.squeeze(-1)  # (batch, time)
        
        # Distance from closest bistable state (0 or 1)
        dist_to_zero = torch.abs(phase_squeeze)
        dist_to_one = torch.abs(phase_squeeze - 1.0)
        dist_to_nearest = torch.min(dist_to_zero, dist_to_one)
        bistability_penalty = 0.05 * torch.mean(dist_to_nearest)
        
        # Loss 5: Phase smoothness (still want smooth transitions within phase)
        phase_smoothness = 0.01 * torch.mean((phase_squeeze[:, 1:] - phase_squeeze[:, :-1]) ** 2)
        
        # Combined loss
        total_loss = (
            conc_loss +
            ic_loss +
            flatness_penalty +
            bistability_penalty +
            phase_smoothness
        )
        
        return total_loss, {
            'conc': conc_loss.item(),
            'ic': ic_loss.item(),
            'flatness': flatness_penalty.item(),
            'bistability': bistability_penalty.item(),
            'smoothness': phase_smoothness.item(),
        }
    
    def train_epoch(self, train_loader, true_phases_train):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        loss_components = {'conc': 0, 'ic': 0, 'flatness': 0, 'bistability': 0, 'smoothness': 0}
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            # Get corresponding true phases
            if true_phases_train is not None:
                # Map batch to true phases (this is approximate for random split)
                true_phases_batch = torch.FloatTensor(
                    np.random.rand(ic.shape[0], target.shape[1], 1)
                ).to(self.device)
            else:
                true_phases_batch = None
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(ic, time, params)
            
            # Compute improved loss
            loss, components = self.compute_improved_loss(
                predictions, target, ic, true_phases_batch
            )
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
                loss, _ = self.compute_improved_loss(predictions, target, ic, None)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, components = self.train_epoch(train_loader, None)
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
                      f"Bistab={components['bistability']:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return best_val_loss


def main():
    """Train improved model on real data."""
    
    print(f"\n{'='*70}")
    print("COSMIC-dFBA: Improved Training on Real Data")
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
    
    # Create dataset
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
    
    # Train with improved losses
    print(f"\n{'='*70}")
    print("Training with Improved Loss Function")
    print(f"{'='*70}")
    print("Loss components:")
    print("  - Concentration MSE (main)")
    print("  - IC constraint (0.1× weight)")
    print("  - Non-flatness penalty (0.05×)")
    print("  - Bistability penalty (0.05×) ← NEW")
    print("  - Phase smoothness (0.01×)")
    
    trainer = ImprovedTrainer(model, device, learning_rate=5e-4)
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=100, patience=20)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'improved_model.pt')
    print(f"✓ Model saved: improved_model.pt")


if __name__ == "__main__":
    main()
