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
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.losses = []
    
    def compute_improved_loss(self, predictions, targets, ics, phases_batch=None):
        """
        Enhanced loss function with binary phase classification:
        1. Concentration MSE (main)
        2. IC constraint (first prediction ≈ IC)
        3. Non-flatness penalty (encourage dynamics)
        4. Phase classification loss (cross-entropy with binary targets)
        """
        conc_pred = predictions['concentrations']
        phase_logits = predictions['phase_weights']  # Now logits, not sigmoid outputs
        
        # Loss 1: Main concentration loss
        conc_loss = nn.functional.mse_loss(conc_pred, targets)
        
        # Loss 2: IC constraint - first timepoint should match initial conditions
        ic_error = torch.mean((conc_pred[:, 0, :] - ics) ** 2)
        ic_loss = 0.1 * ic_error
        
        # Loss 3: Non-flatness penalty - penalize constant predictions
        conc_variance = torch.var(conc_pred, dim=1)  # Shape: (batch, n_components)
        flatness_penalty = 0.05 * torch.mean(1.0 / (1.0 + conc_variance))
        
        # Loss 4: Phase classification with binary cross-entropy
        # phases_batch shape: (batch_size, n_timepoints) - ground truth phase fractions
        if phases_batch is not None:
            batch_size = phases_batch.shape[0]
            n_time = phases_batch.shape[1]
            
            # Classify: <0.2 = growth (0), >0.8 = production (1)
            # Ambiguous middle (0.2-0.8) = ignore (masked)
            phase_targets = torch.zeros((batch_size, n_time), dtype=torch.long, device=phases_batch.device)
            mask = torch.zeros((batch_size, n_time), dtype=torch.bool, device=phases_batch.device)
            
            for b in range(batch_size):
                for t in range(n_time):
                    p = phases_batch[b, t].item()
                    if p < 0.2:
                        phase_targets[b, t] = 0  # Growth
                        mask[b, t] = True
                    elif p > 0.8:
                        phase_targets[b, t] = 1  # Production
                        mask[b, t] = True
                    # else: mask = False (ignore transition zone)
            
            # Reshape phase_logits: (batch, time, 2) -> (batch*time, 2)
            phase_logits_flat = phase_logits.view(-1, 2)
            phase_targets_flat = phase_targets.view(-1)
            mask_flat = mask.view(-1)
            
            # Apply mask and compute cross-entropy only on clear phase regions
            if mask_flat.sum() > 0:
                phase_loss = nn.functional.cross_entropy(
                    phase_logits_flat[mask_flat],
                    phase_targets_flat[mask_flat]
                )
                phase_loss = 0.5 * phase_loss  # Weight: 0.5x
            else:
                phase_loss = torch.tensor(0.0, device=targets.device)
        else:
            phase_loss = torch.tensor(0.0, device=targets.device)
        
        # Combined loss
        total_loss = conc_loss + ic_loss + flatness_penalty + phase_loss
        
        return total_loss, {
            'conc': conc_loss.item(),
            'ic': ic_loss.item(),
            'flatness': flatness_penalty.item(),
            'phase_ce': phase_loss.item() if isinstance(phase_loss, torch.Tensor) else phase_loss,
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        loss_components = {'conc': 0, 'ic': 0, 'flatness': 0, 'phase_ce': 0}
        n_batches = 0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            # Extract phases from batch if available
            phases_batch = batch.get('phases', None)
            if phases_batch is not None:
                phases_batch = phases_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(ic, time, params)
            
            # Compute improved loss
            loss, components = self.compute_improved_loss(
                predictions, target, ic, phases_batch
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
                      f"Phase_CE={components['phase_ce']:.4f}")
            
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
    phases = metadata['phases']  # Extract ground truth phases
    
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
    print(f"Phase data shape: {phases.shape}")
    
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
    print("Training with Binary Phase Classification")
    print(f"{'='*70}")
    print("Loss components:")
    print("  - Concentration MSE (main)")
    print("  - IC constraint (0.1× weight)")
    print("  - Non-flatness penalty (0.05×)")
    print("  - Phase binary cross-entropy (0.5× weight) ← NEW")
    print("  - Classification: <0.2=growth(0), >0.8=production(1)")
    
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
