#!/usr/bin/env python3
"""
Heteroscedastic Multi-Task Model:
- Predicts phase (growth vs production)
- Predicts 4 concentrations WITH uncertainty
- Uses Gaussian NLL loss to learn when to be uncertain (especially early phase)
- Phase prediction helps condition concentration predictions
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

try:
    from cosmic_nn_surrogate import dFBADataset, dfba_collate_fn
    from load_real_data import load_experimental_data
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class HeteroscedasticMultiTask(nn.Module):
    """
    Multi-task model that jointly optimizes:
    1. Phase prediction (growth vs production classification)
    2. Concentration predictions (4 components)
    3. Uncertainty estimates (learns when to be uncertain)
    
    Key insight: Titer is uncertain in early phase (high variance).
    By learning separate variance for each time step, the model can express
    "I'm confident in this prediction" vs "This is noisy, I'm uncertain"
    """
    
    def __init__(self, n_components=4, n_params=0, latent_dim=32, hidden_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_components = n_components
        
        # ====================================================================
        # Shared encoder: Learn representation from initial conditions
        # ====================================================================
        self.ic_encoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # ====================================================================
        # Time encoder: Embed time point
        # ====================================================================
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # ====================================================================
        # Task 1: Phase prediction head (0-1 continuous, growth vs production)
        # ====================================================================
        self.phase_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Phase in [0, 1]
        )
        
        # ====================================================================
        # Task 2: Concentration predictions with uncertainty
        # We predict mean AND log-variance for each component
        # ====================================================================
        # Decode to means
        self.conc_mean_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
            nn.Sigmoid(),  # Concentrations in [0, 1]
        )
        
        # Decode to log-variances (learns uncertainty)
        # These are UNconstrained (can be negative)
        self.conc_logvar_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
            # No activation: log-variance can be any value
        )
    
    def forward(self, ic, time, params=None):
        """
        Args:
            ic: Initial conditions [batch, n_components]
            time: Time points [batch, time_steps]
            params: Parameters [batch, n_params] (unused)
        
        Returns:
            Dict with predictions, uncertainties, and phase
        """
        batch_size = ic.shape[0]
        n_steps = time.shape[1]
        
        # Encode initial condition
        ic_latent = self.ic_encoder(ic)  # [batch, latent_dim]
        
        # Store predictions
        phase_preds = []
        conc_means = []
        conc_logvars = []
        
        for t in range(n_steps):
            # Encode time point
            t_emb = self.time_encoder(time[:, t:t+1])  # [batch, latent_dim]
            
            # Combine initial state + time
            combined = torch.cat([ic_latent, t_emb], dim=1)  # [batch, latent_dim*2]
            
            # ============================================================
            # Predict phase
            # ============================================================
            phase_pred = self.phase_head(combined)  # [batch, 1]
            phase_preds.append(phase_pred)
            
            # ============================================================
            # Predict concentrations (mean and variance)
            # ============================================================
            mean = self.conc_mean_head(combined)  # [batch, n_components]
            logvar = self.conc_logvar_head(combined)  # [batch, n_components]
            
            conc_means.append(mean)
            conc_logvars.append(logvar)
        
        # Stack over time dimension
        phases = torch.cat(phase_preds, dim=1)  # [batch, time_steps]
        means = torch.stack(conc_means, dim=1)  # [batch, time_steps, n_components]
        logvars = torch.stack(conc_logvars, dim=1)  # [batch, time_steps, n_components]
        
        return {
            'phases': phases,
            'concentrations': means,
            'log_variances': logvars,
        }


def gaussian_nll_loss(pred_mean, pred_logvar, target, component_weights=None):
    """
    Gaussian Negative Log-Likelihood loss with optional component weighting.
    
    The model learns to predict not just the mean, but also the variance.
    This naturally learns uncertainty: high variance in uncertain regions,
    low variance in confident regions.
    
    L = 0.5 * (log(var) + (y - mu)^2 / var)
    
    Args:
        pred_mean: Predicted mean [batch, time, components]
        pred_logvar: Predicted log-variance [batch, time, components]
        target: Ground truth [batch, time, components]
        component_weights: [components] weights (default: uniform)
    
    Returns:
        Scalar loss
    """
    precision = torch.exp(-pred_logvar)  # 1 / variance
    mse = (target - pred_mean) ** 2
    loss = 0.5 * (pred_logvar + mse * precision)
    
    # Apply component weights if provided
    if component_weights is not None:
        weights = torch.tensor(component_weights, device=loss.device, dtype=loss.dtype)
        loss = loss * weights.unsqueeze(0).unsqueeze(0)
    
    return loss.mean()


def train_heteroscedastic(epochs=150, batch_size=2, learning_rate=1e-3):
    """Train heteroscedastic multi-task model."""
    
    print(f"\n{'='*70}")
    print("HETEROSCEDASTIC MULTI-TASK MODEL")
    print("Tasks: Phase prediction + Concentration prediction with uncertainty")
    print(f"{'='*70}\n")
    
    # Component importance weights (based on simple baseline MSE)
    # Inverse of baseline MSE: easier components get higher weight
    # This prevents the model from giving up on easy targets like Glucose
    component_weights = torch.tensor([
        1.0 / 0.0898,   # Cell Density: harder
        1.0 / 0.0558,   # Glucose: easy (SHOULD BE PROTECTED)
        1.0 / 0.0592,   # Lactate: medium
        1.0 / 0.1218,   # Titer: hardest (focus here)
    ])
    component_weights = component_weights / component_weights.sum() * 4  # Normalize
    
    print(f"Component weights (to prevent abandoning easy targets):")
    for i, (name, w) in enumerate(zip(['Cell Density', 'Glucose', 'Lactate', 'Titer'], 
                                        component_weights.numpy())):
        print(f"  {name:15s}: {w:.4f}")
    print()
    
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
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroscedasticMultiTask(n_components=4, n_params=0, latent_dim=32, hidden_dim=64)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    phase_criterion = nn.MSELoss()  # Phase prediction uses MSE
    
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Loss: Gaussian NLL (concentrations) + MSE (phase)\n")
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_conc_loss = 0.0
        train_phase_loss = 0.0
        train_count = 0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(device)
            time = batch['time'].to(device)
            target_traj = batch['trajectory'].to(device)
            phase_target = batch['phase'].to(device) if 'phase' in batch else None
            params = batch['parameters'].to(device)
            
            optimizer.zero_grad()
            
            output = model(ic, time, params)
            pred_means = output['concentrations']
            pred_logvars = output['log_variances']
            pred_phases = output['phases']
            
            # Concentration loss (Gaussian NLL)
            conc_loss = gaussian_nll_loss(pred_means, pred_logvars, target_traj, 
                                         component_weights=component_weights)
            
            # Phase loss (MSE)
            if phase_target is not None:
                phase_loss = phase_criterion(pred_phases, phase_target)
            else:
                phase_loss = 0.0
            
            # Combined loss (weighted)
            total_loss = conc_loss + 0.5 * phase_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_conc_loss += conc_loss.item() * ic.shape[0]
            train_phase_loss += (phase_loss.item() if phase_loss != 0 else 0) * ic.shape[0]
            train_count += ic.shape[0]
        
        train_conc_loss /= train_count
        train_phase_loss /= train_count if phase_loss != 0 else 1
        
        # Validation
        model.eval()
        val_conc_loss = 0.0
        val_phase_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(device)
                time = batch['time'].to(device)
                target_traj = batch['trajectory'].to(device)
                phase_target = batch['phase'].to(device) if 'phase' in batch else None
                params = batch['parameters'].to(device)
                
                output = model(ic, time, params)
                pred_means = output['concentrations']
                pred_logvars = output['log_variances']
                pred_phases = output['phases']
                
                conc_loss = gaussian_nll_loss(pred_means, pred_logvars, target_traj,
                                             component_weights=component_weights)
                
                if phase_target is not None:
                    phase_loss = phase_criterion(pred_phases, phase_target)
                else:
                    phase_loss = 0.0
                
                val_conc_loss += conc_loss.item() * ic.shape[0]
                val_phase_loss += (phase_loss.item() if phase_loss != 0 else 0) * ic.shape[0]
                val_count += ic.shape[0]
        
        val_conc_loss /= val_count
        val_phase_loss /= val_count if phase_loss != 0 else 1
        val_total = val_conc_loss + 0.5 * val_phase_loss
        
        # Early stopping
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(model.state_dict(), 'heteroscedastic_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Conc Loss={train_conc_loss:.6f}, "
                  f"Phase Loss={train_phase_loss:.6f} | "
                  f"Val Conc={val_conc_loss:.6f}, Phase={val_phase_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to heteroscedastic_model.pt\n")
    
    return model


if __name__ == "__main__":
    train_heteroscedastic(epochs=200, batch_size=2, learning_rate=1e-3)
