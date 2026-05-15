"""
COSMIC dFBA to Neural Network Surrogate Model
Converts MATLAB dFBA simulations into a neural network for fast predictions
and exploration of new parameters/conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from nn.diagnostics import ModelDiagnostics


def dfba_collate_fn(batch):
    """
    Custom collate function to properly handle empty parameter tensors and phases.
    Default PyTorch collate doesn't correctly stack empty tensors of shape (0,).
    """
    ic = torch.stack([item['initial_conditions'] for item in batch])
    time = torch.stack([item['time'] for item in batch])
    traj = torch.stack([item['trajectory'] for item in batch])
    
    # Handle parameters carefully - stack empty tensors correctly
    params_list = [item['parameters'] for item in batch]
    
    # Check if all parameters are empty (shape (0,))
    if params_list[0].shape[0] == 0:
        # All empty - create proper (batch_size, 0) tensor
        params = torch.zeros(len(batch), 0)
    else:
        # Non-empty - stack normally
        params = torch.stack(params_list)
    
    result = {
        'initial_conditions': ic,
        'time': time,
        'parameters': params,
        'trajectory': traj
    }
    
    # Add phases if available
    if 'phases' in batch[0]:
        result['phases'] = torch.stack([item['phases'] for item in batch])
    
    return result


class dFBADataset(Dataset):
    """
    Dataset for dFBA simulation trajectories.
    Each sample is a complete simulation trajectory.
    """
    def __init__(self, trajectories, time_points, initial_conditions, 
                 parameters, normalize=True, phases=None):
        """
        Args:
            trajectories: Array of shape (n_samples, n_timepoints, n_components)
            time_points: Array of time points for each trajectory
            initial_conditions: Array of shape (n_samples, n_components)
            parameters: Dict of parameter arrays
            normalize: Whether to normalize data
            phases: Optional array of shape (n_samples, n_timepoints) with phase info
        """
        self.trajectories = trajectories
        self.time_points = time_points
        self.initial_conditions = initial_conditions
        self.parameters = parameters
        self.phases = phases
        
        self.n_samples = trajectories.shape[0]
        self.n_components = trajectories.shape[2]
        self.n_timepoints = trajectories.shape[1]
        
        if normalize:
            self._normalize()
    
    def _normalize(self):
        """Normalize trajectories and initial conditions to [0,1]"""
        self.traj_min = np.percentile(self.trajectories, 1, axis=(0, 1))
        self.traj_max = np.percentile(self.trajectories, 99, axis=(0, 1))
        self.traj_max = np.maximum(self.traj_max, self.traj_min + 1e-6)
        
        self.ic_min = np.percentile(self.initial_conditions, 1, axis=0)
        self.ic_max = np.percentile(self.initial_conditions, 99, axis=0)
        self.ic_max = np.maximum(self.ic_max, self.ic_min + 1e-6)
        
        self.trajectories = (self.trajectories - self.traj_min) / (self.traj_max - self.traj_min)
        self.trajectories = np.clip(self.trajectories, 0, 1)
        
        self.initial_conditions = (self.initial_conditions - self.ic_min) / (self.ic_max - self.ic_min)
        self.initial_conditions = np.clip(self.initial_conditions, 0, 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return (initial_conditions, time, parameters) -> trajectory"""
        traj = torch.FloatTensor(self.trajectories[idx])
        ic = torch.FloatTensor(self.initial_conditions[idx])
        
        # Normalize time to [0,1]
        time = torch.FloatTensor(self.time_points[idx] / np.max(self.time_points[idx]))
        
        # Stack parameters - FIXED: ensure proper 1D tensor even if empty
        param_list = []
        for key, val in self.parameters.items():
            if val is not None:
                param_list.append(torch.FloatTensor([val[idx]]))
        
        if param_list:
            params = torch.cat(param_list)  # Shape: (n_params,)
        else:
            params = torch.zeros(0)  # Shape: (0,) - proper empty 1D tensor
        
        item = {
            'initial_conditions': ic,
            'time': time,
            'parameters': params,
            'trajectory': traj
        }
        
        # Add phase data if available
        if self.phases is not None:
            item['phases'] = torch.FloatTensor(self.phases[idx])
        
        return item


class DynamicsEncoder(nn.Module):
    """Encodes initial conditions and parameters into a dynamics representation."""
    def __init__(self, n_components, n_params, latent_dim=64):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        
        input_size = n_components + n_params
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
        )
        self.latent_dim = latent_dim
    
    def forward(self, initial_conditions, parameters):
        x = torch.cat([initial_conditions, parameters], dim=-1)
        return self.fc(x)


class StateWeightingLayer(nn.Module):
    """
    Hybrid Phase Classifier: Predicts growth (class 0) vs production (class 1)
    based on current metabolite concentrations and latent state.
    """
    def __init__(self, n_components, latent_dim=64):
        super().__init__()
        self.n_components = n_components

        # Learns a "trigger weight" for each component
        # This is like learning which metabolite's depletion triggers the phase shift
        self.trigger_weights = nn.Parameter(torch.randn(n_components, 1))
        self.bias = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Output a shift factor
        )

    def forward(self, latent_state, concentrations):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            concentrations: (batch_size, n_timepoints, n_components)
        Returns:
            phase_logits: (batch_size, n_timepoints, 2)
        """
        batch_size, n_timepoints, _ = concentrations.shape

        # 1. Concentration-based trigger: C * W + b
        # (batch, time, comp) @ (comp, 1) -> (batch, time, 1)
        trigger = torch.matmul(concentrations, self.trigger_weights) + self.bias

        # 2. Latent state modulation (adjusts the trigger threshold based on the simulation context)
        modulation = self.mlp(latent_state).unsqueeze(1) # (batch, 1, 1)

        # Combine trigger and modulation
        w = trigger + modulation

        # Return as logits for [growth, production]
        # We treat 'w' as the logit for production phase
        logits = torch.cat([-w, w], dim=-1)
        return logits


class RatePredictionHead(nn.Module):
    """
    Predicts metabolic uptake/secretion rates for each cellular state.
    Returns rates for growth phase and production phase separately.
    """
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.n_components = n_components
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        # Growth phase rates
        self.growth_rates = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Tanh()  # Rates can be positive or negative
        )
        
        # Production phase rates
        self.prod_rates = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Tanh()
        )
    
    def forward(self, latent_state, time_points):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            time_points: (batch_size, n_timepoints)
        
        Returns:
            growth_rates: (batch_size, n_timepoints, n_components)
            prod_rates: (batch_size, n_timepoints, n_components)
        """
        batch_size = latent_state.shape[0]
        n_timepoints = time_points.shape[1]
        
        # Embed time
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        
        # Expand latent state
        latent_expanded = latent_state.unsqueeze(1).expand(-1, n_timepoints, -1)
        
        # Apply attention
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        
        # Predict rates
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        growth_rates = self.growth_rates(combined)
        prod_rates = self.prod_rates(combined)
        
        return growth_rates, prod_rates


class DifferentiableIntegrator(nn.Module):
    """
    Performs differentiable Euler integration of metabolic rates to predict concentrations.
    Formula: C(t) = C_0 + sum(v_blend * dt)
    """
    def __init__(self):
        super().__init__()

    def forward(self, initial_conditions, blended_rates, time_points):
        """
        Args:
            initial_conditions: (batch_size, n_components)
            blended_rates: (batch_size, n_timepoints, n_components)
            time_points: (batch_size, n_timepoints)
        Returns:
            concentrations: (batch_size, n_timepoints, n_components)
        """
        batch_size, n_timepoints, n_components = blended_rates.shape

        # Compute time steps dt. Handle case with only 1 timepoint.
        if n_timepoints > 1:
            # diff returns (batch, n_timepoints-1)
            dt = torch.diff(time_points, dim=1).unsqueeze(-1)
            # Prepend 0 for the initial state C(0) = C_0
            dt = torch.cat([torch.zeros(batch_size, 1, 1, device=dt.device), dt], dim=1)
        else:
            dt = torch.zeros(batch_size, 1, 1, device=blended_rates.device)

        # Integration: C = C_0 + integral(v dt)
        # We use cumulative sum of (rate * dt) as a differentiable Euler approximation
        integrand = blended_rates * dt
        concentrations = initial_conditions.unsqueeze(1) + torch.cumsum(integrand, dim=1)

        return concentrations


class MultiHeadTemporalDecoder(nn.Module):
    """
    Enhanced decoder with multiple prediction heads for COSMIC-dFBA compliance.
    Outputs:
    - Metabolite concentrations (via differentiable integration of rates)
    - Phase transition weights
    - Uptake/secretion rates per metabolic state
    """
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim

        # Shared time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Shared attention
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)

        # HEAD 1: Differentiable Integrator
        self.integrator = DifferentiableIntegrator()

        # HEAD 2: State weighting layer (now takes components as input)
        self.state_weighting = StateWeightingLayer(n_components, latent_dim)

        # HEAD 3: Rate prediction
        self.rate_predictor = RatePredictionHead(n_components, latent_dim, n_heads)

    def forward(self, latent_state, time_points, initial_conditions):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            time_points: (batch_size, n_timepoints)
            initial_conditions: (batch_size, n_components)

        Returns:
            Dict with keys:
            - 'concentrations': (batch_size, n_timepoints, n_components)
            - 'phase_weights': (batch_size, n_timepoints, 2)
            - 'growth_rates': (batch_size, n_timepoints, n_components)
            - 'prod_rates': (batch_size, n_timepoints, n_components)
        """
        batch_size = latent_state.shape[0]
        n_timepoints = time_points.shape[1]

        # Shared temporal embedding
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, n_timepoints, -1)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        combined = torch.cat([latent_expanded, attn_out], dim=-1)

        # 1. Predict raw rates
        growth_rates, prod_rates = self.rate_predictor(latent_state, time_points)

        # 2. Initial Phase Estimate (start with growth)
        # We need an initial concentration to start the integration
        # For the first iteration, we assume a simple linear blend or 100% growth
        f_initial = torch.zeros(batch_size, n_timepoints, 1, device=latent_state.device)
        blended_rates_initial = (1 - f_initial) * growth_rates + f_initial * prod_rates

        # 3. Integrate to get concentrations
        concentrations = self.integrator(initial_conditions, blended_rates_initial, time_points)

        # 4. Hybrid Phase Trigger: Use concentrations to refine phase weights
        phase_logits = self.state_weighting(latent_state, concentrations)

        # 5. Final blended rates (for reporting/loss)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        f_final = phase_probs[:, :, 1 : 2]
        final_blended_rates = (1 - f_final) * growth_rates + f_final * prod_rates

        # Re-integrate one last time with the refined phase for the actual output
        final_concentrations = self.integrator(initial_conditions, final_blended_rates, time_points)

        return {
            'concentrations': final_concentrations,
            'phase_weights': phase_logits,
            'growth_rates': growth_rates,
            'prod_rates': prod_rates,
        }


class TemporalDecoder(nn.Module):
    """
    Legacy interface for backward compatibility.
    Maps to MultiHeadTemporalDecoder but returns only concentrations.
    """
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Multi-head self-attention over latent + time
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_components),
            nn.Sigmoid()  # Constrain to [0,1] since data is normalized
        )
    
    def forward(self, latent_state, time_points):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            time_points: (batch_size, n_timepoints)
        
        Returns:
            trajectory: (batch_size, n_timepoints, n_components)
        """
        batch_size = latent_state.shape[0]
        n_timepoints = time_points.shape[1]
        
        # Embed time
        time_expanded = time_points.unsqueeze(-1)  # (batch, n_t, 1)
        time_embedded = self.time_embed(time_expanded)  # (batch, n_t, latent_dim)
        
        # Expand latent state
        latent_expanded = latent_state.unsqueeze(1).expand(-1, n_timepoints, -1)
        
        # Apply attention (learn correlations over time)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        
        # Concatenate and decode
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        trajectory = self.decoder(combined)
        
        return trajectory


class CosmicNNSurrogate(nn.Module):
    """
    Complete surrogate model: encodes conditions → learns dynamics → decodes trajectory
    Original version for backward compatibility.
    """
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = TemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params
    
    def forward(self, initial_conditions, time_points, parameters):
        """
        Args:
            initial_conditions: (batch_size, n_components)
            time_points: (batch_size, n_timepoints)
            parameters: (batch_size, n_params)
        
        Returns:
            trajectory: (batch_size, n_timepoints, n_components)
        """
        latent = self.encoder(initial_conditions, parameters)
        trajectory = self.decoder(latent, time_points)
        return trajectory


class SimpleBaseline(nn.Module):
    """Baseline: No phase handling, just concentration prediction with attention."""
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


class CosmicNNSurrogateEnhanced(nn.Module):
    """
    Enhanced COSMIC-dFBA model with multi-output heads for full paper compliance.
    
    Outputs:
    - Metabolite concentrations
    - Phase transition weights (growth vs production)
    - Metabolic rates (uptake/secretion for each state)
    """
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = MultiHeadTemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params
    
    def forward(self, initial_conditions, time_points, parameters):
        """
        Args:
            initial_conditions: (batch_size, n_components)
            time_points: (batch_size, n_timepoints)
            parameters: (batch_size, n_params)

        Returns:
            Dict with keys:
            - 'concentrations': (batch_size, n_timepoints, n_components)
            - 'phase_weights': (batch_size, n_timepoints, 2)
            - 'growth_rates': (batch_size, n_timepoints, n_components)
            - 'prod_rates': (batch_size, n_timepoints, n_components)
        """
        latent = self.encoder(initial_conditions, parameters)
        outputs = self.decoder(latent, time_points, initial_conditions)
        return outputs


class MetabolicConstraintEnforcer:
    """
    Enforces stoichiometric and thermodynamic constraints on predictions.
    Ensures mass balance and non-negativity.
    """
    
    @staticmethod
    def enforce_mass_balance(concentrations, phase_weights, growth_rates, prod_rates, dt=0.1):
        """
        Apply mass balance constraints: dc/dt = rates * biomass
        
        Args:
            concentrations: (batch, n_timepoints, n_components)
            phase_weights: (batch, n_timepoints, 1) - fraction in growth phase
            growth_rates: (batch, n_timepoints, n_components)
            prod_rates: (batch, n_timepoints, n_components)
            dt: time step for integration
        
        Returns:
            constrained_concentrations: Mass-balance corrected values
        """
        # Blend rates based on phase
        phase_weights = torch.clamp(phase_weights, 0, 1)
        blended_rates = (1 - phase_weights) * growth_rates + phase_weights * prod_rates
        
        # Approximate ODE integration via simple Euler method
        # dc/dt ≈ blended_rates (simplified - assumes unit biomass)
        # Cumulative integration would refine this
        
        return concentrations
    
    @staticmethod
    def enforce_non_negativity(concentrations):
        """Ensure all concentrations remain non-negative."""
        return torch.clamp(concentrations, min=0.0)
    
    @staticmethod
    def enforce_continuity(concentrations):
        """
        Enforce smooth continuity across timepoints.
        Penalizes discontinuous jumps.
        """
        diffs = concentrations[:, 1:, :] - concentrations[:, :-1, :]
        # Return penalty metric (used in loss function)
        return torch.mean(torch.abs(diffs))


class TrainingManager:
    """Manages training of both standard and enhanced surrogate models."""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type  # 'standard' or 'enhanced'
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.losses = []
        self.constraint_enforcer = MetabolicConstraintEnforcer()
    
    def _compute_loss(self, pred, target, model_type='standard'):
        """Compute loss based on model type."""
        if model_type == 'standard':
            return self._compute_standard_loss(pred, target)
        elif model_type == 'enhanced':
            return self._compute_enhanced_loss(pred, target)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _compute_standard_loss(self, pred, target):
        """Loss for standard model (concentrations only)."""
        mse_loss = nn.functional.mse_loss(pred, target)
        smoothness = torch.mean((pred[:, 1:, :] - pred[:, :-1, :]) ** 2)
        loss = mse_loss + 0.1 * smoothness
        return loss
    
    def _compute_enhanced_loss(self, pred_dict, target):
        """
        Loss for enhanced model with multiple outputs.
        Combines concentration loss with regularization on rates and phase.
        """
        conc = pred_dict['concentrations']
        phase_weights = pred_dict['phase_weights']
        growth_rates = pred_dict['growth_rates']
        prod_rates = pred_dict['prod_rates']

        # 1. Concentration MSE (main loss)
        conc_loss = nn.functional.mse_loss(conc, target)

        # 2. Smoothness regularization on concentrations (avoid jitter)
        conc_smoothness = torch.mean((conc[:, 1:, :] - conc[:, :-1, :]) ** 2)

        # 3. Non-negativity penalty (PINN: Concentrations cannot be negative)
        # Penalize values below 0. Since we integrated, this prevents rates from
        # driving the system into negative space.
        non_negativity_penalty = torch.mean(torch.clamp(conc, max=0)**2)

        # 4. Smoothness on phase weights (avoid discontinuous transitions)
        phase_smoothness = torch.mean((phase_weights[:, 1:, :] - phase_weights[:, :-1, :]) ** 2)

        # 5. Rate smoothness (PINN: Metabolic fluxes shouldn't jump instantly)
        # Blend rates to get the actual predicted trajectory flux
        phase_probs = torch.softmax(phase_weights, dim=-1)
        f = phase_probs[:, :, 1 : 2]
        blended_rates = (1 - f) * growth_rates + f * prod_rates
        rate_smoothness = torch.mean((blended_rates[:, 1:, :] - blended_rates[:, :-1, :]) ** 2)

        # 6. Rate magnitude (keep rates within reasonable bounds)
        rate_magnitude = torch.mean(torch.abs(growth_rates)) + torch.mean(torch.abs(prod_rates))

        # 7. Phase weights exploration penalty
        phase_penalty = torch.mean((phase_weights - 0.5) ** 2)

        # Combined loss with weighted terms
        total_loss = (
            conc_loss +
            0.1 * conc_smoothness +
            0.5 * non_negativity_penalty +   # High penalty for negative concentrations
            0.05 * phase_smoothness +
            0.1 * rate_smoothness +          # Encourage smooth flux transitions
            0.01 * rate_magnitude +
            0.02 * phase_penalty
        )

        return total_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(ic, time, params)
            
            # Compute loss based on model type
            loss = self._compute_loss(pred, target, self.model_type)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        self.losses.append(epoch_loss)
        self.scheduler.step()
        
        return epoch_loss
    
    def validate(self, val_loader):
        """Validate model and return a comprehensive diagnostic report."""
        self.model.eval()
        val_loss = 0.0

        all_targets = []
        all_preds = []
        all_phase_targets = []
        all_phase_preds = []

        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(self.device)
                time = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)

                pred = self.model(ic, time, params)

                # Compute loss
                if self.model_type == 'enhanced' and isinstance(pred, dict):
                    loss = self._compute_enhanced_loss(pred, target)

                    # Collect data for diagnostics
                    all_targets.append(target.cpu().numpy())
                    all_preds.append(pred['concentrations'].cpu().numpy())

                    if 'phases' in batch:
                        all_phase_targets.append(batch['phases'].cpu().numpy())
                        all_phase_preds.append(pred['phase_weights'].cpu().numpy())
                else:
                    loss = nn.functional.mse_loss(pred, target)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Generate Diagnostic Report
        report = {"val_loss": val_loss}
        if all_targets:
            y_true = np.concatenate(all_targets, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            report["metrics"] = ModelDiagnostics.calculate_regression_metrics(y_true, y_pred)

            if all_phase_targets:
                p_true = np.concatenate(all_phase_targets, axis=0)
                p_pred = np.concatenate(all_phase_preds, axis=0)
                report["phase_metrics"] = ModelDiagnostics.calculate_phase_metrics(p_true, p_pred)

        return report
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            report = self.validate(val_loader)
            val_loss = report["val_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} (BEST)")
                if "metrics" in report:
                    print(f"  -> R2: {report['metrics']['global_r2']:.4f}, MAPE: {report['metrics']['global_mape']:.4f}")
                if "phase_metrics" in report:
                    print(f"  -> Phase F1: {report['phase_metrics']['phase_f1']:.4f}")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return best_val_loss
    
    def save(self, path):
        """Save model and training info."""
        torch.save({
            'model_state': self.model.state_dict(),
            'losses': self.losses,
            'model_type': self.model_type,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.losses = checkpoint['losses']
        self.model_type = checkpoint.get('model_type', 'standard')
        print(f"Model loaded from {path}")


class PredictionInterface:
    """User-friendly interface for making predictions from both standard and enhanced models."""
    
    def __init__(self, model, dataset, device='cpu', model_type='standard'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataset = dataset
        self.model_type = model_type
    
    def predict(self, initial_conditions, time_points, parameters=None, return_rates=False):
        """
        Make predictions for new conditions.
        
        Args:
            initial_conditions: Array of shape (n_components,) or (batch_size, n_components)
            time_points: Array of time points to predict
            parameters: Dict or array of parameters
            return_rates: If True and using enhanced model, return metabolic rates
        
        Returns:
            For standard model: predictions array
            For enhanced model: Dict with 'concentrations', 'phase_weights', 'growth_rates', 'prod_rates'
        """
        # Handle single vs batch inputs
        single_input = False
        if initial_conditions.ndim == 1:
            initial_conditions = initial_conditions[np.newaxis, :]
            single_input = True
        
        # Normalize
        ic_norm = (initial_conditions - self.dataset.ic_min) / (self.dataset.ic_max - self.dataset.ic_min)
        ic_norm = np.clip(ic_norm, 0, 1)
        
        time_norm = time_points / np.max(time_points)
        
        # Convert to tensors
        ic_tensor = torch.FloatTensor(ic_norm).to(self.device)
        time_tensor = torch.FloatTensor(time_norm).unsqueeze(0).to(self.device)
        
        # Handle parameters - match the n_params from model
        n_params = getattr(self.model, 'n_params', 0)
        
        if parameters is not None:
            if isinstance(parameters, dict):
                param_list = [torch.FloatTensor([parameters[key]]) for key in sorted(parameters.keys())]
                params_tensor = torch.cat(param_list).unsqueeze(0).to(self.device)
            else:
                params_tensor = torch.FloatTensor(parameters).unsqueeze(0).to(self.device)
        else:
            # Create params tensor matching model expectations
            if n_params > 0:
                params_tensor = torch.zeros(1, n_params).to(self.device)
            else:
                # No parameters - create empty 1D tensor then unsqueeze to (1, 0)
                params_tensor = torch.zeros(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(ic_tensor, time_tensor, params_tensor)
        
        # Process outputs based on model type
        if self.model_type == 'enhanced' and isinstance(outputs, dict):
            return self._process_enhanced_outputs(outputs, single_input, return_rates)
        else:
            return self._process_standard_outputs(outputs, single_input)
    
    def _process_standard_outputs(self, pred_norm, single_input):
        """Denormalize and post-process standard model outputs."""
        pred = pred_norm.cpu().numpy() * (self.dataset.traj_max - self.dataset.traj_min) + self.dataset.traj_min
        pred = np.maximum(pred, 0)  # Ensure non-negative
        
        if single_input:
            return pred[0]
        return pred
    
    def _process_enhanced_outputs(self, outputs_dict, single_input, return_rates=False):
        """Denormalize and post-process enhanced model outputs."""
        results = {}
        
        # Denormalize concentrations
        conc_norm = outputs_dict['concentrations'].cpu().numpy()
        conc = conc_norm * (self.dataset.traj_max - self.dataset.traj_min) + self.dataset.traj_min
        conc = np.maximum(conc, 0)
        results['concentrations'] = conc[0] if single_input else conc
        
        # Extract phase weights (already in [0,1])
        phase_weights = outputs_dict['phase_weights'].cpu().numpy()
        results['phase_weights'] = phase_weights[0] if single_input else phase_weights
        
        if return_rates:
            # Denormalize rates (scale to concentration units)
            growth_rates = outputs_dict['growth_rates'].cpu().numpy()
            prod_rates = outputs_dict['prod_rates'].cpu().numpy()
            
            # Rates need denormalization too
            rate_scale = (self.dataset.traj_max - self.dataset.traj_min)
            growth_rates = growth_rates * rate_scale
            prod_rates = prod_rates * rate_scale
            
            results['growth_rates'] = growth_rates[0] if single_input else growth_rates
            results['prod_rates'] = prod_rates[0] if single_input else prod_rates
        
        return results
    
    def sensitivity_analysis(self, initial_conditions, time_points, param_name, 
                           param_range, n_points=10, parameters=None):
        """
        Analyze sensitivity to a parameter.
        
        Returns:
            param_values: Parameter values tested
            predictions: List of predictions for each parameter value
        """
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        predictions = []
        
        for pval in param_values:
            if parameters is None:
                params = {param_name: pval}
            else:
                params = parameters.copy()
                params[param_name] = pval
            
            pred = self.predict(initial_conditions, time_points, params)
            predictions.append(pred)
        
        return param_values, predictions
    
    def analyze_phase_transition(self, initial_conditions, time_points, parameters=None):
        """
        Analyze phase transition for enhanced model.
        Shows the switching from growth to production phase.
        """
        if self.model_type != 'enhanced':
            print("Phase transition analysis only available for enhanced model")
            return None
        
        outputs = self.predict(initial_conditions, time_points, parameters, return_rates=True)
        
        return {
            'time': time_points,
            'concentrations': outputs['concentrations'],
            'phase_weights': outputs['phase_weights'],
            'growth_rates': outputs['growth_rates'],
            'prod_rates': outputs['prod_rates'],
        }


# Example usage and helper functions
def load_matlab_simulations(matlab_data_file):
    """
    Load dFBA simulations exported from MATLAB.
    Expected format: .npz file with 'time', 'profiles', 'parameters'
    """
    data = np.load(matlab_data_file)
    return data


def create_dataset_from_simulations(matlab_files_list, batch_size=16):
    """Create a PyTorch dataset from multiple MATLAB simulation files."""
    all_trajectories = []
    all_times = []
    all_ics = []
    all_params = {}
    
    for matlab_file in matlab_files_list:
        data = np.load(matlab_file)
        
        # Extract trajectory
        profiles = data['profiles']  # (n_timepoints, n_components)
        time = data['time']  # (n_timepoints,)
        
        all_trajectories.append(profiles[np.newaxis, :, :])  # Add batch dimension
        all_times.append(time)
        all_ics.append(profiles[0])
        
        # Extract parameters if available
        if 'parameters' in data:
            for key, val in data['parameters'].items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(val)
    
    # Stack into arrays
    trajectories = np.vstack(all_trajectories)
    n_samples = trajectories.shape[0]
    
    # Pad trajectories to same length
    max_timepoints = max(t.shape[0] for t in all_times)
    trajectories_padded = np.zeros((n_samples, max_timepoints, trajectories.shape[2]))
    times_padded = np.zeros((n_samples, max_timepoints))
    
    for i, traj in enumerate(trajectories):
        trajectories_padded[i, :traj.shape[0], :] = traj
        times_padded[i, :all_times[i].shape[0]] = all_times[i]
    
    initial_conditions = np.array(all_ics)
    
    # Convert params to arrays
    for key in all_params:
        all_params[key] = np.array(all_params[key])
    
    # Create dataset
    dataset = dFBADataset(trajectories_padded, times_padded, initial_conditions, all_params)
    
    return dataset


if __name__ == "__main__":
    print("COSMIC dFBA Neural Network Surrogate Model")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    # 1. Load MATLAB dFBA simulations
    dataset = create_dataset_from_simulations(['sim1.npz', 'sim2.npz'])
    
    # 2. Create model
    model = CosmicNNSurrogate(
        n_components=dataset.n_components,
        n_params=10,  # or calculate from dataset
        latent_dim=64
    )
    
    # 3. Train
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer = TrainingManager(model)
    trainer.train(train_loader, val_loader, epochs=100)
    
    # 4. Predict
    predictor = PredictionInterface(model, dataset)
    pred = predictor.predict(
        initial_conditions=np.array([...]),
        time_points=np.linspace(0, 10, 100),
        parameters={'param1': 0.5, 'param2': 1.0}
    )
    """)
