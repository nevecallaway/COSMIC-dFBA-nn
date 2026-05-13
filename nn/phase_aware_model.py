#!/usr/bin/env python3
"""
Phase-Aware Architecture: Concentrations blended from phase-conditional predictions.
Phase output directly modulates the concentration trajectory.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class PhaseAwareDecoder(nn.Module):
    """
    Generates phase-conditional concentration trajectories.
    Computes separate trajectories for growth and production phases, 
    then blends them using predicted phase weights.
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
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        # GROWTH PHASE trajectory (uptake-focused)
        self.growth_trajectory = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Sigmoid()  # [0,1] - normalized
        )
        
        # PRODUCTION PHASE trajectory (product formation)
        self.prod_trajectory = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_components),
            nn.Sigmoid()  # [0,1] - normalized
        )
        
        # Phase prediction (growth weight)
        self.phase_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0,1] representing phase transition
        )
    
    def forward(self, latent_state, time_points):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            time_points: (batch_size, n_timepoints) normalized to [0,1]
        
        Returns:
            Dict with:
            - 'concentrations': (batch_size, n_timepoints, n_components) - blended
            - 'phase_weights': (batch_size, n_timepoints, 1) - phase transition
            - 'growth_conc': (batch_size, n_timepoints, n_components) - phase-specific
            - 'prod_conc': (batch_size, n_timepoints, n_components) - phase-specific
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
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        
        # Predict phase-conditional trajectories
        growth_conc = self.growth_trajectory(combined)  # (batch, time, n_comp)
        prod_conc = self.prod_trajectory(combined)      # (batch, time, n_comp)
        
        # Predict phase transition weights (one per timepoint)
        # Expand latent for phase prediction across time
        phase_logits = self.phase_predictor(latent_expanded)  # (batch, time, 1)
        
        # Blend trajectories based on phase
        # phase_weight = 0 → growth, phase_weight = 1 → production
        blended_conc = growth_conc * (1 - phase_logits) + prod_conc * phase_logits
        
        return {
            'concentrations': blended_conc,
            'phase_weights': phase_logits,
            'growth_conc': growth_conc,
            'prod_conc': prod_conc,
        }


class CosmicNNSurrogatePhaseAware(nn.Module):
    """
    Phase-aware neural network surrogate for COSMIC-dFBA.
    Core innovation: Concentrations are blended from phase-conditional predictions.
    """
    def __init__(self, n_components, n_params=0, latent_dim=64, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        
        # Encoder: (IC + params) → latent
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
        
        # Phase-aware decoder
        self.decoder = PhaseAwareDecoder(n_components, latent_dim, n_heads)
    
    def forward(self, initial_conditions, time_points, parameters=None):
        """
        Args:
            initial_conditions: (batch_size, n_components)
            time_points: (batch_size, n_timepoints)
            parameters: (batch_size, n_params) or None
        
        Returns:
            Dict with concentration predictions and phase weights
        """
        batch_size = initial_conditions.shape[0]
        device = initial_conditions.device
        
        # Encode initial conditions + parameters
        if parameters is not None and parameters.shape[1] > 0:
            encoder_input = torch.cat([initial_conditions, parameters], dim=-1)
        else:
            encoder_input = initial_conditions
        
        latent_state = self.encoder(encoder_input)
        
        # Decode with phase awareness
        outputs = self.decoder(latent_state, time_points)
        
        return outputs


if __name__ == "__main__":
    # Test
    batch_size, n_time, n_comp = 2, 13, 4
    n_params = 0
    
    model = CosmicNNSurrogatePhaseAware(n_components=n_comp, n_params=n_params, latent_dim=32, n_heads=2)
    
    ic = torch.randn(batch_size, n_comp)
    time = torch.rand(batch_size, n_time)
    params = torch.zeros(batch_size, 0) if n_params == 0 else torch.randn(batch_size, n_params)
    
    outputs = model(ic, time, params)
    
    print("Model outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    print(f"\nConcentration range: {outputs['concentrations'].min():.3f} - {outputs['concentrations'].max():.3f}")
    print(f"Phase weight range: {outputs['phase_weights'].min():.3f} - {outputs['phase_weights'].max():.3f}")
