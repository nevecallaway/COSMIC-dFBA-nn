#!/usr/bin/env python3
"""
COSMIC dFBA Surrogate Model Architecture.
Provides the NN definitions and the corresponding PyTorch Dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

def dfba_collate_fn(batch):
    """
    Custom collate function to properly handle empty parameter tensors and phases.
    """
    ic = torch.stack([item['initial_conditions'] for item in batch])
    time = torch.stack([item['time'] for item in batch])
    traj = torch.stack([item['trajectory'] for item in batch])

    params_list = [item['parameters'] for item in batch]
    if params_list[0].shape[0] == 0:
        params = torch.zeros(len(batch), 0)
    else:
        params = torch.stack(params_list)

    result = {
        'initial_conditions': ic,
        'time': time,
        'parameters': params,
        'trajectory': traj
    }
    if 'phases' in batch[0]:
        result['phases'] = torch.stack([item['phases'] for item in batch])
    return result


class dFBADataset(Dataset):
    """
    Dataset for dFBA simulation trajectories.
    """
    def __init__(self, trajectories, time_points, initial_conditions,
                 parameters, normalize=True, phases=None):
        self.trajectories = trajectories
        self.time_points = time_points
        self.initial_conditions = initial_conditions
        self.parameters = parameters
        self.phases = phases

        self.n_samples = trajectories.shape[0]
        self.n_components = trajectories.shape[2]
        self.n_timepoints = trajectories.shape[1]
        self.n_params = len(parameters)  # number of process parameter scalars

        if normalize:
            self._normalize()

    def _normalize(self):
        # Step 1: robust percentile anchor (handles outliers without discarding them)
        self.traj_min = np.percentile(self.trajectories, 1, axis=(0, 1))
        self.traj_max = np.percentile(self.trajectories, 99, axis=(0, 1))
        self.traj_max = np.maximum(self.traj_max, self.traj_min + 1e-6)

        self.ic_min = np.percentile(self.initial_conditions, 1, axis=0)
        self.ic_max = np.percentile(self.initial_conditions, 99, axis=0)
        self.ic_max = np.maximum(self.ic_max, self.ic_min + 1e-6)

        traj_norm = (self.trajectories - self.traj_min) / (self.traj_max - self.traj_min)
        ic_norm   = (self.initial_conditions - self.ic_min) / (self.ic_max - self.ic_min)

        # Step 2: rescale to exactly [0,1] using the actual min/max of the
        # normalised values — no clipping, no pile-up at boundaries,
        # outliers land near 0 or 1 rather than being squashed onto them.
        self.traj_scale_min = traj_norm.min(axis=(0, 1))
        self.traj_scale_max = traj_norm.max(axis=(0, 1))
        self.traj_scale_max = np.maximum(self.traj_scale_max, self.traj_scale_min + 1e-6)

        self.ic_scale_min = ic_norm.min(axis=0)
        self.ic_scale_max = ic_norm.max(axis=0)
        self.ic_scale_max = np.maximum(self.ic_scale_max, self.ic_scale_min + 1e-6)

        self.trajectories = (traj_norm - self.traj_scale_min) / (self.traj_scale_max - self.traj_scale_min)
        self.initial_conditions = (ic_norm - self.ic_scale_min) / (self.ic_scale_max - self.ic_scale_min)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        traj = torch.FloatTensor(self.trajectories[idx])
        ic = torch.FloatTensor(self.initial_conditions[idx])
        time = torch.FloatTensor(self.time_points[idx] / np.max(self.time_points[idx]))

        param_list = []
        for key, val in self.parameters.items():
            if val is not None:
                param_list.append(torch.FloatTensor([val[idx]]))

        params = torch.cat(param_list) if param_list else torch.zeros(0)

        item = {
            'initial_conditions': ic,
            'time': time,
            'parameters': params,
            'trajectory': traj
        }
        if self.phases is not None:
            item['phases'] = torch.FloatTensor(self.phases[idx])
        return item


class DynamicsEncoder(nn.Module):
    def __init__(self, n_components, n_params, latent_dim=64):
        super().__init__()
        input_size = n_components + n_params
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, initial_conditions, parameters):
        x = torch.cat([initial_conditions, parameters], dim=-1)
        return self.fc(x)


class StateWeightingLayer(nn.Module):
    def __init__(self, n_components, latent_dim=64):
        super().__init__()
        self.trigger_weights = nn.Parameter(torch.randn(n_components, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, latent_state, concentrations):
        trigger = torch.matmul(concentrations, self.trigger_weights) + self.bias
        modulation = self.mlp(latent_state).unsqueeze(1)
        w = trigger + modulation
        return torch.sigmoid(w)   # (batch, time, 1) — continuous phase in [0, 1]


class RatePredictionHead(nn.Module):
    """
    Predicts growth and production rates at each time step.

    Previous implementation used attention(Q=time_embed, K=V=latent_expanded).
    Since all rows of latent_expanded are identical, the attention output was
    constant across time — making rates constant per reactor and giving the
    decoder a bypass route that ignored the latent code.

    Fixed: directly concatenate per-step time embedding with the latent so rates
    are genuinely time-varying AND reactor-specific.
    """
    TIME_DIM = 32

    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        # n_heads retained in signature for API compat but no longer used
        in_dim = latent_dim + self.TIME_DIM
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.TIME_DIM), nn.ReLU(),
            nn.Linear(self.TIME_DIM, self.TIME_DIM))
        self.growth_rates = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_components), nn.Tanh())
        self.prod_rates = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_components), nn.Tanh())

    def forward(self, latent_state, time_points):
        B, T = time_points.shape
        time_emb = self.time_embed(time_points.unsqueeze(-1))              # (B, T, TIME_DIM)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, T, -1)     # (B, T, latent_dim)
        combined = torch.cat([latent_expanded, time_emb], dim=-1)         # (B, T, latent_dim+TIME_DIM)
        return self.growth_rates(combined), self.prod_rates(combined)


class DifferentiableIntegrator(nn.Module):
    def forward(self, initial_conditions, blended_rates, time_points):
        batch_size, n_timepoints, n_components = blended_rates.shape
        if n_timepoints > 1:
            dt = torch.diff(time_points, dim=1).unsqueeze(-1)
            dt = torch.cat([torch.zeros(batch_size, 1, 1, device=dt.device), dt], dim=1)
        else:
            dt = torch.zeros(batch_size, 1, 1, device=blended_rates.device)
        integrand = blended_rates * dt
        return initial_conditions.unsqueeze(1) + torch.cumsum(integrand, dim=1)


class MultiHeadTemporalDecoder(nn.Module):
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        # n_heads kept for API compat; attention removed (was computing dead output)
        self.integrator = DifferentiableIntegrator()
        self.state_weighting = StateWeightingLayer(n_components, latent_dim)
        self.rate_predictor = RatePredictionHead(n_components, latent_dim, n_heads)

    def forward(self, latent_state, time_points, initial_conditions):
        growth_rates, prod_rates = self.rate_predictor(latent_state, time_points)

        f_initial = torch.zeros(latent_state.shape[0], time_points.shape[1], 1, device=latent_state.device)
        blended_rates_initial = (1 - f_initial) * growth_rates + f_initial * prod_rates
        concentrations = self.integrator(initial_conditions, blended_rates_initial, time_points)

        phase_pred = self.state_weighting(latent_state, concentrations)
        final_blended_rates = (1 - phase_pred) * growth_rates + phase_pred * prod_rates
        final_concentrations = self.integrator(initial_conditions, final_blended_rates, time_points)

        return {
            'concentrations': final_concentrations,
            'phase_weights': phase_pred,
            'growth_rates': growth_rates,
            'prod_rates': prod_rates,
        }


class LSTMTemporalDecoder(nn.Module):
    """
    LSTM-based temporal decoder. Replaces transformer attention with an LSTM
    that steps through time sequentially, which suits rise-then-fall trajectories
    better than attention (which treats all timepoints equally).

    The latent vector initialises the LSTM hidden and cell states.
    Time embeddings are the per-step inputs.
    Rate prediction heads and the differentiable integrator are kept identical
    to MultiHeadTemporalDecoder so the physics-informed structure is preserved.
    """
    def __init__(self, n_components, latent_dim=64, n_layers=2):
        super().__init__()
        self.time_embed  = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.h0_proj     = nn.Linear(latent_dim, latent_dim)
        self.c0_proj     = nn.Linear(latent_dim, latent_dim)
        self.lstm        = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                                   num_layers=n_layers, batch_first=True,
                                   dropout=0.2 if n_layers > 1 else 0.0)
        # Skip-connect the original latent so reactor identity is always
        # explicitly available at every timestep, not just via LSTM hidden state.
        in_dim = latent_dim * 2
        self.growth_rates = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_components), nn.Tanh())
        self.prod_rates   = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_components), nn.Tanh())
        self.state_weighting = StateWeightingLayer(n_components, latent_dim)
        self.integrator      = DifferentiableIntegrator()
        self.n_layers = n_layers

    def forward(self, latent_state, time_points, initial_conditions):
        B, T = time_points.shape
        time_embedded = self.time_embed(time_points.unsqueeze(-1))   # (B, T, latent_dim)

        # Initialise LSTM hidden/cell from latent vector
        h0 = self.h0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = self.c0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
        lstm_out, _ = self.lstm(time_embedded, (h0, c0))             # (B, T, latent_dim)

        latent_expanded = latent_state.unsqueeze(1).expand(-1, T, -1)         # (B, T, latent_dim)
        combined = torch.cat([lstm_out, latent_expanded], dim=-1)             # (B, T, 2*latent_dim)
        growth_rates = self.growth_rates(combined)   # (B, T, C)
        prod_rates   = self.prod_rates(combined)     # (B, T, C)

        # Two-pass phase blending (same as MultiHeadTemporalDecoder)
        f_init = torch.zeros(B, T, 1, device=latent_state.device)
        concentrations = self.integrator(
            initial_conditions, (1 - f_init) * growth_rates + f_init * prod_rates, time_points)

        phase_pred = self.state_weighting(latent_state, concentrations)
        final_concentrations = self.integrator(
            initial_conditions,
            (1 - phase_pred) * growth_rates + phase_pred * prod_rates,
            time_points)

        return {
            'concentrations': final_concentrations,
            'phase_weights':  phase_pred,
            'growth_rates':   growth_rates,
            'prod_rates':     prod_rates,
        }


class TemporalDecoder(nn.Module):
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_components), nn.Sigmoid())

    def forward(self, latent_state, time_points):
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, time_points.shape[1], -1)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        return self.decoder(combined)


class CosmicNNSurrogate(nn.Module):
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = TemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        latent = self.encoder(initial_conditions, parameters)
        return self.decoder(latent, time_points)


class SimpleBaseline(nn.Module):
    def __init__(self, n_components, n_params=0, latent_dim=64):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        input_size = n_components + n_params
        self.encoder = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, latent_dim))
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, 2, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_components), nn.Sigmoid())

    def forward(self, initial_conditions, time_points, parameters=None):
        encoder_input = torch.cat([initial_conditions, parameters], dim=-1) if parameters is not None and parameters.shape[1] > 0 else initial_conditions
        latent_state = self.encoder(encoder_input)
        time_embedded = self.time_embed(time_points.unsqueeze(-1))
        latent_expanded = latent_state.unsqueeze(1).expand(-1, time_points.shape[1], -1)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        concentrations = self.decoder(combined)
        return {
            'concentrations': concentrations,
            'phase_weights': torch.ones(initial_conditions.shape[0], time_points.shape[1], 1, device=initial_conditions.device) * 0.5
        }


class CosmicNNSurrogateEnhanced(nn.Module):
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = MultiHeadTemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        latent = self.encoder(initial_conditions, parameters)
        return self.decoder(latent, time_points, initial_conditions)


class CosmicNNSurrogateLSTM(nn.Module):
    """Same encoder as CosmicNNSurrogateEnhanced, LSTM decoder instead of attention."""
    def __init__(self, n_components, n_params, latent_dim=64, n_layers=2):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = LSTMTemporalDecoder(n_components, latent_dim, n_layers)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        latent = self.encoder(initial_conditions, parameters)
        return self.decoder(latent, time_points, initial_conditions)
