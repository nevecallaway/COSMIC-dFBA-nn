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
    time_scale = torch.stack([item['time_scale'] for item in batch])  # (B, 1)

    params_list = [item['parameters'] for item in batch]
    if params_list[0].shape[0] == 0:
        params = torch.zeros(len(batch), 0)
    else:
        params = torch.stack(params_list)

    result = {
        'initial_conditions': ic,
        'time': time,
        'time_scale': time_scale,
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
        time_scale = float(np.max(self.time_points[idx]))
        time = torch.FloatTensor(self.time_points[idx] / time_scale)

        param_list = []
        for key, val in self.parameters.items():
            if val is not None:
                param_list.append(torch.FloatTensor([val[idx]]))

        params = torch.cat(param_list) if param_list else torch.zeros(0)

        item = {
            'initial_conditions': ic,
            'time': time,
            'time_scale': torch.FloatTensor([time_scale]),  # max day for denorm
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
        w = trigger + modulation                                  # (batch, time, 1)
        # Cumulative softmax: enforces p_m >= p_{m-1} (monotone) and p_M = 1
        # Matches paper constraints Eq 4-6. The softmax weights are a learned
        # probability density over time; the CDF is the phase fraction.
        weights = torch.softmax(w.squeeze(-1), dim=1)             # (batch, time)
        return torch.cumsum(weights, dim=1).unsqueeze(-1)         # (batch, time, 1)


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
    """
    Implicit-Euler integrator implementing COSMIC-dFBA Supplementary Eq 2 in full:

        cells (cols 0-1):   dC_i/dt = v_i * C_i
        metabolites (≥ 2):  dC_i/dt = F*(C_i_in - C_i) + v_i * C_1

    where F = 1 d⁻¹ (perfusion rate).  Time is normalised to [0, 1] over a
    13-day run, so F_norm = 13 in normalised units.

    C_i_in (perfusion feed concentrations) are derived from the DoE coded levels
    (-1, 0, +1) via (level + 1) / 2  →  {0, 0.5, 1.0}:
      • col 2  (Glucose)          ← DoE Glc  (doe_params[:, 2])
      • cols 6-24 (amino acids)   ← DoE AAs  (doe_params[:, 1])
      • all other metabolites     ← C_i_in = 0  (not in perfusion feed)
      • cells                     ← ε = 0, no washout term

    Implicit Euler is used for the washout so the scheme is unconditionally
    stable regardless of F·dt:
        c_next = (c_prev + (v·coupling + F·C_in·ε)·dt) / (1 + F·ε·dt)
    For cells ε = 0, denominator = 1 → reduces to explicit Euler.
    """
    N_CELL_COLS   = 2   # Cell Density (0), Cell Volume (1)
    IDX_GLUCOSE   = 2   # Glucose
    IDX_AAS_START = 6   # Glutamine … Tryptophan (19 components)
    F_NORM        = 13.0  # F=1 d⁻¹ × 13 days (normalised-time perfusion rate)

    def forward(self, initial_conditions, blended_rates, time_points, doe_params=None):
        B, T, C = blended_rates.shape
        device  = blended_rates.device

        # ε: removal fraction — 0 for cells, 1 for metabolites
        eps = torch.ones(C, device=device)
        eps[:self.N_CELL_COLS] = 0.0

        # C_in: perfusion feed concentrations built from DoE coded levels
        c_in = torch.zeros(B, C, device=device)
        if doe_params is not None and doe_params.shape[1] >= 3:
            glc = (doe_params[:, 2:3] + 1.0) / 2.0          # (B,1) → {0, 0.5, 1}
            aas = (doe_params[:, 1:2] + 1.0) / 2.0          # (B,1)
            c_in[:, self.IDX_GLUCOSE : self.IDX_GLUCOSE + 1]  = glc
            n_aa = C - self.IDX_AAS_START
            if n_aa > 0:
                c_in[:, self.IDX_AAS_START:]                 = aas.expand(-1, n_aa)

        concentrations = [initial_conditions]

        for t in range(1, T):
            c_prev = concentrations[t - 1]
            v      = blended_rates[:, t - 1, :]
            dt     = (time_points[:, t] - time_points[:, t - 1]).unsqueeze(-1)  # (B,1)

            c1 = c_prev[:, 0:1]
            coupling = torch.cat([
                c_prev[:, :self.N_CELL_COLS],
                c1.expand(-1, C - self.N_CELL_COLS),
            ], dim=-1)

            # Implicit Euler: stable for any F·dt
            numerator   = c_prev + (v * coupling + self.F_NORM * c_in * eps) * dt
            denominator = 1.0 + self.F_NORM * eps * dt
            concentrations.append(numerator / denominator)

        return torch.stack(concentrations, dim=1)      # (B, T, C)


class MultiHeadTemporalDecoder(nn.Module):
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        # n_heads kept for API compat; attention removed (was computing dead output)
        self.integrator = DifferentiableIntegrator()
        self.state_weighting = StateWeightingLayer(n_components, latent_dim)
        self.rate_predictor = RatePredictionHead(n_components, latent_dim, n_heads)

    def forward(self, latent_state, time_points, initial_conditions,
                doe_params=None, data3_growth=None, data3_prod=None):
        T = time_points.shape[1]
        if data3_growth is not None and data3_prod is not None:
            # Use measured specific rates from data_3 directly (paper's vg / v_stat)
            growth_rates = data3_growth.unsqueeze(1).expand(-1, T, -1)
            prod_rates   = data3_prod.unsqueeze(1).expand(-1, T, -1)
        else:
            growth_rates, prod_rates = self.rate_predictor(latent_state, time_points)

        f_initial = torch.zeros(latent_state.shape[0], T, 1, device=latent_state.device)
        blended_rates_initial = (1 - f_initial) * growth_rates + f_initial * prod_rates
        concentrations = self.integrator(initial_conditions, blended_rates_initial, time_points, doe_params)

        phase_pred = self.state_weighting(latent_state, concentrations)
        final_blended_rates = (1 - phase_pred) * growth_rates + phase_pred * prod_rates
        final_concentrations = self.integrator(initial_conditions, final_blended_rates, time_points, doe_params)

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

    def forward(self, latent_state, time_points, initial_conditions,
                doe_params=None, data3_growth=None, data3_prod=None):
        B, T = time_points.shape

        if data3_growth is not None and data3_prod is not None:
            # Use measured specific rates from data_3 directly (paper's vg / v_stat)
            growth_rates = data3_growth.unsqueeze(1).expand(-1, T, -1)
            prod_rates   = data3_prod.unsqueeze(1).expand(-1, T, -1)
        else:
            time_embedded   = self.time_embed(time_points.unsqueeze(-1))
            h0 = self.h0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
            c0 = self.c0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
            lstm_out, _     = self.lstm(time_embedded, (h0, c0))
            latent_expanded = latent_state.unsqueeze(1).expand(-1, T, -1)
            combined        = torch.cat([lstm_out, latent_expanded], dim=-1)
            growth_rates    = self.growth_rates(combined)
            prod_rates      = self.prod_rates(combined)

        f_init = torch.zeros(B, T, 1, device=latent_state.device)
        concentrations = self.integrator(
            initial_conditions, (1 - f_init) * growth_rates + f_init * prod_rates,
            time_points, doe_params)

        phase_pred = self.state_weighting(latent_state, concentrations)
        final_concentrations = self.integrator(
            initial_conditions,
            (1 - phase_pred) * growth_rates + phase_pred * prod_rates,
            time_points, doe_params)

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


def _extract_doe_and_rates(parameters):
    """
    Extract DoE and data_3 specific rates from the parameters tensor.

    Parameter layout (set by train.py load_data):
      [0:3]   DoE coded levels  [O2, AAs, Glc]
      [3:28]  growth-phase specific rates  (25 components)
      [28:53] production-phase specific rates  (25 components)
      [53:]   FBA objective efficiencies  (22 values, if present)

    Returns (doe, growth_rates, prod_rates) — any may be None if the
    parameters tensor is too short (e.g. NPZ-only runs without CSV data).
    """
    n = parameters.shape[1]
    doe    = parameters[:, :3]       if n >= 3  else None
    growth = parameters[:, 3:28]     if n >= 28 else None
    prod   = parameters[:, 28:53]    if n >= 53 else None
    return doe, growth, prod


class CosmicNNSurrogateEnhanced(nn.Module):
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = MultiHeadTemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        latent = self.encoder(initial_conditions, parameters)
        doe, growth, prod = _extract_doe_and_rates(parameters)
        return self.decoder(latent, time_points, initial_conditions,
                            doe_params=doe, data3_growth=growth, data3_prod=prod)


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
        doe, growth, prod = _extract_doe_and_rates(parameters)
        return self.decoder(latent, time_points, initial_conditions,
                            doe_params=doe, data3_growth=growth, data3_prod=prod)
