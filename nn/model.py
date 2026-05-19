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
        return torch.cat([-w, w], dim=-1)


class RatePredictionHead(nn.Module):
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.growth_rates = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_components), nn.Tanh())
        self.prod_rates = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_components), nn.Tanh())

    def forward(self, latent_state, time_points):
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, time_points.shape[1], -1)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
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
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.integrator = DifferentiableIntegrator()
        self.state_weighting = StateWeightingLayer(n_components, latent_dim)
        self.rate_predictor = RatePredictionHead(n_components, latent_dim, n_heads)

    def forward(self, latent_state, time_points, initial_conditions):
        time_expanded = time_points.unsqueeze(-1)
        time_embedded = self.time_embed(time_expanded)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, time_points.shape[1], -1)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)

        growth_rates, prod_rates = self.rate_predictor(latent_state, time_points)

        f_initial = torch.zeros(latent_state.shape[0], time_points.shape[1], 1, device=latent_state.device)
        blended_rates_initial = (1 - f_initial) * growth_rates + f_initial * prod_rates
        concentrations = self.integrator(initial_conditions, blended_rates_initial, time_points)

        phase_logits = self.state_weighting(latent_state, concentrations)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        f_final = phase_probs[:, :, 1 : 2]
        final_blended_rates = (1 - f_final) * growth_rates + f_final * prod_rates
        final_concentrations = self.integrator(initial_conditions, final_blended_rates, time_points)

        return {
            'concentrations': final_concentrations,
            'phase_weights': phase_logits,
            'growth_rates': growth_rates,
            'prod_rates': prod_rates,
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
