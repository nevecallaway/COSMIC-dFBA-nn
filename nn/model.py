#!/usr/bin/env python3
"""
COSMIC dFBA Surrogate Model Architecture.
Provides the NN definitions and the corresponding PyTorch Dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, n_components, n_params, latent_dim=32):
        super().__init__()
        input_size = n_components + n_params
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, initial_conditions, parameters):
        x = torch.cat([initial_conditions, parameters], dim=-1)
        return self.fc(x)


class PhaseTransitionHead(nn.Module):
    """
    Phase probability from current concentrations -- no time input.

    f_t = sigmoid(W · c_t)

    The cell's metabolic state drives the phase, not the clock.
    A single linear layer keeps the head minimal and interpretable:
    each weight tells you how much one component pushes toward production.
    """
    def __init__(self, n_components):
        super().__init__()
        self.linear = nn.Linear(n_components, 1)

    def forward(self, concentrations):
        return torch.sigmoid(self.linear(concentrations))   # (B, 1)




class DifferentiableIntegrator(nn.Module):
    """
    Implicit-Euler integrator implementing COSMIC-dFBA Supplementary Eq 2:

        dC_i/dt = F * (C_i_in - eta * C_i) + v_{i,m} * C_1

    where F = 1 d⁻¹ (perfusion rate), C_1 is cell density, and eta is the
    removal fraction per the paper:
      - eta = 0 for cell density and cell volume at all times
      - eta = 0 for titer before day 8, eta = 1 for titer from day 8 onward
      - eta = 1 for all other metabolites at all times

    Time is normalised to [0, 1] over a 13-day run, so F_NORM = 13.
    DAY8_NORM = 8/13: normalised time at which titer washout activates.

    C_i_in (perfusion feed concentrations) are derived from the DoE coded levels
    (-1, 0, +1) via (level + 1) / 2  →  {0, 0.5, 1.0}:
      • col 2  (Glucose)          ← DoE Glc  (doe_params[:, 2])
      • cols 6-24 (amino acids)   ← DoE AAs  (doe_params[:, 1])
      • all other metabolites     ← C_i_in = 0  (not in perfusion feed)

    Explicit Euler (forward difference), matching COSMIC-dFBA:
        dc/dt  = v · coupling + F · (C_in - c_prev) · eta
        c_next = c_prev + dc/dt · dt
    For cells and titer-before-day8 eta = 0, washout term drops out.
    """
    N_CELL_COLS   = 2              # Cell Density (0), Cell Volume (1)
    IDX_GLUCOSE   = 2              # Glucose
    IDX_TITER     = 5              # Antibody titer
    IDX_AAS_START = 6              # Glutamine ... Tryptophan (19 components)
    F_NORM        = 13.0           # F=1 d⁻¹ × 13 days (normalised-time perfusion rate)
    F_TITER       = 1.0            # Effective washout rate for titer (physical units, not
                                   # normalised): keeps denominator ~1.08/step so learned
                                   # production rates can compensate the washout.
    DAY8_NORM     = 8.0 / 13.0    # titer washout activates at day 8

    def forward(self, initial_conditions, blended_rates, time_points, doe_params=None):
        B, T, C = blended_rates.shape
        device  = blended_rates.device

        # eta_base: (C,) -- 0 for cells, 0 for titer (activated after day 8), 1 elsewhere.
        eta_base = torch.ones(C, device=device)
        eta_base[:self.N_CELL_COLS] = 0.0
        if C > self.IDX_TITER:
            eta_base[self.IDX_TITER] = 0.0

        # One-hot for titer column: used to add washout contribution after day 8.
        titer_onehot = torch.zeros(C, device=device)
        if C > self.IDX_TITER:
            titer_onehot[self.IDX_TITER] = 1.0

        # Per-column F: F_NORM for all metabolites, F_TITER for titer.
        # c_in for titer is 0 (not in feed), so f_vec only affects the denominator.
        f_vec = torch.full((C,), self.F_NORM, device=device)
        if C > self.IDX_TITER:
            f_vec[self.IDX_TITER] = self.F_TITER

        # Titer production rate is always ≥ 0 (cells can't un-produce antibody).
        # Use torch.cat to avoid in-place ops: the pattern clone()[...] = x.clamp()
        # saves the LHS view for backward then immediately invalidates it in-place.
        if C > self.IDX_TITER:
            blended_rates = torch.cat([
                blended_rates[:, :, :self.IDX_TITER],
                blended_rates[:, :, self.IDX_TITER:self.IDX_TITER + 1].clamp(min=0),
                blended_rates[:, :, self.IDX_TITER + 1:],
            ], dim=-1)

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
            concentrations.append(
                self.step(concentrations[t-1], blended_rates[:, t-1, :],
                          time_points[:, t-1], time_points[:, t],
                          doe_params, eta_base, titer_onehot, f_vec, c_in))
        return torch.stack(concentrations, dim=1)      # (B, T, C)

    def step(self, c_prev, v, t_prev, t_curr,
             doe_params=None, eta_base=None, titer_onehot=None, f_vec=None, c_in=None):
        """One explicit-Euler step.

        Can be called standalone (from the model's sequential forward loop) or
        from DifferentiableIntegrator.forward() with pre-computed constants.
        When called standalone, all optional tensor args are recomputed from scratch.
        """
        B, C   = c_prev.shape
        device = c_prev.device
        dt     = (t_curr - t_prev).unsqueeze(-1)   # (B, 1)

        if eta_base is None:
            eta_base = torch.ones(C, device=device)
            eta_base[:self.N_CELL_COLS] = 0.0
            if C > self.IDX_TITER:
                eta_base[self.IDX_TITER] = 0.0

        if titer_onehot is None:
            titer_onehot = torch.zeros(C, device=device)
            if C > self.IDX_TITER:
                titer_onehot[self.IDX_TITER] = 1.0

        if f_vec is None:
            f_vec = torch.full((C,), self.F_NORM, device=device)
            if C > self.IDX_TITER:
                f_vec[self.IDX_TITER] = self.F_TITER

        if c_in is None:
            c_in = torch.zeros(B, C, device=device)
            if doe_params is not None and doe_params.shape[1] >= 3:
                glc = (doe_params[:, 2:3] + 1.0) / 2.0
                aas = (doe_params[:, 1:2] + 1.0) / 2.0
                c_in[:, self.IDX_GLUCOSE:self.IDX_GLUCOSE + 1] = glc
                n_aa = C - self.IDX_AAS_START
                if n_aa > 0:
                    c_in[:, self.IDX_AAS_START:] = aas.expand(-1, n_aa)

        titer_on = (t_curr >= self.DAY8_NORM).float().unsqueeze(-1)   # (B, 1)
        eta_t    = eta_base + titer_on * titer_onehot

        c1       = c_prev[:, 0:1]
        coupling = torch.cat([c_prev[:, :self.N_CELL_COLS],
                               c1.expand(-1, C - self.N_CELL_COLS)], dim=-1)
        dc_dt    = v * coupling + f_vec * (c_in - c_prev) * eta_t
        return c_prev + dc_dt * dt


class CosmicNNSurrogate(nn.Module):
    """
    Minimal surrogate for COSMIC-dFBA.

    Encoder compresses (IC + DoE) into a reactor-specific latent vector.
    Two constant rate heads predict per-reactor growth and production rates
    (biologically appropriate -- rates are phase-constant, not time-varying).
    A phase head predicts the transition sigmoid f(t) directly from the latent.
    A single ODE integration blends the rates via f(t) to produce trajectories.
    """
    def __init__(self, n_components, n_params, latent_dim=32):
        super().__init__()
        self.encoder     = DynamicsEncoder(n_components, n_params, latent_dim)
        # Rate heads: Tanh gives signed direction and fractional utilisation in (-1, 1).
        # Scaled by v_max buffers (set from data_3) so rates are bounded by the
        # phase-specific biological maxima.  Default = ones until set_v_max() is called.
        self.growth_head = nn.Sequential(nn.Linear(latent_dim, n_components), nn.Tanh())
        self.prod_head   = nn.Sequential(nn.Linear(latent_dim, n_components), nn.Tanh())
        self.register_buffer('v_max_growth', torch.ones(n_components))
        self.register_buffer('v_max_prod',   torch.ones(n_components))
        self.phase_head   = PhaseTransitionHead(n_components)
        self.integrator   = DifferentiableIntegrator()
        self.n_components = n_components
        self.n_params     = n_params

    def set_v_max(self, v_max_growth: torch.Tensor, v_max_prod: torch.Tensor) -> None:
        """Set per-component rate ceilings from data_3 (normalised)."""
        self.v_max_growth.copy_(v_max_growth)
        self.v_max_prod.copy_(v_max_prod)

    def forward(self, initial_conditions, time_points, parameters):
        B, T = time_points.shape
        latent = self.encoder(initial_conditions, parameters)

        growth = self.growth_head(latent) * self.v_max_growth   # (B, C)
        prod   = self.prod_head(latent)   * self.v_max_prod     # (B, C)

        # Clamp titer production rate >= 0 (cells can't un-produce antibody).
        if self.n_components > self.integrator.IDX_TITER:
            idx = self.integrator.IDX_TITER
            prod = torch.cat([prod[:, :idx],
                              prod[:, idx:idx+1].clamp(min=0),
                              prod[:, idx+1:]], dim=-1)

        # Sequential: phase from concentrations, one integration step, repeat.
        c    = initial_conditions
        all_c = [c]
        all_f = []
        for t in range(T):
            f_t = self.phase_head(c)                               # (B, 1)
            all_f.append(f_t)
            if t < T - 1:
                v = (1 - f_t) * growth + f_t * prod               # (B, C)
                c = self.integrator.step(c, v,
                                         time_points[:, t], time_points[:, t + 1],
                                         parameters)
                all_c.append(c)

        concentrations = torch.stack(all_c, dim=1)   # (B, T, C)
        phase_weights  = torch.stack(all_f, dim=1)   # (B, T, 1)

        return {
            'concentrations': concentrations,   # (B, T, C)
            'phase_weights':  phase_weights,    # (B, T, 1)
            'growth_rates':   growth,           # (B, C)
            'prod_rates':     prod,             # (B, C)
        }


class CosmicNNSurrogateLSTM(nn.Module):
    """
    LSTM variant of CosmicNNSurrogate for comparison.

    Identical encoder, phase head, and integrator. The only difference:
    rate heads use a single-layer LSTM initialized from the latent state,
    producing time-varying rates (B, T, C) instead of constant rates.

    The LSTM processes the normalized time sequence as input, so it can
    learn non-constant within-phase dynamics if they exist in the data.
    """
    def __init__(self, n_components, n_params, latent_dim=32):
        super().__init__()
        self.encoder     = DynamicsEncoder(n_components, n_params, latent_dim)
        self.h0_proj     = nn.Linear(latent_dim, latent_dim)
        self.c0_proj     = nn.Linear(latent_dim, latent_dim)
        self.lstm        = nn.LSTM(input_size=1, hidden_size=latent_dim,
                                   num_layers=1, batch_first=True)
        self.growth_head = nn.Sequential(nn.Linear(latent_dim, n_components), nn.Tanh())
        self.prod_head   = nn.Sequential(nn.Linear(latent_dim, n_components), nn.Tanh())
        self.register_buffer('v_max_growth', torch.ones(n_components))
        self.register_buffer('v_max_prod',   torch.ones(n_components))
        self.phase_head   = PhaseTransitionHead(n_components)
        self.integrator   = DifferentiableIntegrator()
        self.n_components = n_components
        self.n_params     = n_params

    def set_v_max(self, v_max_growth: torch.Tensor, v_max_prod: torch.Tensor) -> None:
        """Set per-component rate ceilings from data_3 (normalised)."""
        self.v_max_growth.copy_(v_max_growth)
        self.v_max_prod.copy_(v_max_prod)

    def forward(self, initial_conditions, time_points, parameters):
        B, T = time_points.shape
        latent = self.encoder(initial_conditions, parameters)

        h0 = self.h0_proj(latent).unsqueeze(0)   # (1, B, latent_dim)
        c0 = self.c0_proj(latent).unsqueeze(0)
        lstm_out, _ = self.lstm(time_points.unsqueeze(-1), (h0, c0))  # (B, T, latent_dim)

        growth_rates = self.growth_head(lstm_out) * self.v_max_growth   # (B, T, C)
        prod_rates   = self.prod_head(lstm_out)   * self.v_max_prod     # (B, T, C)

        # Sequential: phase from concentrations, one integration step, repeat.
        c     = initial_conditions
        all_c = [c]
        all_f = []
        for t in range(T):
            f_t = self.phase_head(c)                                   # (B, 1)
            all_f.append(f_t)
            if t < T - 1:
                v = (1 - f_t) * growth_rates[:, t, :] + f_t * prod_rates[:, t, :]
                c = self.integrator.step(c, v,
                                         time_points[:, t], time_points[:, t + 1],
                                         parameters)
                all_c.append(c)

        return {
            'concentrations': torch.stack(all_c, dim=1),   # (B, T, C)
            'phase_weights':  torch.stack(all_f, dim=1),   # (B, T, 1)
            'growth_rates':   growth_rates,                 # (B, T, C)
            'prod_rates':     prod_rates,                   # (B, T, C)
        }
