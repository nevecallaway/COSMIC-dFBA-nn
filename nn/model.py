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


class PhaseTransitionHead(nn.Module):
    """
    Predicts f(t) = sigmoid((t - mu) / sigma) conditioned on the
    concentration trajectory and the latent state.

    The concentration trajectory (from a first-pass ODE integration with
    f=0) encodes the metabolic state that triggers the phase switch —
    e.g. glucose depletion drives cells from growth to production.
    A mean-pool over time summarises which nutrients depleted and how fast.
    This is combined with the latent state to predict:

      mu   in (0,1): normalised transition midpoint (mu * 13 = transition day)
      sigma > 0:     transition width — small = sharp, large = gradual

    f(t) = sigmoid((t - mu) / sigma) is monotone by construction and
    directly parameterises what transition MAE measures.
    """
    def __init__(self, n_components, latent_dim=64):
        super().__init__()
        self.conc_encoder = nn.Sequential(
            nn.Linear(n_components, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 16, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, latent_state, time_points, concentrations):
        # concentrations: (B, T, C) — mean-pool over time → (B, C)
        conc_summary = self.conc_encoder(concentrations.mean(dim=1))   # (B, 16)
        combined = torch.cat([latent_state, conc_summary], dim=-1)     # (B, latent+16)
        raw   = self.head(combined)                                     # (B, 2)
        mu    = torch.sigmoid(raw[:, 0])                               # (B,) in (0, 1)
        sigma = F.softplus(raw[:, 1]) + 0.01                          # (B,) > 0
        f = torch.sigmoid((time_points - mu.unsqueeze(1)) / sigma.unsqueeze(1))
        return f.unsqueeze(-1)                                         # (B, T, 1)


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

    def __init__(self, n_components, latent_dim=64):
        super().__init__()
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
        # Per-reactor amplitude: Tanh captures shape/direction,
        # amplitude scales magnitude from the latent state.
        # Softplus ensures positive, per-component scaling.
        self.amplitude = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, n_components), nn.Softplus())

    def forward(self, latent_state, time_points):
        B, T = time_points.shape
        time_emb = self.time_embed(time_points.unsqueeze(-1))              # (B, T, TIME_DIM)
        latent_expanded = latent_state.unsqueeze(1).expand(-1, T, -1)     # (B, T, latent_dim)
        combined = torch.cat([latent_expanded, time_emb], dim=-1)         # (B, T, latent_dim+TIME_DIM)
        amp = self.amplitude(latent_state).unsqueeze(1)                    # (B, 1, C)
        return self.growth_rates(combined) * amp, self.prod_rates(combined) * amp


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
    N_CELL_COLS   = 2         # Cell Density (0), Cell Volume (1)
    IDX_GLUCOSE   = 2         # Glucose
    IDX_TITER     = 5         # Antibody titer
    IDX_AAS_START = 6         # Glutamine … Tryptophan (19 components)
    F_NORM        = 13.0      # F=1 d⁻¹ × 13 days (normalised-time perfusion rate)
    DAY8_NORM     = 8.0/13.0  # Day 8 in normalised [0,1] time
    # Before day 8: titer retained in reactor (ATF membrane, ε=0).
    # After day 8: hypothermic shift + harvest begins, titer subject to washout (ε=1).
    # Paper Supplementary Eq 2: "ε=0 for antibody before day 8, ε=1 otherwise."

    def forward(self, initial_conditions, blended_rates, time_points, doe_params=None):
        B, T, C = blended_rates.shape
        device  = blended_rates.device

        # ε: removal fraction — 0 for cells, 1 for metabolites.
        # Titer: ε=0 before day 8 (retained by ATF membrane), ε=1 after day 8
        # (washout begins with hypothermic shift). Paper Supplementary Eq 2.
        eps_base = torch.ones(C, device=device)
        eps_base[:self.N_CELL_COLS] = 0.0
        if C > self.IDX_TITER:
            eps_base[self.IDX_TITER] = 0.0   # start at 0; flipped per-step below

        # Titer production rate is always ≥ 0 (cells can't un-produce antibody).
        # Concentration may still decrease after day 8 due to the washout term.
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

        # Pre-compute per-step titer epsilon as Python floats using .item().
        # .item() extracts a Python scalar — completely outside the autograd graph
        # with no tensor storage sharing. This avoids the version-counter corruption
        # that occurs when any tensor derived from time_points is used near in-place ops.
        # Assumes all samples in a batch share the same time axis (true for this dataset).
        if C > self.IDX_TITER:
            titer_eps_steps = [
                1.0 if time_points[0, t].item() >= self.DAY8_NORM else 0.0
                for t in range(T)
            ]
        else:
            titer_eps_steps = None

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

            # Build eps for this timestep. Before day 8: titer col = 0 (retained).
            # After day 8: titer col = 1 (washed out by perfusion).
            if titer_eps_steps is not None:
                te  = torch.full((B, 1), titer_eps_steps[t], device=device)           # (B, 1) constant
                eps = torch.cat([
                    eps_base[:self.IDX_TITER].unsqueeze(0).expand(B, -1),              # (B, IDX_TITER)
                    te,                                                                  # (B, 1)
                    eps_base[self.IDX_TITER + 1:].unsqueeze(0).expand(B, -1),          # (B, rest)
                ], dim=1)                                                                # (B, C)
            else:
                eps = eps_base                                                           # (C,)

            # Implicit Euler: stable for any F·dt
            numerator   = c_prev + (v * coupling + self.F_NORM * c_in * eps) * dt
            denominator = 1.0 + self.F_NORM * eps * dt
            concentrations.append(numerator / denominator)

        return torch.stack(concentrations, dim=1)      # (B, T, C)


class MultiHeadTemporalDecoder(nn.Module):
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.integrator     = DifferentiableIntegrator()
        self.phase_head     = PhaseTransitionHead(n_components, latent_dim)
        self.rate_predictor = RatePredictionHead(n_components, latent_dim)

    def forward(self, latent_state, time_points, initial_conditions, doe_params=None):
        growth_rates, prod_rates = self.rate_predictor(latent_state, time_points)

        # Pass 1: integrate with pure growth rates to get growth-phase concentration trajectory
        conc_pass1 = self.integrator(initial_conditions,
                                     growth_rates,   # f=0 → pure growth rates
                                     time_points, doe_params)

        # Pass 2: condition phase prediction on concentrations, then re-integrate
        phase_pred    = self.phase_head(latent_state, time_points, conc_pass1)
        blended_rates = (1 - phase_pred) * growth_rates + phase_pred * prod_rates
        concentrations = self.integrator(initial_conditions, blended_rates, time_points, doe_params)

        return {
            'concentrations': concentrations,
            'phase_weights':  phase_pred,
            'growth_rates':   growth_rates,
            'prod_rates':     prod_rates,
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
        self.amplitude   = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, n_components), nn.Softplus())
        self.phase_head  = PhaseTransitionHead(n_components, latent_dim)
        self.integrator  = DifferentiableIntegrator()
        self.n_layers    = n_layers

    def forward(self, latent_state, time_points, initial_conditions, doe_params=None):
        B, T = time_points.shape

        time_embedded   = self.time_embed(time_points.unsqueeze(-1))
        h0 = self.h0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = self.c0_proj(latent_state).unsqueeze(0).repeat(self.n_layers, 1, 1)
        lstm_out, _     = self.lstm(time_embedded, (h0, c0))
        latent_expanded = latent_state.unsqueeze(1).expand(-1, T, -1)
        combined        = torch.cat([lstm_out, latent_expanded], dim=-1)
        amp          = self.amplitude(latent_state).unsqueeze(1)           # (B, 1, C)
        growth_rates = self.growth_rates(combined) * amp
        prod_rates   = self.prod_rates(combined)   * amp

        # Pass 1: pure growth-phase concentrations to inform transition timing
        conc_pass1 = self.integrator(initial_conditions, growth_rates, time_points, doe_params)

        # Pass 2: condition phase prediction on concentrations, then re-integrate
        phase_pred    = self.phase_head(latent_state, time_points, conc_pass1)
        blended_rates = (1 - phase_pred) * growth_rates + phase_pred * prod_rates
        concentrations = self.integrator(initial_conditions, blended_rates, time_points, doe_params)

        return {
            'concentrations': concentrations,
            'phase_weights':  phase_pred,
            'growth_rates':   growth_rates,
            'prod_rates':     prod_rates,
        }



def _extract_doe(parameters):
    """Extract DoE coded levels [O2, AAs, Glc] from the parameters tensor.

    Parameter layout (set by train.py load_data):
      [0:3]   DoE coded levels  [O2, AAs, Glc]
      [3:28]  growth-phase specific rates  (encoder features only)
      [28:53] production-phase specific rates  (encoder features only)
      [53:]   FBA objective efficiencies  (encoder features only)
    """
    return parameters[:, :3] if parameters.shape[1] >= 3 else None


class CosmicNNSurrogateEnhanced(nn.Module):
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = MultiHeadTemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        latent = self.encoder(initial_conditions, parameters)
        return self.decoder(latent, time_points, initial_conditions,
                            doe_params=_extract_doe(parameters))


class CosmicNNSurrogateLSTM(nn.Module):
    """
    Top-level surrogate model combining a DynamicsEncoder (shared with attention version)
    and an LSTMTemporalDecoder (instead of multi-head attention).
    
    This class is the entry point for LSTM-based training. The only difference from
    CosmicNNSurrogateEnhanced is the decoder type; everything else (data handling,
    loss computation, evaluation) is identical.
    """
    
    def __init__(self, n_components, n_params, latent_dim=64, n_layers=2):
        """
        Args:
            n_components (int): Number of metabolite components (25 in this dataset)
            n_params (int): Number of encoder features (DoE + specific rates + FBA efficiencies)
            latent_dim (int): Dimension of latent state (default 64)
            n_layers (int): Number of LSTM layers (default 2; deeper = more capacity but more params)
        """
        super().__init__()  # Initialize nn.Module
        
        # ENCODER: Compresses initial conditions + parameters into a 64-dim latent vector
        # This latent vector captures reactor identity and will be used to initialize LSTM hidden state
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        
        # DECODER: LSTM-based temporal processor
        # Takes latent state, time points, and initial conditions
        # Returns concentrations, phase weights f(t), and intermediate rates
        self.decoder = LSTMTemporalDecoder(n_components, latent_dim, n_layers)
        
        # Store metadata for later use (e.g., in evaluation scripts)
        self.n_components = n_components
        self.n_params = n_params

    def forward(self, initial_conditions, time_points, parameters):
        """
        Forward pass: encodes then decodes.
        
        Args:
            initial_conditions (Tensor): (B, 25) — metabolite concentrations at t=0
            time_points (Tensor): (B, T) — normalized time [0, 1] for each reactor
            parameters (Tensor): (B, n_params) — DoE levels + specific rates + FBA efficiencies
        
        Returns:
            Dict with keys:
                'concentrations': (B, T, 25) — predicted metabolite trajectories
                'phase_weights': (B, T, 1) — f(t) phase interpolation
                'growth_rates': (B, T, 25) — predicted growth-phase rates
                'prod_rates': (B, T, 25) — predicted production-phase rates
        """
        
        # Step 1: ENCODING
        # Compress initial conditions + parameters into a single latent vector per reactor
        # Output shape: (B, latent_dim=64)
        latent = self.encoder(initial_conditions, parameters)
        
        # Step 2: DECODING (two-pass ODE integration)
        # Pass latent state to decoder along with:
        #   - time_points: for generating time embeddings
        #   - initial_conditions: needed for ODE integration (boundary conditions)
        #   - doe_params: extracted DoE levels [O2, AAs, Glc] for perfusion feed setup
        return self.decoder(latent, time_points, initial_conditions,
                            doe_params=_extract_doe(parameters))
