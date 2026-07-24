#!/usr/bin/env python3
"""
EN PRIMEUR: Efficient Neural Predictor Recursively Integrating Mechanistic
Equations for Uptake Rates.

Flux-prediction ODE decoder (hybrid mechanistic surrogate), part of the
CHOteau project.

Difference from model.py (pure NN):
    model.py       window --NN--> next-day concentrations   (physics-free)
    model_primeur  window --NN--> fluxes v --ODE step--> next-day concentrations

The network predicts the per-cell fluxes v_i. A fixed, parameter-free ODE-step
layer then applies the paper's mass balance (eq. 2) to produce the next-day
concentrations. Mass balance is therefore exact by construction: whatever v the
network emits, the resulting step obeys the ODE. There is no lambda to tune.

Feature layout (8 model features, order matches model.py FEATURE_INDICES):
    0  Cell Density   dC = v*X                 (eta=0, no perfusion)
    1  Cell Size      dC = v*X                 (eta=0, driven by X)
    2  Titer          dC = v*X - F*C           (eta=1, no feed source)
    3  Glucose        dC = F*(cin - C) + v*X   (perfused metabolite)
    4  Glutamine      "
    5  Asparagine     "
    6  Serine         "
    7  Glycine        "

Integration: the generator (generate_synthetic_ode.py) integrates each 1-day
interval with RK45 holding v constant. To match, the ODE step sub-divides the
day into N_SUBSTEPS explicit-Euler steps with v held constant and X updated each
sub-step. This is arithmetic only (no extra network passes), so cost is
negligible relative to the conv stack.

Normalization: the network works in [0,1] (per-feature MinMaxScaler). The ODE
step needs physical units (cin, F, X are physical). The layer un-normalizes the
current state, steps in physical units, then re-normalizes the result. The
scaler's per-feature min and range are held as buffers (set via set_scaler).
"""

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model feature layout (indices into the 8-vector the model sees)
# ---------------------------------------------------------------------------
IDX_CD    = 0    # Cell Density  (this is X, the flux driver)
IDX_SIZE  = 1    # Cell Size
IDX_TITER = 2    # Titer
MET_SLICE = slice(3, 8)   # Glucose, Glutamine, Asparagine, Serine, Glycine

N_FEATURES       = 8
N_INPUT_FEATURES = N_FEATURES + 1   # + normalized day-index column
SEQ_LEN          = 6
N_DAYS           = 13               # day 0..12 (for reconstructing the day index)
N_SUBSTEPS       = 50               # Euler sub-steps per 1-day interval
                                    # (50 -> ~0.5% vs generator RK45; scales 1/n)
F_PERFUSION      = 1.0              # bioreactor volumes/day (paper: F = 1)
ETA_SWITCH_DAY   = 8                # titer washout (eta) is 0 before this day, 1 after
                                    # (paper: antibody retained then harvested at day 8)


# ---------------------------------------------------------------------------
# ODE-step layer (no parameters)
# ---------------------------------------------------------------------------

def ode_step(C_phys, v, cin_phys, eta_titer=1.0, n_substeps=N_SUBSTEPS, F=F_PERFUSION):
    """
    Advance the physical state one day using the paper's mass balance, with v
    held constant over the day (matches the generator's per-interval scheme).

    Args:
        C_phys:   (B, 8) current physical concentrations (last day of window)
        v:        (B, 8) per-cell fluxes predicted by the network
        cin_phys: (B, 8) physical feed concentrations (only MET_SLICE used)
        eta_titer: titer removal fraction (0 = retained/accumulating, 1 = washed
                   out). Scalar or (B,1); the day-8 switch is applied by the caller.
        n_substeps: Euler sub-steps within the day
        F:        perfusion rate

    Returns:
        (B, 8) physical concentrations at the next day
    """
    dt = 1.0 / n_substeps
    C = C_phys
    for _ in range(n_substeps):
        X = C[:, IDX_CD:IDX_CD + 1]        # (B,1) current cell density

        d_cd    = v[:, IDX_CD:IDX_CD + 1] * X
        d_size  = v[:, IDX_SIZE:IDX_SIZE + 1] * X
        d_titer = (v[:, IDX_TITER:IDX_TITER + 1] * X
                   - eta_titer * F * C[:, IDX_TITER:IDX_TITER + 1])
        d_met   = F * (cin_phys[:, MET_SLICE] - C[:, MET_SLICE]) + v[:, MET_SLICE] * X

        dC = torch.cat([d_cd, d_size, d_titer, d_met], dim=1)
        C = C + dt * dC
    return C


def _expm1_over_x(x, eps=1e-6):
    """(exp(x) - 1) / x, numerically stable near 0 (limit -> 1)."""
    small  = x.abs() < eps
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 + 0.5 * x, torch.expm1(x) / safe_x)


def closed_form_step(C_phys, v, cin_phys, eta_titer=1.0, F=F_PERFUSION):
    """
    Exact one-day advance. With v and cin constant over the day, X(t) = X0*e^{vX t}
    and every equation has a closed-form solution, so no Euler sub-steps are needed
    (faster than ode_step and exact modulo the generator's own RK45 tolerance).

        cell density : X0 * e^{vX}
        cell size    : CV0 + v_CV*X0 * (e^{vX}-1)/vX
        titer        : Tit0*e^{-ηF} + v_tit*X0*e^{-ηF} * (e^{vX+ηF}-1)/(vX+ηF)
        metabolite   : C0*e^{-F} + cin*(1-e^{-F}) + v*X0*e^{-F} * (e^{vX+F}-1)/(vX+F)

    eta_titer (η) is the titer removal fraction: 0 before day 8 (retained, so titer
    accumulates: the term reduces to Tit0 + v_tit*X0*(e^{vX}-1)/vX), 1 after (washed
    out). Scalar or (B,1). The (e^{·}-1)/· factors use _expm1_over_x, which handles
    the degenerate cases vX -> 0 and vX+ηF -> 0.
    """
    X0  = C_phys[:, IDX_CD:IDX_CD + 1]
    vX  = v[:, IDX_CD:IDX_CD + 1]
    emF = float(np.exp(-F))

    g_vXF = _expm1_over_x(vX + F)   # metabolite factor (eta = 1 always for metabolites)

    # Titer with variable eta: washout coefficient b = eta*F.
    b       = eta_titer * F
    e_negb  = torch.exp(-b) if torch.is_tensor(b) else float(np.exp(-b))
    g_titer = _expm1_over_x(vX + b)

    cd    = X0 * torch.exp(vX)
    size  = (C_phys[:, IDX_SIZE:IDX_SIZE + 1]
             + v[:, IDX_SIZE:IDX_SIZE + 1] * X0 * _expm1_over_x(vX))
    titer = (C_phys[:, IDX_TITER:IDX_TITER + 1] * e_negb
             + v[:, IDX_TITER:IDX_TITER + 1] * X0 * e_negb * g_titer)
    met   = (C_phys[:, MET_SLICE] * emF
             + cin_phys[:, MET_SLICE] * (1.0 - emF)
             + v[:, MET_SLICE] * X0 * emF * g_vXF)

    return torch.cat([cd, size, titer, met], dim=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FluxDecoder(nn.Module):
    """
    1D CNN + attention body predicting fluxes, followed by a fixed ODE step.

    Forward:
        x        (B, seq_len, n_input_features)  normalized window (+ time col)
        doe      (B, n_doe)                      DoE coded levels
        cin_phys (B, n_features)                 physical feed concentrations
      returns:
        C_next_norm (B, n_features)  normalized next-day concentrations
        v           (B, n_features)  predicted fluxes (for inspection / aux loss)
    """

    def __init__(self, n_features=N_FEATURES, n_input_features=N_INPUT_FEATURES,
                 hidden=64, n_conv_layers=3, dropout=0.1, n_doe=3,
                 n_substeps=N_SUBSTEPS, integrator='closed', residual_weight=0.0):
        super().__init__()
        self.n_features = n_features
        self.n_doe      = n_doe
        self.n_substeps = n_substeps
        self.integrator = integrator   # 'closed' (exact, fast) or 'euler'

        # ODE-relaxation knob. residual_weight=0 is the pure hybrid (the ODE is a
        # hard layer). >0 adds a free, non-ODE correction the network can use to
        # bend away from the physics where the data disagrees (e.g. the day-8 peak
        # the real data doesn't share). Sweeping it trades real-data fit against
        # the generalization the ODE provides. The correction head is built lazily
        # in forward so it only exists when the knob is on.
        self.residual_weight = residual_weight
        self.residual_head = None

        conv_layers = [
            nn.Conv1d(n_input_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_conv_layers - 1):
            conv_layers += [
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.conv = nn.Sequential(*conv_layers)
        self.attn = nn.Linear(hidden, 1)

        # Head now emits fluxes (can be negative), so no output activation.
        self.head = nn.Sequential(
            nn.Linear(hidden + n_doe, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )
        if residual_weight > 0:
            self.residual_head = nn.Linear(hidden + n_doe, n_features)

        # Scaler params (physical <-> normalized). Set via set_scaler().
        self.register_buffer('feat_min',   torch.zeros(n_features))
        self.register_buffer('feat_scale', torch.ones(n_features))

    def set_scaler(self, scaler):
        """Load per-feature min and range from a fitted sklearn MinMaxScaler."""
        self.feat_min.copy_(torch.tensor(scaler.data_min_, dtype=torch.float32))
        rng = scaler.data_max_ - scaler.data_min_
        rng[rng == 0] = 1.0
        self.feat_scale.copy_(torch.tensor(rng, dtype=torch.float32))

    def _context(self, x, doe):
        """Network body: window -> pooled context (B, hidden [+ n_doe])."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)     # (B, seq, hidden)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=-1)  # (B, seq)
        context = (h * w.unsqueeze(-1)).sum(dim=1)           # (B, hidden)
        if self.n_doe > 0 and doe is not None:
            context = torch.cat([context, doe], dim=-1)
        return context

    def predict_flux(self, x, doe):
        """Window -> fluxes v (B, n_features)."""
        return self.head(self._context(x, doe))

    def forward(self, x, doe, cin_phys, eta_ext=None):
        context = self._context(x, doe)
        v = self.head(context)

        # Current physical state = last day of the window, un-normalized.
        C_last_norm = x[:, -1, :self.n_features]
        C_phys = C_last_norm * self.feat_scale + self.feat_min

        # Titer eta (removal fraction). If eta_ext is given (phase-driven washout,
        # --phase), use it directly: (B,1) tensor in training, scalar in rollout.
        # Otherwise fall back to the blanket day-8 switch, reconstructing the
        # window's last day t from the time column (eta = 0 while t < day 8,
        # titer retained/accumulating; 1 after, harvested/washed out).
        if eta_ext is not None:
            eta_titer = eta_ext
        else:
            day = torch.round(x[:, -1, self.n_features] * (N_DAYS - 1))
            eta_titer = (day >= ETA_SWITCH_DAY).float().unsqueeze(1)   # (B,1)

        if self.integrator == 'closed':
            C_next_phys = closed_form_step(C_phys, v, cin_phys, eta_titer=eta_titer)
        else:
            C_next_phys = ode_step(C_phys, v, cin_phys, eta_titer=eta_titer,
                                   n_substeps=self.n_substeps)

        C_next_norm = (C_next_phys - self.feat_min) / self.feat_scale

        # ODE relaxation: add a free correction (in normalized space) so the network
        # can deviate from the physics. residual_weight=0 -> pure hybrid (no-op).
        if self.residual_head is not None:
            C_next_norm = C_next_norm + self.residual_weight * self.residual_head(context)

        return C_next_norm, v


# ---------------------------------------------------------------------------
# Self-test: shapes + mass-balance sanity (run locally or on Colab)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    B = 4
    model = FluxDecoder()

    # Fake a fitted scaler: physical range [0, 10] for every feature.
    class _S:
        data_min_ = np.zeros(N_FEATURES, dtype=np.float32)
        data_max_ = np.full(N_FEATURES, 10.0, dtype=np.float32)
    model.set_scaler(_S())

    x   = torch.rand(B, SEQ_LEN, N_INPUT_FEATURES)
    doe = torch.rand(B, 3)
    cin = torch.rand(B, N_FEATURES) * 5.0

    C_next, v = model(x, doe, cin)
    print(f'C_next: {tuple(C_next.shape)}   v: {tuple(v.shape)}')
    assert C_next.shape == (B, N_FEATURES)
    assert v.shape == (B, N_FEATURES)

    # Mass-balance sanity: with v = 0 and feed == current metabolite conc,
    # metabolites must not change; titer must decay by F*C; cell density/size
    # must stay put (no growth). Check in physical units directly.
    C_phys = torch.rand(B, N_FEATURES) * 5.0
    v0     = torch.zeros(B, N_FEATURES)
    cin_eq = C_phys.clone()                     # feed matches current state
    out    = ode_step(C_phys, v0, cin_eq, n_substeps=10)

    # metabolites: cin == C and v == 0 -> unchanged
    assert torch.allclose(out[:, MET_SLICE], C_phys[:, MET_SLICE], atol=1e-5), \
        'metabolites should be unchanged when v=0 and feed matches'
    # cell density/size: v == 0 -> unchanged
    assert torch.allclose(out[:, IDX_CD], C_phys[:, IDX_CD], atol=1e-5)
    assert torch.allclose(out[:, IDX_SIZE], C_phys[:, IDX_SIZE], atol=1e-5)
    # titer: v == 0 -> decays by factor (1 - F*dt)^n_substeps ~ exp(-F)
    expected_titer = C_phys[:, IDX_TITER] * (1 - 1.0 / 10) ** 10
    assert torch.allclose(out[:, IDX_TITER], expected_titer, atol=1e-5)

    print('Mass-balance sanity checks passed.')
    print('\nNext: fidelity test on Colab -- feed a real window and the TRUE')
    print('flux v_net for that reactor/day; ode_step output must match the')
    print('generate_synthetic_ode.py trajectory to tolerance.')
