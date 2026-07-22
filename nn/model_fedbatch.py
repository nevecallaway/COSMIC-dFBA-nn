#!/usr/bin/env python3
"""
Fed-batch hybrid decoder: the fed-batch counterpart of model_primeur / model_stripped.

Same architecture principle (a small network predicts per-cell fluxes, a fixed
parameter-free ODE layer integrates them), but the mass balance is fed-batch
rather than perfusion, which is considerably simpler:

    perfusion:  dC/dt = F*(cin - eta*C) + v*X      (continuous flow + washout)
    fed-batch:  dC/dt = v*X                        (closed vessel between feeds)
                plus a DISCRETE feed bolus on feed days

Consequences worth noting:
  * No perfusion term, no eta, no washout. Product is retained, so titer
    accumulates monotonically. The whole day-8 eta question does not exist here.
  * A feed both ADDS medium and DILUTES everything already in the vessel. With
    bolus volume v_b added to working volume V, the update is a convex blend:

        f = v_b / (V + v_b)
        C <- C*(1 - f) + C_feed*f
        X <- X*(1 - f)                 (cells are diluted too, and not fed)

Closed-form day step (fluxes held constant over the day, as in the generator):
    cell density   X(1) = X0 * e^{vX}
    everything else C(1) = C0 + v_i * X0 * (e^{vX} - 1) / vX
then the feed blend if that day is fed. Built from exp/multiply/add only, so
autograd flows through it exactly as in the perfusion model.

Feature layout is set by the dataset loader (see prep_fedbatch.py); index 0 must
be cell density, since it is the flux driver X.
"""

import numpy as np
import torch
import torch.nn as nn

IDX_CD = 0            # cell density = X, the flux driver (must be feature 0)

N_FEATURES       = 6  # CellDensity, Titer, Glucose, Asparagine, Serine, Glycine
N_INPUT_FEATURES = N_FEATURES + 1      # + normalized day index
SEQ_LEN          = 6
N_DAYS           = 15                  # days 0..14 in the fed-batch dataset


def _expm1_over_x(x, eps=1e-6):
    """(exp(x) - 1) / x, numerically stable near 0 (limit -> 1)."""
    small  = x.abs() < eps
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 + 0.5 * x, torch.expm1(x) / safe_x)


def fedbatch_step(C_phys, v, feed_conc, feed_frac):
    """
    Advance the physical state one day under fed-batch mass balance.

    Args:
        C_phys:    (B, F) current physical concentrations; column 0 is cell density
        v:         (B, F) per-cell fluxes predicted by the network
        feed_conc: (B, F) feed-medium concentrations (0 for components not fed,
                          e.g. cell density and titer)
        feed_frac: (B, 1) f = v_bolus/(V + v_bolus) for this day; 0 on unfed days

    Returns:
        (B, F) physical concentrations at the next day
    """
    X0 = C_phys[:, IDX_CD:IDX_CD + 1]
    vX = v[:, IDX_CD:IDX_CD + 1]

    # --- closed vessel growth / consumption over one day ---
    grown = C_phys + v * X0 * _expm1_over_x(vX)          # dC/dt = v*X for all
    cd    = X0 * torch.exp(vX)                            # cell density is exponential
    C1    = torch.cat([cd, grown[:, IDX_CD + 1:]], dim=1)

    # --- discrete feed bolus: convex blend with the feed medium ---
    return C1 * (1.0 - feed_frac) + feed_conc * feed_frac


class FedBatchDecoder(nn.Module):
    """
    Low-capacity flux predictor (one conv layer + linear head, the design we
    settled on for the perfusion model) followed by the fixed fed-batch ODE step.

    Forward:
        x          (B, seq_len, n_input_features)  normalized window (+ time col)
        doe        (B, n_doe)                      run-level conditions
        feed_conc  (B, n_features)                 physical feed concentrations
        feed_frac  (B, 1)                          dilution fraction for this step
      returns:
        C_next_norm (B, n_features), v (B, n_features)
    """

    def __init__(self, n_features=N_FEATURES, n_input_features=N_INPUT_FEATURES,
                 hidden=16, n_doe=0, dropout=0.0):
        super().__init__()
        self.n_features = n_features
        self.n_doe      = n_doe

        self.conv = nn.Sequential(
            nn.Conv1d(n_input_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        ) if dropout == 0 else nn.Sequential(
            nn.Conv1d(n_input_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden + n_doe, n_features)

        self.register_buffer('feat_min',   torch.zeros(n_features))
        self.register_buffer('feat_scale', torch.ones(n_features))

    def set_scaler(self, scaler):
        """Load per-feature min/range from a fitted sklearn MinMaxScaler."""
        self.feat_min.copy_(torch.tensor(scaler.data_min_, dtype=torch.float32))
        rng = scaler.data_max_ - scaler.data_min_
        rng[rng == 0] = 1.0
        self.feat_scale.copy_(torch.tensor(rng, dtype=torch.float32))

    def predict_flux(self, x, doe=None):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=-1)
        context = (h * w.unsqueeze(-1)).sum(dim=1)
        if self.n_doe > 0 and doe is not None:
            context = torch.cat([context, doe], dim=-1)
        return self.head(context)

    def forward(self, x, doe, feed_conc, feed_frac):
        v = self.predict_flux(x, doe)
        C_phys = x[:, -1, :self.n_features] * self.feat_scale + self.feat_min
        C_next_phys = fedbatch_step(C_phys, v, feed_conc, feed_frac)
        return (C_next_phys - self.feat_min) / self.feat_scale, v


# ---------------------------------------------------------------------------
# Self-test: mass-balance sanity for the fed-batch step
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B, F = 4, N_FEATURES

    C = torch.rand(B, F) * 5 + 1.0
    v = torch.zeros(B, F)
    feed = torch.rand(B, F) * 10
    zero_frac = torch.zeros(B, 1)

    # v = 0 and no feed -> nothing changes
    out = fedbatch_step(C, v, feed, zero_frac)
    assert torch.allclose(out, C, atol=1e-5), 'closed vessel with v=0 must be static'

    # v = 0 with a feed -> exact convex blend
    f = torch.full((B, 1), 0.25)
    out = fedbatch_step(C, v, feed, f)
    assert torch.allclose(out, C * 0.75 + feed * 0.25, atol=1e-5), 'feed blend wrong'

    # feeding pure water (feed_conc = 0) must dilute by exactly (1 - f)
    out = fedbatch_step(C, v, torch.zeros_like(feed), f)
    assert torch.allclose(out, C * 0.75, atol=1e-5), 'dilution wrong'

    # positive growth flux with no feed -> cell density grows exponentially
    v2 = torch.zeros(B, F); v2[:, IDX_CD] = 0.5
    out = fedbatch_step(C, v2, feed, zero_frac)
    assert torch.allclose(out[:, IDX_CD], C[:, IDX_CD] * np.exp(0.5), atol=1e-4)

    # gradients flow through the ODE step
    v3 = torch.zeros(B, F, requires_grad=True)
    fedbatch_step(C, v3, feed, f).sum().backward()
    assert v3.grad is not None and torch.isfinite(v3.grad).all()

    print('Fed-batch mass-balance and gradient checks passed.')
