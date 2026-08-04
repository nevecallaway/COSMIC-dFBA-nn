#!/usr/bin/env python3
"""
Stripped flux decoder: the low-capacity, "old-fashioned" version of en Primeur.

Same idea as model_primeur.FluxDecoder (network predicts fluxes, a fixed ODE step
integrates them, mass balance exact by construction), but the learnable body is
deliberately small to suit the tiny real dataset (10 reactors):

    body:  ONE 1D conv layer, few channels, attention pool, a SINGLE linear head
    no dropout (regularize with weight decay instead)
    no flux clamp (a hard cap is biologically artificial; low capacity alone keeps
                   the fluxes from running away)

The physics (ode_step / closed_form_step, the eta day-8 switch, the un/normalize
logic) is inherited unchanged from model_primeur, so this is purely a capacity
reduction of the same architecture, not a different model. Interface (forward,
predict_flux, set_scaler) is identical, so it drops into train_sample.py,
train_real.py, evaluate.py and plot_loro.py with a one-line import swap.

Rationale (Kimberly): SOA nets "learn everything" and need many parameters; with
10 reactors we go back to old-fashioned, low-parameter methods and lean on the
mechanism (the fixed ODE) as the prior instead of a big learned body.
"""

import torch.nn as nn

from model_primeur import (
    FluxDecoder as _PrimeurDecoder,
    N_FEATURES, N_INPUT_FEATURES, SEQ_LEN, N_SUBSTEPS,
)

__all__ = ['FluxDecoder', 'N_FEATURES', 'N_INPUT_FEATURES', 'SEQ_LEN', 'N_SUBSTEPS']


class FluxDecoder(_PrimeurDecoder):
    """en Primeur with a low-capacity body: 1 conv layer + linear head."""

    def __init__(self, n_features=N_FEATURES, n_input_features=N_INPUT_FEATURES,
                 hidden=16, n_conv_layers=1, dropout=0.0, n_doe=3,
                 n_substeps=N_SUBSTEPS, integrator='closed', residual_weight=0.0):
        # Parent builds a single conv layer (n_conv_layers=1) and the buffers/attn;
        # we then replace the MLP head with one linear layer to drop parameters.
        super().__init__(n_features=n_features, n_input_features=n_input_features,
                         hidden=hidden, n_conv_layers=1, dropout=dropout,
                         n_doe=n_doe, n_substeps=n_substeps, integrator=integrator,
                         residual_weight=residual_weight)
        self.head = nn.Linear(hidden + n_doe, n_features)
