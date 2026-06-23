#!/usr/bin/env python3
"""
COSMIC dFBA Surrogate Model v2.

Architecture: 1D CNN + attention head.
Task: given a 6-day window of 8 bioprocess features, predict the next day.

Features (8 total):
    0  Cell Density     (E9 cells/L)
    1  Cell Size        (um^3 / 1000)
    2  Titer            (mg/L)
    3  Glucose          (mmol/L)
    4  Glutamine        (mmol/L)
    5  Asparagine       (mmol/L)
    6  Serine           (mmol/L)
    7  Glycine          (mmol/L)

Normalization: inputs normalized to [0, 1] per feature using training-set
statistics. Targets are raw (unnormalized). No output normalization needed.

Workflow:
    dataset = WindowDataset(trajectories)        # builds sliding windows
    model   = NextDayPredictor()
    # train: predict next day from 6-day window
    # infer: roll forward autoregressively
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Feature selection: indices into the 25-component trajectory array
# ---------------------------------------------------------------------------
# Order here defines the 8-feature vector seen by the model.
FEATURE_INDICES = [
    0,   # Cell Density
    1,   # Cell Size
    5,   # Titer
    2,   # Glucose
    6,   # Glutamine
    8,   # Asparagine
    10,  # Serine
    11,  # Glycine
]
N_FEATURES = len(FEATURE_INDICES)   # 8
SEQ_LEN    = 4                      # window size (days)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Wraps pre-built windows and targets from the npz produced by
    generate_synthetic_ode.py.

        x: normalized 6-day window  shape (SEQ_LEN, N_FEATURES)
        y: raw next-day values       shape (N_FEATURES,)

    Windows are built and normalized at generation time so training
    can load directly without recomputing.
    """

    def __init__(self, windows, targets, doe=None):
        """
        Args:
            windows: np.ndarray (n_obs, SEQ_LEN, N_FEATURES)  normalized inputs
            targets: np.ndarray (n_obs, N_FEATURES)            raw targets
            doe:     np.ndarray (n_obs, 3) or None             DoE coded levels
        """
        self.windows = windows.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.doe     = doe.astype(np.float32) if doe is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])
        y = torch.from_numpy(self.targets[idx])
        d = torch.from_numpy(self.doe[idx]) if self.doe is not None else torch.zeros(0)
        return x, d, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NextDayPredictor(nn.Module):
    """
    1D CNN + attention head for next-day bioprocess prediction.

    Input:  (batch, SEQ_LEN, N_FEATURES)  -- normalized 6-day window
            (batch, n_doe)                -- DoE coded levels (O2, AAs, Glc)
    Output: (batch, N_FEATURES)           -- raw next-day values

    Architecture:
        1. Conv1d layers extract local temporal patterns across the 6-day window
        2. Attention scores each time step and produces a single context vector
        3. DoE vector concatenated to context (DoE is constant within a window)
        4. Linear head maps context+DoE to next-day prediction
    """

    def __init__(self, n_features=N_FEATURES, seq_len=SEQ_LEN,
                 hidden=64, n_conv_layers=3, dropout=0.1, n_doe=3):
        super().__init__()
        self.n_doe      = n_doe
        self.n_features = n_features

        # Conv stack: (batch, n_features, seq_len) -> (batch, hidden, seq_len)
        conv_layers = [
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
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

        # Attention: learn a scalar score per time step, softmax -> weighted sum
        self.attn = nn.Linear(hidden, 1)

        # Output head: raw prediction per feature
        self.head = nn.Sequential(
            nn.Linear(hidden + n_doe, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x, doe=None):
        """
        x:   (batch, seq_len, n_features)   normalized [0, 1] window
        doe: (batch, n_doe) or None
        returns: (batch, n_features)         raw (unnormalized) predictions
        """
        h = self.conv(x.transpose(1, 2))        # (batch, hidden, seq_len)
        h = h.transpose(1, 2)                    # (batch, seq_len, hidden)

        scores  = self.attn(h).squeeze(-1)       # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)

        if self.n_doe > 0 and doe is not None:
            context = torch.cat([context, doe], dim=-1)  # (batch, hidden + n_doe)

        return self.head(context)                         # (batch, n_features)
