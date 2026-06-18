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
SEQ_LEN    = 6                      # window size (days)


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

    def __init__(self, windows, targets):
        """
        Args:
            windows: np.ndarray (n_obs, SEQ_LEN, N_FEATURES)  normalized inputs
            targets: np.ndarray (n_obs, N_FEATURES)            raw targets
        """
        self.windows = windows.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),  # x: (SEQ_LEN, N_FEATURES)
            torch.from_numpy(self.targets[idx]),  # y: (N_FEATURES,)
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NextDayPredictor(nn.Module):
    """
    1D CNN + attention head for next-day bioprocess prediction.

    Input:  (batch, SEQ_LEN, N_FEATURES)  -- normalized 6-day window
    Output: (batch, N_FEATURES)           -- raw next-day values

    Architecture:
        1. Conv1d layers extract local temporal patterns across the 6-day window
        2. Attention scores each time step and produces a single context vector
        3. Linear head maps context to next-day prediction
    """

    def __init__(self, n_features=N_FEATURES, seq_len=SEQ_LEN,
                 hidden=64, n_conv_layers=3, dropout=0.1):
        super().__init__()

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

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch, n_features)
        """
        h = self.conv(x.transpose(1, 2))        # (batch, hidden, seq_len)
        h = h.transpose(1, 2)                    # (batch, seq_len, hidden)

        scores  = self.attn(h).squeeze(-1)       # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)

        return self.head(context)                # (batch, n_features)
