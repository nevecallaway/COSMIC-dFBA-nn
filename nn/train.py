#!/usr/bin/env python3
"""
Training script for COSMIC-dFBA surrogate v2.

Trains a NextDayPredictor (1D CNN + attention) on sliding 6-day windows
from synthetic ODE data. Predicts the next day's values for 8 bioprocess
features: cell density, cell size, titer, glucose, glutamine, asparagine,
serine, glycine.

Normalization: inputs normalized to [0,1] per feature (training stats).
               targets are raw (unnormalized).

Usage:
    !python train.py                              # default settings
    !python train.py --data synthetic_ode.npz     # explicit data path
    !python train.py --epochs 300 --hidden 128    # hyperparameter override
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model import NextDayPredictor, WindowDataset, N_FEATURES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LR         = 1e-3
EPOCHS     = 200
PATIENCE   = 20
VAL_SPLIT  = 0.2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default=str(here / 'synthetic_ode.npz'))
    parser.add_argument('--output',  default=str(here / 'model_v2.pt'))
    parser.add_argument('--epochs',  type=int,   default=EPOCHS)
    parser.add_argument('--hidden',  type=int,   default=64)
    parser.add_argument('--lr',      type=float, default=LR)
    parser.add_argument('--batch',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--seed',    type=int,   default=42)
    parser.add_argument('--shuffle', action='store_true',
                        help='Permutation baseline: shuffle targets before training')
    parser.add_argument('--titer-weight', type=float, default=1.0,
                        help='Loss weight multiplier for titer feature (index 2). '
                             'Values >1 penalise titer errors more heavily.')
    args = parser.parse_args()

    # Default output name for shuffled run
    if args.shuffle and args.output == str(here / 'model_v2.pt'):
        args.output = str(here / 'shuffled_v2.pt')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.shuffle:
        print('Permutation baseline: targets will be shuffled')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    npz         = np.load(args.data, allow_pickle=True)
    windows     = npz['windows']      # (n_obs, SEQ_LEN, N_FEATURES)  normalized
    targets     = npz['targets']      # (n_obs, N_FEATURES)            raw
    feature_min = npz['feature_min']
    feature_max = npz['feature_max']

    # Normalize DoE inputs to [0, 1] using per-column min/max
    doe_min    = npz['doe_min'].astype(np.float32)
    doe_max    = npz['doe_max'].astype(np.float32)
    doe_scale  = doe_max - doe_min
    doe_scale[doe_scale == 0] = 1.0
    window_doe = ((npz['window_doe'] - doe_min) / doe_scale).astype(np.float32)
    print(f'Windows: {len(windows)} | Features: {windows.shape[2]} | '
          f'Seq len: {windows.shape[1]} | DoE: {window_doe.shape[1]}')

    if args.shuffle:
        rng     = np.random.default_rng(args.seed)
        targets = targets[rng.permutation(len(targets))]

    dataset = WindowDataset(windows, targets, doe=window_doe)

    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    print(f'Train windows: {n_train} | Val windows: {n_val}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    n_doe = window_doe.shape[1]
    model = NextDayPredictor(hidden=args.hidden, n_doe=n_doe).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Per-feature loss weights: titer is feature index 2 in the 8-feature vector
    loss_weights = torch.ones(N_FEATURES, device=device)
    loss_weights[2] = args.titer_weight   # IDX_TITER = 2

    def criterion(pred, target):
        return ((pred - target) ** 2 * loss_weights).mean()

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss    = float('inf')
    patience_count   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, d, y in train_loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, d), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, d, y in val_loader:
                x, d, y = x.to(device), d.to(device), y.to(device)
                val_loss += criterion(model(x, d), y).item() * len(x)
        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'model_state': model.state_dict(),
                'feature_min': feature_min,
                'feature_max': feature_max,
                'doe_min':     doe_min,
                'doe_max':     doe_max,
                'hidden':      args.hidden,
                'n_features':  N_FEATURES,
                'n_doe':       n_doe,
            }, args.output)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
