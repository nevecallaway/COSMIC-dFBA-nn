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
from torch.utils.data import DataLoader
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

    # Split by reactor, not by window, to prevent data leakage
    reactor_idx  = npz['window_reactor_idx']
    all_reactors = np.unique(reactor_idx)
    rng_split    = np.random.default_rng(args.seed)
    rng_split.shuffle(all_reactors)
    n_val_reactors = max(1, int(len(all_reactors) * VAL_SPLIT))
    val_reactors   = set(all_reactors[:n_val_reactors].tolist())
    train_mask = np.array([r not in val_reactors for r in reactor_idx])
    val_mask   = ~train_mask

    if args.shuffle:
        rng     = np.random.default_rng(args.seed)
        targets = targets[rng.permutation(len(targets))]

    train_ds = WindowDataset(windows[train_mask], targets[train_mask], doe=window_doe[train_mask])
    val_ds   = WindowDataset(windows[val_mask],   targets[val_mask],   doe=window_doe[val_mask])
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    print(f'Train reactors: {len(all_reactors) - n_val_reactors} | Val reactors: {n_val_reactors}')
    print(f'Train windows: {n_train} | Val windows: {n_val}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    n_doe = window_doe.shape[1]
    model = NextDayPredictor(hidden=args.hidden, n_doe=n_doe).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def criterion(mu, log_var, target):
        # New loss function!
        # Gaussian NLL: exp(-log_var) * (mu - y)^2 + log_var
        # log_var is clamped for stability
        log_var = torch.clamp(log_var, -10, 10)
        return (torch.exp(-log_var) * (mu - target) ** 2 + log_var).mean()

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
            mu, log_var = model(x, d)
            loss = criterion(mu, log_var, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, d, y in val_loader:
                x, d, y = x.to(device), d.to(device), y.to(device)
                mu, log_var = model(x, d)
                val_loss += criterion(mu, log_var, y).item() * len(x)
        val_loss /= n_val

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}')

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
                print(f'Early stop at epoch {epoch}  best_val={best_val_loss:.6f}')
                break

    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
