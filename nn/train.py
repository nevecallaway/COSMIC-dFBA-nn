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

FEATURE_NAMES = [
    'Cell Density', 'Cell Size', 'Titer',
    'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine',
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    """Mean squared error on raw targets, per feature and overall."""
    model.eval()
    sq_err = np.zeros(N_FEATURES)
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            sq_err += ((pred - y) ** 2).sum(dim=0).cpu().numpy()
            n += len(x)
    mse_per_feature = sq_err / n
    return mse_per_feature


def print_metrics(mse_per_feature, label=''):
    rmse = np.sqrt(mse_per_feature)
    header = f'  {label}' if label else ''
    print(header)
    for name, r in zip(FEATURE_NAMES, rmse):
        print(f'    {name:<18}: RMSE {r:.4f}')
    print(f'    {"Mean":<18}: RMSE {rmse.mean():.4f}')


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    npz         = np.load(args.data, allow_pickle=True)
    windows     = npz['windows']      # (n_obs, SEQ_LEN, N_FEATURES)  normalized
    targets     = npz['targets']      # (n_obs, N_FEATURES)            raw
    feature_min = npz['feature_min']
    feature_max = npz['feature_max']
    print(f'Windows: {len(windows)} | Features: {windows.shape[2]} | '
          f'Seq len: {windows.shape[1]}')

    dataset = WindowDataset(windows, targets)

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
    model     = NextDayPredictor(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
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
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= n_val

        if epoch % 10 == 0:
            print(f'Epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'model_state': model.state_dict(),
                'feature_min': feature_min,
                'feature_max': feature_max,
                'hidden':      args.hidden,
                'n_features':  N_FEATURES,
            }, args.output)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    # ------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    print(f'\nBest val loss: {best_val_loss:.6f}')
    print(f'Saved to {args.output}\n')

    train_mse = evaluate(model, train_loader, device)
    val_mse   = evaluate(model, val_loader,   device)
    print_metrics(train_mse, label='Train RMSE (raw units):')
    print()
    print_metrics(val_mse,   label='Val RMSE (raw units):')


if __name__ == '__main__':
    main()
