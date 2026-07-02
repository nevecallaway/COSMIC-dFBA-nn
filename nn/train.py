#!/usr/bin/env python3
"""
Training script for COSMIC-dFBA surrogate v2.

Trains a NextDayPredictor (1D CNN + attention) on sliding 6-day windows
from synthetic ODE data. Predicts the next day's values for 8 bioprocess
features: cell density, cell size, titer, glucose, glutamine, asparagine,
serine, glycine.

Normalization: inputs and targets normalized to [0,1] per feature (training stats).

Usage:
    !python train.py                              # default settings
    !python train.py --data synthetic_ode.npz     # explicit data path
    !python train.py --epochs 300 --hidden 128    # hyperparameter override
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from model import NextDayPredictor, WindowDataset, N_FEATURES, N_INPUT_FEATURES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BATCH_SIZE    = 8
LR            = 1e-3
EPOCHS        = 200
PATIENCE      = 20
VAL_SPLIT     = 0.2
SIGMA_WARMUP  = 50   # epochs to train with sigma frozen before unlocking

# Asparagine (5) and Serine (6) are flat in ODE data (initial concentrations
# too low relative to consumption rates). Pin their log_sigma high so the loss
# treats them as low-confidence and stops pulling gradient away from titer.
FROZEN_SIGMA_IDX = [5, 6]
FROZEN_LOG_SIGMA = 2.0   # sigma≈7.4, weight≈0.018


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
    parser.add_argument('--sigma-warmup', type=int, default=SIGMA_WARMUP,
                        help='Epochs to train with sigma frozen (default: 50)')
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
    npz     = np.load(args.data, allow_pickle=True)
    windows = npz['windows'].astype(np.float32)   # (n_obs, SEQ_LEN, N_FEATURES)  raw
    targets = npz['targets'].astype(np.float32)   # (n_obs, N_FEATURES)            raw

    # Normalize DoE inputs to [0, 1] using per-column min/max from extra reactors
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

    # Separate feature columns from the time column (last column, already [0,1])
    win_feats = windows[:, :, :N_FEATURES]   # (n_obs, SEQ_LEN, N_FEATURES)  raw
    win_time  = windows[:, :, N_FEATURES:]   # (n_obs, SEQ_LEN, 1)           normalized day

    # Fit scaler on training features only (time column excluded)
    train_flat = np.vstack([
        win_feats[train_mask].reshape(-1, N_FEATURES),
        targets[train_mask],
    ])
    scaler = MinMaxScaler()
    scaler.fit(train_flat)
    print(f'Scaler fitted on {len(train_flat)} training samples '
          f'(train windows + targets; val excluded)')

    def _norm_windows(wf, wt):
        n, s, f = wf.shape
        scaled = scaler.transform(wf.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
        return np.concatenate([scaled, wt], axis=2)  # (n, SEQ_LEN, N_INPUT_FEATURES)

    windows_tr = _norm_windows(win_feats[train_mask], win_time[train_mask])
    targets_tr = scaler.transform(targets[train_mask]).astype(np.float32)
    windows_vl = _norm_windows(win_feats[val_mask],   win_time[val_mask])
    targets_vl = scaler.transform(targets[val_mask]).astype(np.float32)

    train_ds = WindowDataset(windows_tr, targets_tr, doe=window_doe[train_mask])
    val_ds   = WindowDataset(windows_vl, targets_vl, doe=window_doe[val_mask])
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    print(f'Train reactors: {len(all_reactors) - n_val_reactors} | Val reactors: {n_val_reactors}')
    print(f'Train windows: {n_train} | Val windows: {n_val}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    n_doe = window_doe.shape[1]
    model = NextDayPredictor(hidden=args.hidden, n_doe=n_doe,
                             n_input_features=N_INPUT_FEATURES).to(device)

    # Sigma starts frozen so the model learns to predict all features first.
    # After sigma_warmup epochs it is unlocked and added to the optimizer.
    # Asparagine and Serine sigmas are permanently frozen at FROZEN_LOG_SIGMA.
    log_sigma = torch.zeros(N_FEATURES, device=device, requires_grad=False)
    log_sigma[FROZEN_SIGMA_IDX] = FROZEN_LOG_SIGMA
    sigma_unlocked = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def criterion(pred, target):
        per_feat_mse = ((pred - target) ** 2).mean(dim=0)
        return (torch.exp(-2 * log_sigma) * per_feat_mse + log_sigma).mean()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,} model + {N_FEATURES} log_sigma')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss  = float('inf')
    patience_count = 0

    feature_names = ['CellDensity', 'CellSize', 'Titer',
                     'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine']
    log_path = Path(args.output).with_suffix('.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ['epoch', 'train_loss', 'val_loss']
        + [f'sigma_{n}' for n in feature_names]
        + [f'mse_{n}' for n in feature_names])

    for epoch in range(1, args.epochs + 1):
        if not sigma_unlocked and epoch > args.sigma_warmup:
            log_sigma.requires_grad_(True)
            optimizer.add_param_group({'params': [log_sigma]})
            sigma_unlocked = True
            free_names  = [feature_names[i] for i in range(N_FEATURES)
                           if i not in FROZEN_SIGMA_IDX]
            print(f'Epoch {epoch:4d}  sigma unlocked for: {free_names}')

        model.train()
        train_loss = 0.0
        feat_mse_accum = np.zeros(N_FEATURES)
        for x, d, y in train_loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, d)
            loss = criterion(pred, y)
            loss.backward()
            if sigma_unlocked and log_sigma.grad is not None:
                log_sigma.grad[FROZEN_SIGMA_IDX] = 0.0
            optimizer.step()
            train_loss += loss.item() * len(x)
            with torch.no_grad():
                log_sigma[FROZEN_SIGMA_IDX] = FROZEN_LOG_SIGMA
                feat_mse_accum += ((pred - y) ** 2).mean(dim=0).cpu().numpy() * len(x)
        train_loss /= n_train
        feat_mse = feat_mse_accum / n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, d, y in val_loader:
                x, d, y = x.to(device), d.to(device), y.to(device)
                pred = model(x, d)
                val_loss += criterion(pred, y).item() * len(x)
        val_loss /= n_val

        sigmas = torch.exp(log_sigma).detach().cpu().numpy()
        log_writer.writerow(
            [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}']
            + [f'{s:.6f}' for s in sigmas]
            + [f'{m:.6f}' for m in feat_mse])

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}'
                  f'  sigma=[{", ".join(f"{s:.3f}" for s in sigmas)}]')

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'model_state':     model.state_dict(),
                'log_sigma':       log_sigma.detach().cpu(),
                'scaler':          scaler,
                'doe_min':         doe_min,
                'doe_max':         doe_max,
                'hidden':          args.hidden,
                'n_features':      N_FEATURES,
                'n_input_features': N_INPUT_FEATURES,
                'n_doe':           n_doe,
            }, args.output)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'Early stop at epoch {epoch}  best_val={best_val_loss:.4f}')
                break

    log_file.close()
    print(f'Saved to {args.output}')
    print(f'Training log: {log_path}')


if __name__ == '__main__':
    main()
