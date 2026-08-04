#!/usr/bin/env python3
"""
Training script for the flux-decoder (hybrid mechanistic) surrogate.

Same data pipeline as train.py, but the model is FluxDecoder (model_primeur.py):
the network predicts fluxes, a fixed ODE-step layer turns them into next-day
concentrations. Loss is on the predicted concentrations, so mass balance is
enforced by construction, not by a penalty term.

Differences from train.py:
  - loads window_cin (physical feed per window) and passes it to forward
  - model.set_scaler(scaler) so the ODE step can move between normalized and
    physical units
  - forward returns (C_next_norm, v); loss uses C_next_norm
  - no frozen asparagine/serine sigmas: with the corrected media those features
    vary normally, so the old flat-feature workaround is unnecessary

Usage:
    python train_sample.py
    python train_sample.py --epochs 300 --substeps 50
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from device_utils import pick_device
from model_primeur import (FluxDecoder, N_FEATURES, N_INPUT_FEATURES, SEQ_LEN,
                           N_SUBSTEPS, N_DAYS)

BATCH_SIZE   = 8
LR           = 1e-3
EPOCHS       = 200
PATIENCE     = 20
VAL_SPLIT    = 0.2
SIGMA_WARMUP = 50

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer', 'Glucose',
                 'Glutamine', 'Asparagine', 'Serine', 'Glycine']


class FluxWindowDataset(Dataset):
    """Windows + DoE + physical feed (cin) + per-window titer eta + targets."""

    def __init__(self, windows, doe, cin, eta, targets):
        self.windows = windows.astype(np.float32)
        self.doe     = doe.astype(np.float32)
        self.cin     = cin.astype(np.float32)
        self.eta     = eta.astype(np.float32).reshape(-1, 1)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return (torch.from_numpy(self.windows[i]),
                torch.from_numpy(self.doe[i]),
                torch.from_numpy(self.cin[i]),
                torch.from_numpy(self.eta[i]),
                torch.from_numpy(self.targets[i]))


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default=str(here / 'synthetic_ode.npz'))
    parser.add_argument('--output',  default=str(here / 'model_flux.pt'))
    parser.add_argument('--epochs',  type=int,   default=EPOCHS)
    parser.add_argument('--hidden',  type=int,   default=64)
    parser.add_argument('--lr',      type=float, default=LR)
    parser.add_argument('--batch',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--seed',    type=int,   default=42)
    parser.add_argument('--substeps', type=int,  default=N_SUBSTEPS)
    parser.add_argument('--sigma-warmup', type=int, default=SIGMA_WARMUP)
    parser.add_argument('--stripped', action='store_true',
                        help='Pretrain the low-capacity model_stripped decoder (for '
                             'synthetic->real transfer of the stripped model)')
    args = parser.parse_args()

    if args.stripped:
        from model_stripped import FluxDecoder as ModelClass
        print('Using stripped (low-capacity) decoder')
    else:
        ModelClass = FluxDecoder

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f'Device: {device}')

    # ------------------------------------------------------------------ data
    npz     = np.load(args.data, allow_pickle=True)
    windows = npz['windows'].astype(np.float32)     # (n, seq, N_FEATURES+1) raw feats + time
    targets = npz['targets'].astype(np.float32)     # (n, N_FEATURES)        raw
    if 'window_cin' not in npz:
        raise SystemExit('window_cin missing: regenerate synthetic_ode.npz with '
                         'the updated generate_synthetic_ode.py (cin plumbing).')
    window_cin = npz['window_cin'].astype(np.float32)   # (n, N_FEATURES) physical feed
    seq_len = int(npz['seq_len']) if 'seq_len' in npz else SEQ_LEN
    if 'window_eta' in npz:
        window_eta = npz['window_eta'].astype(np.float32)
    else:
        # Old npz (pre-eta): reconstruct the day-8 eta from each window's last day.
        last_day = np.round(windows[:, -1, N_FEATURES] * (N_DAYS - 1))
        window_eta = (last_day >= 8).astype(np.float32)

    doe_min   = npz['doe_min'].astype(np.float32)
    doe_max   = npz['doe_max'].astype(np.float32)
    doe_scale = doe_max - doe_min
    doe_scale[doe_scale == 0] = 1.0
    window_doe = ((npz['window_doe'] - doe_min) / doe_scale).astype(np.float32)
    print(f'Windows: {len(windows)} | Features: {windows.shape[2]} | Seq: {windows.shape[1]}')

    # Split by reactor to prevent leakage
    reactor_idx  = npz['window_reactor_idx']
    all_reactors = np.unique(reactor_idx)
    rng_split    = np.random.default_rng(args.seed)
    rng_split.shuffle(all_reactors)
    n_val_reactors = max(1, int(len(all_reactors) * VAL_SPLIT))
    val_reactors   = set(all_reactors[:n_val_reactors].tolist())
    train_mask = np.array([r not in val_reactors for r in reactor_idx])
    val_mask   = ~train_mask

    win_feats = windows[:, :, :N_FEATURES]
    win_time  = windows[:, :, N_FEATURES:]

    # Scaler on training features + targets only (time column excluded)
    train_flat = np.vstack([win_feats[train_mask].reshape(-1, N_FEATURES),
                            targets[train_mask]])
    scaler = MinMaxScaler().fit(train_flat)

    def _norm_windows(wf, wt):
        n, s, f = wf.shape
        scaled = scaler.transform(wf.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
        return np.concatenate([scaled, wt], axis=2)

    windows_tr = _norm_windows(win_feats[train_mask], win_time[train_mask])
    targets_tr = scaler.transform(targets[train_mask]).astype(np.float32)
    windows_vl = _norm_windows(win_feats[val_mask],   win_time[val_mask])
    targets_vl = scaler.transform(targets[val_mask]).astype(np.float32)

    # cin stays in PHYSICAL units; the model un-normalizes internally.
    train_ds = FluxWindowDataset(windows_tr, window_doe[train_mask],
                                 window_cin[train_mask], window_eta[train_mask], targets_tr)
    val_ds   = FluxWindowDataset(windows_vl, window_doe[val_mask],
                                 window_cin[val_mask], window_eta[val_mask], targets_vl)
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    print(f'Train reactors: {len(all_reactors) - n_val_reactors} | '
          f'Val reactors: {n_val_reactors}')
    print(f'Train windows: {n_train} | Val windows: {n_val}')

    # ----------------------------------------------------------------- model
    n_doe = window_doe.shape[1]
    model = ModelClass(hidden=args.hidden, n_doe=n_doe,
                       n_input_features=N_INPUT_FEATURES,
                       n_substeps=args.substeps).to(device)
    model.set_scaler(scaler)

    # Heteroscedastic loss, all sigmas learnable after warmup (no frozen features)
    log_sigma = torch.zeros(N_FEATURES, device=device, requires_grad=False)
    sigma_unlocked = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def criterion(pred, target):
        per_feat_mse = ((pred - target) ** 2).mean(dim=0)
        return (torch.exp(-2 * log_sigma) * per_feat_mse + log_sigma).mean()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,} model + {N_FEATURES} log_sigma | '
          f'substeps={args.substeps}')

    # -------------------------------------------------------------- training
    best_val = float('inf')
    patience_count = 0
    log_path = Path(args.output).with_suffix('.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss']
                        + [f'mse_{n}' for n in FEATURE_NAMES])

    for epoch in range(1, args.epochs + 1):
        if not sigma_unlocked and epoch > args.sigma_warmup:
            log_sigma.requires_grad_(True)
            optimizer.add_param_group({'params': [log_sigma]})
            sigma_unlocked = True
            print(f'Epoch {epoch:4d}  sigma unlocked')

        model.train()
        train_loss = 0.0
        feat_mse = np.zeros(N_FEATURES)
        for x, d, cin, eta, y in train_loader:
            x, d, cin, eta, y = (x.to(device), d.to(device), cin.to(device),
                                 eta.to(device), y.to(device))
            optimizer.zero_grad()
            pred, _ = model(x, d, cin, eta_ext=eta)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            with torch.no_grad():
                feat_mse += ((pred - y) ** 2).mean(dim=0).cpu().numpy() * len(x)
        train_loss /= n_train
        feat_mse /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, d, cin, eta, y in val_loader:
                x, d, cin, eta, y = (x.to(device), d.to(device), cin.to(device),
                                     eta.to(device), y.to(device))
                pred, _ = model(x, d, cin, eta_ext=eta)
                val_loss += criterion(pred, y).item() * len(x)
        val_loss /= n_val

        log_writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}']
                            + [f'{m:.6f}' for m in feat_mse])
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save({
                'model_state':      model.state_dict(),
                'log_sigma':        log_sigma.detach().cpu(),
                'scaler':           scaler,
                'doe_min':          doe_min,
                'doe_max':          doe_max,
                'hidden':           args.hidden,
                'n_features':       N_FEATURES,
                'n_input_features': N_INPUT_FEATURES,
                'seq_len':          seq_len,
                'n_doe':            n_doe,
                'n_substeps':       args.substeps,
                'integrator':       model.integrator,
                'arch':             'stripped' if args.stripped else 'primeur',
            }, args.output)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'Early stop at epoch {epoch}  best_val={best_val:.4f}')
                break

    log_file.close()
    print(f'Saved to {args.output}')
    print(f'Training log: {log_path}')


if __name__ == '__main__':
    main()
