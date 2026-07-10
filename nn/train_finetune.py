#!/usr/bin/env python3
"""
Transfer learning: fine-tune the synthetic-pretrained decoder on the real
reactors, holding out a stratified subset for testing.

Loads model_flux.pt (pretrained on synthetic), builds windows from the physical
real-reactor trajectories (excluding the held-out reactors), and continues
training at a low learning rate. Then evaluate on the held-out reactors:

    python evaluate.py --model model_finetuned.pt --eval-reactor 3

IMPORTANT on "real data": the real-reactor trajectories used here are the
ODE-generated ones in the npz (physical units, SAME physics as the synthetic
data). This tests whether fine-tuning on a subset of real reactors helps predict
held-out ones. True wet-lab fine-tuning would use the raw MEASURED concentrations
and feed schedule, which we do not have in physical units (data_2 is normalized).
That needs raw data from Sarat.

Usage:
    python train_finetune.py --holdout 3 4     # fine-tune on the other 8 reactors
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from model_primeur import FluxDecoder, N_FEATURES, N_INPUT_FEATURES, SEQ_LEN
from generate_synthetic_ode import build_windows
from train_sample import FluxWindowDataset


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   default=str(here / 'model_flux.pt'),
                    help='Pretrained (synthetic) checkpoint to start from')
    ap.add_argument('--data',    default=str(here / 'synthetic_ode.npz'))
    ap.add_argument('--output',  default=str(here / 'model_finetuned.pt'))
    ap.add_argument('--holdout', type=int, nargs='+', default=[],
                    help='Real reactor indices to hold out (not fine-tuned on)')
    ap.add_argument('--epochs',  type=int,   default=50)
    ap.add_argument('--lr',      type=float, default=1e-4)   # low: fine-tuning
    ap.add_argument('--batch',   type=int,   default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- load pretrained model + its scaler / doe stats ----
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    scaler   = ckpt['scaler']
    doe_min, doe_max = ckpt['doe_min'], ckpt['doe_max']
    feat_min = scaler.data_min_.astype(np.float32)
    feat_rng = scaler.data_range_.astype(np.float32); feat_rng[feat_rng == 0] = 1.0

    model = FluxDecoder(hidden=ckpt.get('hidden', 64), n_doe=ckpt.get('n_doe', 3),
                        n_input_features=ckpt.get('n_input_features', N_INPUT_FEATURES),
                        n_substeps=int(ckpt['n_substeps'])).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f'Loaded pretrained {args.model}')

    # ---- build windows from the REAL reactors, excluding the holdout set ----
    npz = np.load(args.data, allow_pickle=True)
    trajectories = npz['trajectories'].astype(np.float32)
    doe_params   = npz['doe_params'].astype(np.float32)
    cin_params   = npz['cin_params'].astype(np.float32)
    n_original   = int(npz['n_original'])
    seq_len      = int(npz['seq_len']) if 'seq_len' in npz else SEQ_LEN

    hold = set(args.holdout)
    ft_idx = [i for i in range(n_original) if i not in hold]
    print(f'Fine-tune on real reactors {ft_idx}; holding out {sorted(hold)}')

    windows, targets, wdoe, wcin, _ = build_windows(
        trajectories[ft_idx], doe_params=doe_params[ft_idx],
        cin_params=cin_params[ft_idx], seq_len=seq_len)

    # normalize with the PRETRAINED scaler (do not refit)
    win_feats = windows[:, :, :N_FEATURES]
    win_time  = windows[:, :, N_FEATURES:]
    n, s, f = win_feats.shape
    win_scaled = scaler.transform(win_feats.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    windows_n = np.concatenate([win_scaled, win_time], axis=2)
    targets_n = scaler.transform(targets).astype(np.float32)
    doe_scale = doe_max - doe_min; doe_scale[doe_scale == 0] = 1.0
    wdoe_n = ((wdoe - doe_min) / doe_scale).astype(np.float32)

    ds = FluxWindowDataset(windows_n, wdoe_n, wcin, targets_n)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    print(f'Fine-tune windows: {len(ds)}')

    # ---- fine-tune (low LR, plain MSE) ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, d, cin, y in loader:
            x, d, cin, y = x.to(device), d.to(device), cin.to(device), y.to(device)
            opt.zero_grad()
            pred, _ = model(x, d, cin)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  ft_loss={tot / len(ds):.5f}')

    torch.save({**{k: ckpt[k] for k in ckpt if k != 'model_state'},
                'model_state': model.state_dict()}, args.output)
    print(f'Saved fine-tuned model to {args.output}')
    print(f'Evaluate the held-out reactors, e.g.:  '
          f'python evaluate.py --model {Path(args.output).name} '
          f'--eval-reactor {sorted(hold)[0] if hold else 0}')


if __name__ == '__main__':
    main()
