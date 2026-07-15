#!/usr/bin/env python3
"""
Train the flux decoder on the REAL reactors (Kimberly's #3): fit the decoder to
the measured concentration shapes so the network learns fluxes that reproduce the
real trajectories. The embedded mass balance handles the perfusion correction
automatically, so this is the perfusion-aware version of Sarat's slope-to-rate
method (dC/dt has the F(cin - C) term subtracted before the flux is inferred).

Denormalization: data_2 is measured but normalized (per reactor/component / max).
We recover approximate physical units as

    real_phys = data2_norm * ODE_max

where ODE_max is the synthetic trajectory's peak for that reactor/component. The
SHAPE is real (from data_2); the SCALE is our best guess (raw data_2 is
proprietary). Non-model components keep their ODE values (unused by the model).

Modes:
  --init model_flux.pt   transfer: start from the synthetic-pretrained model (#2)
  (no --init)            from scratch on real data only (#3)

LORO: --holdout excludes reactors from training so held-out ones can be tested
      (evaluate.py --model model_real.pt --data <ode_npz> --eval-reactor H).

Usage:
    python train_real.py --holdout 3 4
    python train_real.py --init model_flux.pt --holdout 3 4
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from model_primeur import FluxDecoder, N_FEATURES, N_INPUT_FEATURES, SEQ_LEN
from generate_synthetic_ode import build_windows
from train_sample import FluxWindowDataset
from real_data import denormalize_data2


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--ode-data', default=str(here / 'synthetic_ode.npz'),
                    help='ODE npz for the physical scale, DoE and feed')
    ap.add_argument('--init', default=None,
                    help='Pretrained checkpoint to start from (transfer). Omit = from scratch')
    ap.add_argument('--output', default=str(here / 'model_real.pt'))
    ap.add_argument('--holdout', type=int, nargs='+', default=[])
    ap.add_argument('--epochs', type=int,   default=300)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--batch',  type=int,   default=32)
    ap.add_argument('--hidden', type=int,   default=32)   # small: few real windows
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--substeps', type=int, default=50)
    ap.add_argument('--freeze-conv', action='store_true',
                    help='Freeze the conv feature-extractor; fine-tune only attention + head '
                         '(prevents overfitting the few real windows, for transfer)')
    ap.add_argument('--stripped', action='store_true',
                    help='Use the low-capacity model_stripped decoder (1 conv layer, '
                         'linear head) instead of the full en Primeur body')
    ap.add_argument('--phase', action='store_true',
                    help='Phase-driven titer washout: eta = production fraction f(t) per '
                         'reactor/day (from data_2 phases) instead of the day-8 switch')
    args = ap.parse_args()

    if args.stripped:
        from model_stripped import FluxDecoder as ModelClass
        print('Using stripped (low-capacity) decoder')
    else:
        ModelClass = FluxDecoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    npz = np.load(args.ode_data, allow_pickle=True)
    ode_traj   = npz['trajectories'].astype(np.float32)
    doe_params = npz['doe_params'].astype(np.float32)
    cin_params = npz['cin_params'].astype(np.float32)
    n_original = int(npz['n_original'])
    doe_min, doe_max = npz['doe_min'].astype(np.float32), npz['doe_max'].astype(np.float32)

    # ---- real physical trajectories from data_2 ----
    real = denormalize_data2(here / 'data' / 'data_2.csv', ode_traj, n_original)
    print(f'Denormalized {n_original} real reactors from data_2 (ODE-scaled).')

    # Phase-driven eta uses the real per-reactor production fraction f(t).
    phase_traj = npz['phases'].astype(np.float32)[:n_original] if args.phase else None
    windows, targets, wdoe, wcin, weta, ridx = build_windows(
        real, doe_params=doe_params[:n_original], cin_params=cin_params[:n_original],
        phase_traj=phase_traj)

    hold = set(args.holdout)
    keep = np.array([r not in hold for r in ridx])
    print(f'Training on real reactors {[i for i in range(n_original) if i not in hold]}; '
          f'holding out {sorted(hold)}')
    windows, targets, wdoe, wcin, weta = (windows[keep], targets[keep], wdoe[keep],
                                          wcin[keep], weta[keep])

    win_feats, win_time = windows[:, :, :N_FEATURES], windows[:, :, N_FEATURES:]

    # ---- scaler: reuse pretrained (transfer) or fit on real (from scratch) ----
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        scaler = ckpt['scaler']
    else:
        scaler = MinMaxScaler().fit(np.vstack([win_feats.reshape(-1, N_FEATURES), targets]))

    n, s, f = win_feats.shape
    win_scaled = scaler.transform(win_feats.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    windows_n = np.concatenate([win_scaled, win_time], axis=2)
    targets_n = scaler.transform(targets).astype(np.float32)
    doe_scale = doe_max - doe_min; doe_scale[doe_scale == 0] = 1.0
    wdoe_n = ((wdoe - doe_min) / doe_scale).astype(np.float32)

    ds = FluxWindowDataset(windows_n, wdoe_n, wcin, weta, targets_n)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    print(f'Real training windows: {len(ds)}')

    # ---- model ----
    hidden = args.hidden
    n_doe = wdoe_n.shape[1]
    if args.init:
        hidden = ckpt.get('hidden', hidden)
    model = ModelClass(hidden=hidden, n_doe=n_doe,
                       n_input_features=N_INPUT_FEATURES, n_substeps=args.substeps).to(device)
    if args.init:
        model.load_state_dict(ckpt['model_state'])
        print(f'Transfer: initialized from {args.init}')
    model.set_scaler(scaler)

    if args.freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False
        print('Froze conv feature-extractor; fine-tuning attention + head only')

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in trainable)
    print(f'Trainable parameters: {n_params:,}  hidden={hidden}')

    # ---- train (plain MSE; small model + weight decay for the tiny dataset) ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, d, cin, eta, y in loader:
            x, d, cin, eta, y = (x.to(device), d.to(device), cin.to(device),
                                 eta.to(device), y.to(device))
            opt.zero_grad()
            pred, _ = model(x, d, cin, eta_ext=eta)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        if epoch % 25 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  loss={tot / len(ds):.5f}')

    torch.save({
        'model_state': model.state_dict(), 'scaler': scaler,
        'doe_min': doe_min, 'doe_max': doe_max, 'hidden': hidden,
        'n_features': N_FEATURES, 'n_input_features': N_INPUT_FEATURES,
        'seq_len': SEQ_LEN, 'n_doe': n_doe, 'n_substeps': args.substeps,
        'arch': 'stripped' if args.stripped else 'primeur',
        'phase': args.phase,
    }, args.output)
    print(f'Saved to {args.output}')
    print(f'Evaluate held-out: python evaluate.py --model {Path(args.output).name} '
          f'--data {Path(args.ode_data).name} --eval-reactor {sorted(hold)[0] if hold else 0}')


if __name__ == '__main__':
    main()
