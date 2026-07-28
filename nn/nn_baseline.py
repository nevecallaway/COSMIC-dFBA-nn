#!/usr/bin/env python3
"""
Honest baseline: does the plain NN (NO ODE) predict cell-density MAGNITUDE?

Per the end-of-internship plan: train the pure NextDayPredictor (window -> next-day
concentrations directly, no ODE forward step) on SYNTHETIC data alone, hold out a
few WINDOWS per reactor (not whole reactors), and check whether it reproduces the
cell-density trajectories in absolute units. Cell density is the priority target;
titer comes later.

This is the control the group asked for: strip the physics out entirely and see
how well an ordinary supervised NN does on magnitude. If it does well, the ODE was
not doing much for cell density; if it does poorly, the ODE was carrying it.

Usage:
    # make synthetic data with extras first (more windows to train on):
    python generate_synthetic_ode.py --n-extra 3000 --fast --output synthetic_ode.npz
    python nn_baseline.py --data synthetic_ode.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from device_utils import pick_device
from model import (NextDayPredictor, WindowDataset, N_FEATURES, SEQ_LEN, N_DAYS)
from generate_synthetic_ode import WINDOW_FEATURE_INDICES
from evaluate import rollout

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer', 'Glucose',
                 'Glutamine', 'Asparagine', 'Serine', 'Glycine']


def norm_windows(w, scaler, wt):
    n, s, f = w.shape
    scaled = scaler.transform(w.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    return np.concatenate([scaled, wt], axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='synthetic_ode.npz')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='fraction of WINDOWS held out per reactor (not whole reactors)')
    ap.add_argument('--feature', type=int, default=0, help='scored feature (0 = cell density)')
    ap.add_argument('--all-features', action='store_true',
                    help='score + summarize every feature in one run (a table over all 8), '
                         'still plots the --feature one')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    here = Path(__file__).parent
    feat = args.feature

    npz = np.load(args.data, allow_pickle=True)
    windows = npz['windows'].astype(np.float32)          # (n, seq, 8+time)
    targets = npz['targets'].astype(np.float32)          # (n, 8)
    ridx    = npz['window_reactor_idx']
    doe_min = npz['doe_min'].astype(np.float32)
    doe_max = npz['doe_max'].astype(np.float32)
    dsc = doe_max - doe_min; dsc[dsc == 0] = 1.0
    wdoe = ((npz['window_doe'] - doe_min) / dsc).astype(np.float32)
    traj = npz['trajectories'].astype(np.float32)
    n_original = int(npz['n_original'])
    sub = traj[:, :, WINDOW_FEATURE_INDICES].astype(np.float32)   # (N, N_DAYS, 8) absolute

    # --- WINDOW-level split: a few windows per reactor held out, reactors NOT held out ---
    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(windows), bool)
    for r in np.unique(ridx):
        idx = np.where(ridx == r)[0]
        k = max(1, int(round(len(idx) * args.val_frac)))
        val_mask[rng.choice(idx, k, replace=False)] = True
    tr = ~val_mask
    print(f'{len(windows)} windows | train {tr.sum()} | val {val_mask.sum()} '
          f'(>=1 per reactor, window-level)')

    wf, wt = windows[:, :, :N_FEATURES], windows[:, :, N_FEATURES:]
    scaler = MinMaxScaler().fit(np.vstack([wf[tr].reshape(-1, N_FEATURES), targets[tr]]))
    win_tr = norm_windows(wf[tr], scaler, wt[tr])
    win_vl = norm_windows(wf[val_mask], scaler, wt[val_mask])
    y_tr = scaler.transform(targets[tr]).astype(np.float32)
    y_vl = scaler.transform(targets[val_mask]).astype(np.float32)

    n_doe = wdoe.shape[1]
    tr_ds = WindowDataset(win_tr, y_tr, doe=wdoe[tr])
    vl_ds = WindowDataset(win_vl, y_vl, doe=wdoe[val_mask])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    vl_ld = DataLoader(vl_ds, batch_size=args.batch, shuffle=False)

    model = NextDayPredictor(hidden=args.hidden, n_doe=n_doe).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, d, y in tr_ld:
            x, d, y = x.to(device), d.to(device), y.to(device)
            opt.zero_grad()
            loss = ((model(x, d) - y) ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item() * len(x)
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for x, d, y in vl_ld:
                x, d, y = x.to(device), d.to(device), y.to(device)
                vtot += ((model(x, d) - y) ** 2).mean().item() * len(x)
        vl = vtot / len(vl_ds)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0 or ep == 1:
            print(f'epoch {ep:4d}  train={tot/len(tr_ds):.5f}  val={vl:.5f}')
    model.load_state_dict(best_state)

    # --- per-reactor rollout vs synthetic truth, in ABSOLUTE units ---
    fmin = scaler.data_min_.astype(np.float32)
    fsc  = (scaler.data_max_ - scaler.data_min_).astype(np.float32); fsc[fsc == 0] = 1.0
    all_pred = np.zeros((n_original, N_DAYS - SEQ_LEN, N_FEATURES), np.float32)
    for i in range(n_original):
        seed = np.clip((sub[i, :SEQ_LEN] - fmin) / fsc, 0.0, 1.0)
        tcol = (np.arange(SEQ_LEN, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed = np.concatenate([seed, tcol], axis=1)
        doe_i = (npz['doe_params'][i].astype(np.float32) - doe_min) / dsc
        pr = rollout(model, seed, n_steps=N_DAYS - SEQ_LEN, device=device, doe=doe_i)
        all_pred[i] = pr * fsc + fmin                       # absolute units, all features

    def score(f):
        rhos, maes, ratios, ranges = [], [], [], []
        for i in range(n_original):
            pred, real = all_pred[i, :, f], sub[i, SEQ_LEN:, f]
            rmax = real.max() if real.max() > 0 else 1.0
            # relative dynamic range of the TRUE trajectory over the forecast window:
            # near 0 means the feature is flat, so rho is not meaningful there.
            ranges.append(float((real.max() - real.min()) / rmax))
            if np.std(pred) > 0 and np.std(real) > 0:
                rhos.append(float(np.corrcoef(pred, real)[0, 1]))
            maes.append(float(np.mean(np.abs(pred - real)) / rmax))
            ratios.append(float(pred.max() / rmax))
        return np.mean(rhos), np.mean(maes), np.mean(ratios), np.mean(ranges)

    feats = range(N_FEATURES) if args.all_features else [feat]
    print(f'\npure NN (no ODE), rollout vs synthetic truth '
          f'(range = true signal span; rho is not meaningful when range ~ 0):')
    print(f'{"feature":>12} | {"rho":>5} {"normMAE":>8} {"peak":>5} {"range":>6}  note')
    print('-' * 52)
    stats = {}
    for f in feats:
        r, m, p, rng = score(f)
        stats[f] = (r, m, p, rng)
        note = 'flat -> rho N/A' if rng < 0.1 else ''
        print(f'{FEATURE_NAMES[f]:>12} | {r:>5.2f} {m:>8.3f} {p:>5.2f} {rng:>6.2f}  {note}')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    def plot_feature(f):
        r, m, p, rng = stats[f]
        flat = rng < 0.1
        fig, axes = plt.subplots(2, 5, figsize=(20, 7)); axes = axes.flatten()
        for i in range(min(n_original, 10)):
            ax = axes[i]
            ax.plot(np.arange(N_DAYS), sub[i, :, f], 'k-', lw=1.8,
                    label='ODE simulation (synthetic)')
            ax.plot(np.arange(SEQ_LEN, N_DAYS), all_pred[i, :, f], 'r--', lw=2,
                    label='pure NN (no ODE)')
            ax.axvline(SEQ_LEN, color='gray', lw=0.6, ls=':')
            ax.set_title(f'reactor {i}', fontsize=9)
            if flat:
                # anchor the y-axis at 0 so a near-constant metabolite reads as flat,
                # instead of matplotlib auto-zooming trivial <1% wiggle to fill the panel
                top = max(sub[i, :, f].max(), all_pred[i, :, f].max())
                ax.set_ylim(0, top * 1.15)
            if i == 0:
                ax.legend(fontsize=8)
        tag = '  (flat signal, predicted near-exactly; rho N/A)' if flat else ''
        fig.suptitle(f'{FEATURE_NAMES[f]} in ABSOLUTE units: pure NN (no ODE) vs ODE '
                     f'simulation  |  peak={p:.2f}  MAE={m:.3f}  rho={r:.2f}{tag}', y=1.02)
        fig.tight_layout()
        out = here / f'nn_baseline_{FEATURE_NAMES[f]}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')

    # one plot per feature in --all-features, else just the requested one
    for f in feats:
        plot_feature(f)

    # combined single-chart summary: all features, all reactors overlaid
    if args.all_features:
        fig, axes = plt.subplots(2, 4, figsize=(22, 9)); axes = axes.flatten()
        for f in range(N_FEATURES):
            r, m, p, rng = stats[f]
            ax = axes[f]
            for i in range(n_original):
                ax.plot(np.arange(N_DAYS), sub[i, :, f], 'k-', lw=0.8, alpha=0.5)
                ax.plot(np.arange(SEQ_LEN, N_DAYS), all_pred[i, :, f], 'r--',
                        lw=1.0, alpha=0.7)
            if rng < 0.1:
                ax.set_ylim(0, sub[:, :, f].max() * 1.15)     # flat -> anchor at 0
            tag = ' (flat)' if rng < 0.1 else ''
            ax.set_title(f'{FEATURE_NAMES[f]}{tag}  peak={p:.2f} MAE={m:.3f} '
                         f'rho={r:.2f}', fontsize=10)
        axes[0].plot([], [], 'k-', label='ODE simulation'); axes[0].plot([], [], 'r--',
                     label='pure NN (no ODE)'); axes[0].legend(fontsize=8, loc='upper left')
        fig.suptitle('Pure NN (no ODE) vs ODE simulation, all 8 variables in ABSOLUTE '
                     'units, 10 reactors overlaid. Trained on synthetic, window-level holdout.',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        out = here / 'nn_baseline_ALL.png'
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
