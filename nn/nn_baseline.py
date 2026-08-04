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
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN,
                    help='input window length in days (rebuilds windows; try shorter)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conv-layers', type=int, default=3,
                    help='number of 1D conv layers (capacity vs speed ablation; the '
                         'speed benchmark takes the same flag so the two line up)')
    ap.add_argument('--tag', default='',
                    help='suffix appended to saved figure names, e.g. --tag v2 -> '
                         'nn_baseline_r2_Titer_v2.png (avoids overwriting prior runs)')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    here = Path(__file__).parent
    fig_tag = f'_{args.tag}' if args.tag else ''
    feat = args.feature
    seq_len = args.seq_len

    npz = np.load(args.data, allow_pickle=True)
    doe_params = npz['doe_params'].astype(np.float32)
    cin_params = npz['cin_params'].astype(np.float32)
    doe_min = npz['doe_min'].astype(np.float32)
    doe_max = npz['doe_max'].astype(np.float32)
    dsc = doe_max - doe_min; dsc[dsc == 0] = 1.0
    traj = npz['trajectories'].astype(np.float32)
    n_original = int(npz['n_original'])
    sub = traj[:, :, WINDOW_FEATURE_INDICES].astype(np.float32)   # (N, N_DAYS, 8) absolute

    # Rebuild windows from the EXTRA reactors at the requested seq_len (originals
    # reserved for eval). This is what --seq-len varies.
    from generate_synthetic_ode import build_windows
    windows, targets, wdoe_raw, _wcin, _weta, ridx = build_windows(
        traj[n_original:], doe_params=doe_params[n_original:],
        cin_params=cin_params[n_original:], seq_len=seq_len)
    windows = windows.astype(np.float32); targets = targets.astype(np.float32)
    wdoe = ((wdoe_raw - doe_min) / dsc).astype(np.float32)
    print(f'window length = {seq_len} days')

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

    model = NextDayPredictor(hidden=args.hidden, n_conv_layers=args.conv_layers,
                             n_doe=n_doe).to(device)
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
    all_pred = np.zeros((n_original, N_DAYS - seq_len, N_FEATURES), np.float32)
    for i in range(n_original):
        seed = np.clip((sub[i, :seq_len] - fmin) / fsc, 0.0, 1.0)
        tcol = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed = np.concatenate([seed, tcol], axis=1)
        doe_i = (npz['doe_params'][i].astype(np.float32) - doe_min) / dsc
        pr = rollout(model, seed, n_steps=N_DAYS - seq_len, device=device, doe=doe_i)
        all_pred[i] = pr * fsc + fmin                       # absolute units, all features

    def score(f):
        rhos, maes, ratios, ranges = [], [], [], []
        for i in range(n_original):
            pred, real = all_pred[i, :, f], sub[i, seq_len:, f]
            rmax = real.max() if real.max() > 0 else 1.0
            # relative dynamic range of the TRUE trajectory over the forecast window:
            # near 0 means the feature is flat, so rho and R2 are not meaningful.
            ranges.append(float((real.max() - real.min()) / rmax))
            if np.std(pred) > 0 and np.std(real) > 0:
                rhos.append(float(np.corrcoef(pred, real)[0, 1]))
            maes.append(float(np.mean(np.abs(pred - real)) / rmax))
            ratios.append(float(pred.max() / rmax))
        # R2 (coefficient of determination), pooled over ALL reactors' forecast
        # points: 1 - SS_res/SS_tot with the mean as baseline. Unitless, the
        # standard regression metric. Pooling includes cross-reactor level spread,
        # so it rewards getting each reactor's overall level right. NOTE: like rho,
        # meaningless for flat signals (SS_tot ~ 0 -> large negative); only reported
        # for dynamic variables.
        pf = all_pred[:, :, f].ravel()
        tf = sub[:n_original, seq_len:, f].ravel()   # match all_pred's eval reactors
        ss_tot = float(np.sum((tf - tf.mean()) ** 2))
        r2 = 1.0 - float(np.sum((pf - tf) ** 2)) / ss_tot if ss_tot > 0 else float('nan')
        return r2, np.mean(maes), np.mean(ratios), np.mean(rhos), np.mean(ranges)

    feats = range(N_FEATURES) if args.all_features else [feat]
    print(f'\npure NN (no ODE), rollout vs synthetic truth '
          f'(range = true signal span; R2 and rho are not meaningful when range ~ 0):')
    print(f'{"feature":>12} | {"R2":>6} {"peak":>5} {"normMAE":>8} {"rho":>5} {"range":>6}  note')
    print('-' * 60)
    stats = {}
    for f in feats:
        r2, m, p, rho, rng = score(f)
        stats[f] = (r2, m, p, rho, rng)
        flat = rng < 0.1
        note = 'flat -> R2/rho N/A' if flat else ''
        r2s = f'{"n/a":>6}' if flat else f'{r2:>6.2f}'
        print(f'{FEATURE_NAMES[f]:>12} | {r2s} {p:>5.2f} {m:>8.3f} {rho:>5.2f} {rng:>6.2f}  {note}')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    def plot_feature(f):
        r2, m, p, rho, rng = stats[f]
        flat = rng < 0.1
        fig, axes = plt.subplots(2, 5, figsize=(20, 7)); axes = axes.flatten()
        for i in range(min(n_original, 10)):
            ax = axes[i]
            # very light band over the forecast region so it reads as "the part the
            # NN predicts on its own", no under-curve fill (that only meant something
            # for AUC; the metric now is R2 in the title).
            ax.axvspan(seq_len, N_DAYS - 1, color='0.5', alpha=0.06)
            ax.plot(np.arange(N_DAYS), sub[i, :, f], 'k-', lw=1.8,
                    label='ODE simulation (synthetic)')
            ax.plot(np.arange(seq_len, N_DAYS), all_pred[i, :, f], 'r--', lw=2,
                    label='pure NN (no ODE)')
            ax.axvline(seq_len, color='gray', lw=0.8, ls=':')
            ax.set_title(f'reactor {i}', fontsize=9)
            if flat:
                # anchor the y-axis at 0 so a near-constant metabolite reads as flat,
                # instead of matplotlib auto-zooming trivial <1% wiggle to fill the panel
                top = max(sub[i, :, f].max(), all_pred[i, :, f].max())
                ax.set_ylim(0, top * 1.15)
            if i == 0:
                ax.legend(fontsize=8)
        tag = '  (flat signal, predicted near-exactly; R2/rho N/A)' if flat else ''
        r2str = 'n/a' if flat else f'{r2:.2f}'
        fig.suptitle(f'{FEATURE_NAMES[f]} in ABSOLUTE units: pure NN (no ODE) vs ODE '
                     f'simulation  |  peak={p:.2f}  R2={r2str}  MAE={m:.3f}  rho={rho:.2f}{tag}',
                     y=1.02)
        fig.tight_layout()
        out = here / f'nn_baseline_r2_{FEATURE_NAMES[f]}{fig_tag}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')

    # one readable per-feature plot (2x5 reactor grid) each -- the presentation style
    for f in feats:
        plot_feature(f)

    # parity plot: predicted vs true, all forecast points, with the y=x diagonal.
    # The single simplest "how does the model do" view -- points on the diagonal
    # are perfect, and R2 is exactly how tightly they hug it.
    def plot_parity():
        fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.flatten()
        for f in range(N_FEATURES):
            ax = axes[f]
            x = sub[:n_original, seq_len:, f].ravel()      # true (ODE)
            y = all_pred[:, :, f].ravel()                  # predicted (NN)
            r2, m, p, rho, rng = stats[f]
            lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            lo -= pad; hi += pad
            ax.plot([lo, hi], [lo, hi], color='0.5', lw=1, ls='--', zorder=1)
            ax.scatter(x, y, s=9, c='#6B2333', alpha=0.30, edgecolors='none', zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
            r2str = 'n/a' if rng < 0.1 else f'{r2:.2f}'
            ax.set_title(f'{FEATURE_NAMES[f]}  (R2={r2str})', fontsize=11)
            ax.set_xlabel('true (ODE)', fontsize=8)
            ax.set_ylabel('predicted (NN)', fontsize=8)
        fig.suptitle('Predicted vs true, forecast window, all reactors -- points on '
                     'the dashed diagonal are perfect', y=1.0, fontsize=13)
        fig.tight_layout()
        out = here / f'nn_baseline_parity{fig_tag}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')

    if args.all_features:
        plot_parity()


if __name__ == '__main__':
    main()
