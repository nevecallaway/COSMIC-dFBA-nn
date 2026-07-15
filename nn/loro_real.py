#!/usr/bin/env python3
"""
Leakage-free LORO for the STRIPPED model trained DIRECTLY on real reactors
(Kimberly's "old-fashioned, no synthetic" direction).

For each real reactor i: train the low-capacity decoder (model_stripped) from
scratch on the other 9 real reactors (train_real.py --stripped --holdout i, no
--init, so NO synthetic pretraining), then predict reactor i from its own early
real days. Nothing about reactor i touches its model. This is the direct
counterpart to plot_loro.py (which trains on synthetic); compare the two plots.

Assembled plot per reactor:
    blue  = ODE (synthetic reference), gray-scaled to its own peak
    red   = held-out stripped model prediction (mean + band), scaled to REAL peak
    black = real (denormalized data_2), scaled to its own peak
Red above 1.0 = the model over-predicts titer for that held-out reactor. No clamp.

Usage:
    python loro_real.py                 # full 10-fold, titer
    python loro_real.py --single-step   # one point per day from the real window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from model import FEATURE_INDICES, N_DAYS, SEQ_LEN
from real_data import denormalize_data2, REACTOR_IDS
from plot_loro import run, load_model, ensemble, single_step, forecast


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', type=int, default=2, help='0-7 (default 2=Titer)')
    ap.add_argument('--hidden', type=int, default=16, help='stripped body width')
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN,
                    help='Input window length in days (default 6)')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--single-step', action='store_true',
                    help='Teacher-forced: predict each day from the REAL previous window '
                         '(one-day-ahead accuracy, not a forecast)')
    ap.add_argument('--band', action='store_true',
                    help='Ensemble of all windows with the min/max band')
    ap.add_argument('--phase', action='store_true',
                    help='Phase-driven titer washout (eta = f(t)) instead of the day-8 switch')
    ap.add_argument('--phase-threshold', type=float, default=None,
                    help='With --phase, step eta = 1 once f crosses this value (e.g. 0.5)')
    args = ap.parse_args()

    feat = args.feature
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    py = sys.executable

    # ODE npz once: physical scale + DoE + feed for train_real (no extras needed).
    npz_path = here / 'loro_real.npz'
    gen_cmd = [py, here / 'generate_synthetic_ode.py', '--n-extra', 0,
               '--output', npz_path, '--fast']
    if args.phase:
        gen_cmd.append('--phase')
        if args.phase_threshold is not None:
            gen_cmd += ['--phase-threshold', args.phase_threshold]
    run(gen_cmd)
    data = np.load(npz_path, allow_pickle=True)
    ode_traj   = data['trajectories'].astype(np.float32)
    cin_params = data['cin_params'].astype(np.float32)
    doe_params = data['doe_params'].astype(np.float32)
    phases     = data['phases'].astype(np.float32)
    n_original = int(data['n_original'])
    if args.phase and args.phase_threshold is not None:
        phases = (phases > args.phase_threshold).astype(np.float32)

    # Real (denormalized data_2) trajectories = the seeds and the ground truth.
    real = denormalize_data2(here / 'data' / 'data_2.csv', ode_traj, n_original)
    real_sub = real[:, :, FEATURE_INDICES].astype(np.float32)          # (10, N_DAYS, 8)
    ode_sub  = ode_traj[:, :, FEATURE_INDICES].astype(np.float32)      # synthetic ref

    predict = single_step if args.single_step else (ensemble if args.band else forecast)
    preds = {}
    for i in range(n_original):
        pt = here / f'loro_real_{i}.pt'
        tr_cmd = [py, here / 'train_real.py', '--stripped', '--holdout', i,
                  '--ode-data', npz_path, '--output', pt,
                  '--hidden', args.hidden, '--epochs', args.epochs,
                  '--seq-len', args.seq_len]
        if args.phase:
            tr_cmd.append('--phase')
            if args.phase_threshold is not None:
                tr_cmd += ['--phase-threshold', args.phase_threshold]
        run(tr_cmd)
        model, fmin, scale, dmin, dmax = load_model(pt, device)
        dsc = dmax - dmin; dsc[dsc == 0] = 1.0
        preds[i] = predict(model, real_sub[i], cin_params[i],
                           (doe_params[i] - dmin) / dsc,
                           fmin, scale, feat, args.seq_len, device,
                           phase=phases[i] if args.phase else None)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(20, 7)); axes = axes.flatten()
    for i in range(min(n_original, 10)):
        rtrace = real_sub[i, :, feat]; rmax = rtrace.max() if rtrace.max() > 0 else 1.0
        otrace = ode_sub[i, :, feat];  omax = otrace.max() if otrace.max() > 0 else 1.0
        ax = axes[i]
        ax.plot(np.arange(N_DAYS), otrace / omax, 'b-', lw=1.5, alpha=0.6,
                label='synthetic (ODE)')
        d, m, lo, hi = preds[i]
        if lo is not None:
            ax.fill_between(d, lo / rmax, hi / rmax, color='red', alpha=0.2, lw=0)
        ax.plot(d, m / rmax, 'r--', lw=2, label='predicted (held-out stripped)')
        ax.plot(np.arange(N_DAYS), rtrace / rmax, 'k:', lw=1.8, label='real (data_2)')
        ax.axvline(args.seq_len, color='gray', lw=0.6, ls=':')
        ax.set_title(REACTOR_IDS[i], fontsize=9); ax.set_ylim(-0.05, 1.6)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Titer, STRIPPED trained on REAL only (leakage-free): red = model '
                 'that held that reactor out; red > 1 = overshoot vs real', y=1.02)
    fig.tight_layout()
    out = here / 'loro_real_Titer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
