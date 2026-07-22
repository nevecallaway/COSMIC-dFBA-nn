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

--pretrain adds a leakage-free synthetic prior: for each fold, generate synthetic
with reactor i excluded from the donor pool, pretrain the stripped model on it,
then fine-tune on the 9 real reactors. Tests whether the synthetic flux-range
prior reins in the reactors that overshoot (Kimberly's suggestion).

Usage:
    python loro_real.py                 # full 10-fold forecast, real-only
    python loro_real.py --pretrain      # synthetic-pretrained then fine-tuned
    python loro_real.py --single-step   # one point per day from the real window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from device_utils import pick_device
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
    ap.add_argument('--pretrain', action='store_true',
                    help='Pretrain the stripped model on synthetic (held-out reactor '
                         'excluded from donors) before fine-tuning on the 9 real reactors. '
                         'Tests whether a synthetic flux-range prior reins in the overshoots.')
    ap.add_argument('--n-extra', type=int, default=3000,
                    help='Synthetic reactors for the pretrain step (--pretrain only)')
    ap.add_argument('--val-reactors', type=int, default=2,
                    help='reactors held out of each fold for validation curves')
    ap.add_argument('--patience', type=int, default=0,
                    help='early-stopping patience per fold (0 = fixed epochs, log only)')
    ap.add_argument('--curve-dir', default=None,
                    help='write per-fold train/val curves as CSV here')
    ap.add_argument('--gap-stop', type=float, default=None,
                    help='stop each fold when val_loss exceeds this multiple of '
                         'train_loss (val "lifts off" train)')
    ap.add_argument('--rollout-train', action='store_true',
                    help='Train each fold on the autoregressive forecast (matches how '
                         'we score it) instead of one-day-ahead')
    ap.add_argument('--eta-day', type=int, default=None,
                    help='day the titer washout switches on (default 8). The predicted '
                         'peak is pinned to this day, so try 7 if the real peak is a '
                         'day earlier than the prediction')
    args = ap.parse_args()

    if args.eta_day is not None:
        import model_primeur, generate_synthetic_ode as _gen
        model_primeur.ETA_SWITCH_DAY = args.eta_day
        _gen.ETA_SWITCH_DAY = args.eta_day
        print(f'Eta switch day set to {args.eta_day} (default 8)')

    feat = args.feature
    device = pick_device()
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
        ode_for_train = npz_path      # npz used for real trajectories + DoE scale
        init_args = []
        if args.pretrain:
            # Leakage-free synthetic prior: generate with reactor i excluded from the
            # donor pool, pretrain the stripped model, then fine-tune (--init) on real.
            # Fine-tune uses the SAME npz so the DoE normalization is consistent.
            pre_npz = here / f'loro_real_pre_{i}.npz'
            pre_pt  = here / f'loro_real_pre_{i}.pt'
            gen_i = [py, here / 'generate_synthetic_ode.py', '--holdout', i,
                     '--n-extra', args.n_extra, '--output', pre_npz, '--fast',
                     '--rate-mix', 0.2, '--rate-scale', 0.1]
            if args.phase:
                gen_i.append('--phase')
                if args.phase_threshold is not None:
                    gen_i += ['--phase-threshold', args.phase_threshold]
            run(gen_i)
            run([py, here / 'train_sample.py', '--stripped', '--data', pre_npz,
                 '--output', pre_pt, '--hidden', args.hidden, '--batch', 256])
            ode_for_train = pre_npz
            init_args = ['--init', pre_pt]

        tr_cmd = [py, here / 'train_real.py', '--stripped', '--holdout', i,
                  '--ode-data', ode_for_train, '--output', pt,
                  '--hidden', args.hidden, '--epochs', args.epochs,
                  '--seq-len', args.seq_len,
                  '--val-reactors', args.val_reactors] + init_args
        if args.eta_day is not None:
            tr_cmd += ['--eta-day', args.eta_day]
        if args.rollout_train:
            tr_cmd.append('--rollout')
        if args.gap_stop is not None:
            tr_cmd += ['--gap-stop', args.gap_stop]
        if args.patience:
            tr_cmd += ['--patience', args.patience]
        if args.curve_dir:
            Path(args.curve_dir).mkdir(parents=True, exist_ok=True)
            tr_cmd += ['--curve-csv', str(Path(args.curve_dir) / f'curve_fold{i}.csv')]
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

    # ---- metrics on the forecast days (predicted vs real) ----
    # shape = trajectory correlation (scale-free); magnitude = MAE and peak height,
    # both normalized by the real peak (same scaling as the plot, so peak ratio > 1
    # means overshoot). Absolute titer scale is a denorm guess, hence the normalization.
    print(f'\n{"reactor":>8} | {"shape rho":>9} {"norm MAE":>9} {"peak ratio":>10}  verdict')
    print('-' * 56)
    rhos, maes, ratios = [], [], []
    for i in range(n_original):
        d, m = preds[i][0], preds[i][1]
        r = real_sub[i, d, feat]
        rmax = real_sub[i, :, feat].max(); rmax = rmax if rmax > 0 else 1.0
        rho = (np.corrcoef(m, r)[0, 1]
               if len(m) > 1 and np.std(m) > 0 and np.std(r) > 0 else float('nan'))
        mae = float(np.mean(np.abs(m - r)) / rmax)
        ratio = float(m.max() / rmax)
        verdict = 'overshoot' if ratio > 1.1 else ('undershoot' if ratio < 0.9 else 'on-target')
        rhos.append(rho); maes.append(mae); ratios.append(ratio)
        print(f'{REACTOR_IDS[i]:>8} | {rho:>9.2f} {mae:>9.3f} {ratio:>10.2f}  {verdict}')
    n_over  = sum(x > 1.1 for x in ratios)
    n_under = sum(x < 0.9 for x in ratios)
    print('-' * 56)
    print(f'{"MEAN":>8} | {np.nanmean(rhos):>9.2f} {np.mean(maes):>9.3f} {np.mean(ratios):>10.2f}'
          f'  {n_over} over / {n_under} under / {n_original - n_over - n_under} on-target')

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
    train_desc = ('synthetic-pretrained then fine-tuned on REAL'
                  if args.pretrain else 'REAL only')
    fig.suptitle(f'Titer, STRIPPED {train_desc} (leakage-free): red = model that held '
                 f'that reactor out; red > 1 = overshoot vs real', y=1.02)
    fig.tight_layout()
    out = here / 'loro_real_Titer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
