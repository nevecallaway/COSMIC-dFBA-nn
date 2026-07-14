#!/usr/bin/env python3
"""
Leakage-free three-way titer plot.

Each reactor's predicted line comes from a model that HELD THAT REACTOR OUT of
synthetic training (leave-one-reactor-out), so no reactor is predicted by a model
that saw its rates. For each fold: generate synthetic with the fold's reactors
excluded from the donor pool, train, then predict those held-out reactors.

Assembled plot per reactor: blue = ODE trajectory, red = held-out model
prediction (mean over overlapping windows + min/max band), black = real (data_2).

Usage:
    python plot_loro.py                                  # full LORO (10 models; slow, use GPU)
    python plot_loro.py --folds "0 1 2 3 4" "5 6 7 8 9"  # 2 folds (faster, train on 5 each)
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import FEATURE_INDICES, N_FEATURES, SEQ_LEN, N_DAYS
from model_primeur import FluxDecoder
from evaluate import rollout

REACTOR_IDS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
               'R0006', 'R0008', 'R0010', 'R0011', 'R0012']


def run(cmd):
    print('>>', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); sys.exit('command failed')
    return r.stdout


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('arch') == 'stripped':
        from model_stripped import FluxDecoder as ModelClass
    else:
        ModelClass = FluxDecoder
    m = ModelClass(hidden=ck.get('hidden', 64), n_doe=ck.get('n_doe', 3),
                   n_input_features=ck.get('n_input_features', N_FEATURES + 1),
                   n_substeps=int(ck['n_substeps'])).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    sc = ck['scaler']
    scale = sc.data_range_.astype(np.float32); scale[scale == 0] = 1.0
    return m, sc.data_min_.astype(np.float32), scale, ck['doe_min'], ck['doe_max']


def single_step(model, sub_i, cin_i, doe_i, fmin, scale, feat, seq_len, device):
    """One prediction per day from the REAL previous seq_len days (teacher-forced,
    no rollout). Returns (days, values, None, None) -- no band."""
    days, vals = [], []
    for d in range(seq_len, N_DAYS):
        seed = np.clip((sub_i[d - seq_len:d] - fmin) / scale, 0.0, 1.0)
        tcol = (np.arange(d - seq_len, d, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed = np.concatenate([seed, tcol], axis=1)
        pr = rollout(model, seed, n_steps=1, device=device, doe=doe_i,
                     cin=cin_i, is_decoder=True)
        vals.append(pr[0, feat] * scale[feat] + fmin[feat])
        days.append(d)
    return np.array(days), np.array(vals), None, None


def ensemble(model, sub_i, cin_i, doe_i, fmin, scale, feat, seq_len, device):
    """Predict each future day from every seed window; return per-day mean/min/max."""
    per_day = defaultdict(list)
    for s in range(0, N_DAYS - seq_len):
        seed = np.clip((sub_i[s:s + seq_len] - fmin) / scale, 0.0, 1.0)
        tcol = (np.arange(s, s + seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed = np.concatenate([seed, tcol], axis=1)
        ns = N_DAYS - (s + seq_len)
        if ns <= 0:
            continue
        pr = rollout(model, seed, n_steps=ns, device=device, doe=doe_i,
                     cin=cin_i, is_decoder=True)
        pp = pr[:, feat] * scale[feat] + fmin[feat]
        for k in range(ns):
            per_day[s + seq_len + k].append(pp[k])
    days = sorted(per_day)
    return (np.array(days),
            np.array([np.mean(per_day[d]) for d in days]),
            np.array([np.min(per_day[d]) for d in days]),
            np.array([np.max(per_day[d]) for d in days]))


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', default=[str(i) for i in range(10)],
                    help='Holdout groups, each a space-joined index string. Default = full LORO.')
    ap.add_argument('--feature', type=int, default=2, help='0-7 (default 2=Titer)')
    ap.add_argument('--n-extra', type=int, default=3000)
    ap.add_argument('--single-step', action='store_true',
                    help='Predict each day from the REAL previous window (one point per '
                         'day, no rollout, no band) instead of autoregressive forecast')
    args = ap.parse_args()

    feat = args.feature
    comp = FEATURE_INDICES[feat]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    py = sys.executable
    folds = [[int(x) for x in f.split()] for f in args.folds]

    preds, ref_npz = {}, None
    for fold in folds:
        H = [str(i) for i in fold]
        npz = here / f'loroplot_{"_".join(H)}.npz'
        pt  = here / f'loroplot_{"_".join(H)}.pt'
        run([py, here / 'generate_synthetic_ode.py', '--holdout', *H, '--output', npz,
             '--n-extra', args.n_extra, '--rate-mix', 0.2, '--rate-scale', 0.1, '--fast'])
        run([py, here / 'train_sample.py', '--data', npz, '--output', pt, '--batch', 256])
        if ref_npz is None:
            ref_npz = npz
        model, fmin, scale, dmin, dmax = load_model(pt, device)
        data = np.load(npz, allow_pickle=True)
        sub  = data['trajectories'].astype(np.float32)[:, :, FEATURE_INDICES]
        cinp = data['cin_params'].astype(np.float32)
        doep = data['doe_params'].astype(np.float32)
        seq_len = int(data['seq_len']) if 'seq_len' in data else SEQ_LEN
        dsc = dmax - dmin; dsc[dsc == 0] = 1.0
        predict = single_step if args.single_step else ensemble
        for i in fold:
            preds[i] = predict(model, sub[i], cinp[i], (doep[i] - dmin) / dsc,
                               fmin, scale, feat, seq_len, device)

    # references (ODE + real) from any npz; the 10 real reactors are identical across folds
    ref = np.load(ref_npz, allow_pickle=True)
    sub = ref['trajectories'].astype(np.float32)[:, :, FEATURE_INDICES]
    n_original = int(ref['n_original'])
    df2 = pd.read_csv(here / 'data' / 'data_2.csv', skiprows=1)
    df2.columns = ['Vessel', 'Time', 'Phase'] + [f'C{i}' for i in range(25)]
    df2['Time'] = pd.to_numeric(df2['Time'], errors='coerce'); df2 = df2.dropna(subset=['Time'])

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(20, 7)); axes = axes.flatten()
    for i in range(min(n_original, 10)):
        synth = sub[i, :, feat]; smax = synth.max() if synth.max() > 0 else 1.0
        rdf = df2[df2['Vessel'] == REACTOR_IDS[i]].sort_values('Time')
        real = rdf[f'C{comp}'].to_numpy(float); rdays = rdf['Time'].to_numpy(float)
        rmax = real.max() if real.size and real.max() > 0 else 1.0
        ax = axes[i]
        ax.plot(np.arange(N_DAYS), synth / smax, 'b-', lw=1.8, label='synthetic (ODE)')
        if i in preds:
            d, m, lo, hi = preds[i]
            if lo is not None:
                ax.fill_between(d, lo / smax, hi / smax, color='red', alpha=0.2, lw=0)
            ax.plot(d, m / smax, 'r--', lw=2, label='predicted (held-out model)')
        ax.plot(rdays, real / rmax, 'k:', lw=1.8, label='real (data_2)')
        ax.axvline(SEQ_LEN, color='gray', lw=0.6, ls=':')
        ax.set_title(REACTOR_IDS[i], fontsize=9); ax.set_ylim(-0.05, 1.2)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Titer, LEAKAGE-FREE: each red line from a model that held that reactor out',
                 y=1.02)
    fig.tight_layout()
    out = here / 'three_way_loro_Titer.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
