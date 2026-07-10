#!/usr/bin/env python3
"""
Three-way trajectory plot per reactor: real (data_2) vs synthetic (ODE) vs
predicted (model rollout), for one feature (default Titer).

Shows three things at once:
  - synthetic vs predicted: how well en Primeur emulates the ODE (should overlay
    tightly on days 6-12, the rollout region);
  - synthetic/predicted vs real: the wet-lab gap (large for titer by design,
    since eta=1 intentionally does not match the measured titer).

Each line is normalized to its own peak so shapes are comparable. Predicted is
normalized by the synthetic peak (same units), real by its own peak.

Usage:
    python plot_three_way.py                 # Titer
    python plot_three_way.py --feature 0     # by MODEL feature index (0=CellDensity)
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from model import FEATURE_INDICES, N_FEATURES, SEQ_LEN, N_DAYS
from model_primeur import FluxDecoder
from evaluate import rollout

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer', 'Glucose',
                 'Glutamine', 'Asparagine', 'Serine', 'Glycine']
REACTOR_IDS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
               'R0006', 'R0008', 'R0010', 'R0011', 'R0012']


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(here / 'model_flux.pt'))
    parser.add_argument('--data',  default=str(here / 'synthetic_ode.npz'))
    parser.add_argument('--feature', type=int, default=2,
                        help='MODEL feature index 0-7 (default 2=Titer)')
    args = parser.parse_args()
    feat = args.feature
    comp = FEATURE_INDICES[feat]   # index into the 25-component trajectory / data_2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    scaler = ckpt['scaler']
    feature_min = scaler.data_min_.astype(np.float32)
    scale = scaler.data_range_.astype(np.float32); scale[scale == 0] = 1.0
    doe_min, doe_max = ckpt['doe_min'], ckpt['doe_max']

    model = FluxDecoder(hidden=ckpt.get('hidden', 64), n_doe=ckpt.get('n_doe', 3),
                        n_input_features=ckpt.get('n_input_features', N_FEATURES + 1),
                        n_substeps=int(ckpt['n_substeps'])).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()

    npz = np.load(args.data, allow_pickle=True)
    trajectories = npz['trajectories'].astype(np.float32)
    doe_params   = npz['doe_params'].astype(np.float32)
    cin_params   = npz['cin_params'].astype(np.float32)
    n_original   = int(npz['n_original'])
    seq_len      = int(npz['seq_len']) if 'seq_len' in npz else SEQ_LEN

    df2 = pd.read_csv(here / 'data' / 'data_2.csv', skiprows=1)
    df2.columns = ['Vessel', 'Time', 'Phase'] + [f'C{i}' for i in range(25)]
    df2['Time'] = pd.to_numeric(df2['Time'], errors='coerce')
    df2 = df2.dropna(subset=['Time'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes = axes.flatten()

    sub = trajectories[:, :, FEATURE_INDICES]   # (N, days, 8)
    for i in range(min(n_original, 10)):
        # rollout prediction (days seq_len..N_DAYS-1)
        seed_feats = np.clip((sub[i, :seq_len, :] - feature_min) / scale, 0.0, 1.0)
        time_col = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed_norm = np.concatenate([seed_feats, time_col], axis=1)
        doe_scale = doe_max - doe_min; doe_scale[doe_scale == 0] = 1.0
        doe_n = (doe_params[i] - doe_min) / doe_scale
        preds = rollout(model, seed_norm, n_steps=N_DAYS - seq_len, device=device,
                        doe=doe_n, cin=cin_params[i], is_decoder=True)      # (7, 8) norm
        pred_phys = preds[:, feat] * scale[feat] + feature_min[feat]

        synth = sub[i, :, feat]                                # (13,) physical
        smax  = synth.max() if synth.max() > 0 else 1.0

        rdf = df2[df2['Vessel'] == REACTOR_IDS[i]].sort_values('Time')
        real = rdf[f'C{comp}'].to_numpy(dtype=float)
        real_days = rdf['Time'].to_numpy(dtype=float)
        rmax = real.max() if real.size and real.max() > 0 else 1.0

        ax = axes[i]
        ax.plot(np.arange(N_DAYS), synth / smax, 'b-', lw=1.8, label='synthetic (ODE)')
        ax.plot(np.arange(seq_len, N_DAYS), pred_phys / smax, 'r--', lw=2,
                label='predicted (model)')
        ax.plot(real_days, real / rmax, 'k:', lw=1.8, label='real (data_2)')
        ax.axvline(seq_len, color='gray', lw=0.6, ls=':')
        ax.set_title(REACTOR_IDS[i], fontsize=9)
        ax.set_ylim(-0.05, 1.2)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'{FEATURE_NAMES[feat]}: real vs synthetic vs predicted '
                 f'(each normalized to its own peak)', y=1.02)
    fig.tight_layout()
    out = here / f'three_way_{FEATURE_NAMES[feat]}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    print('  blue = synthetic ODE (0-12), red dashed = model rollout (6-12), '
          'black dotted = real data_2.')
    print('  red should track blue (model emulates ODE); gap to black is the '
          'wet-lab difference.')


if __name__ == '__main__':
    main()
