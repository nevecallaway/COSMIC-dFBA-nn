#!/usr/bin/env python3
"""
Compare real experimental data (data_2.csv), ODE synthetic trajectories,
and model predictions side by side.

data_2.csv contains normalized values (per-reactor, per-component / max).
We approximate physical units by scaling data_2 by the ODE trajectory's
per-reactor per-component max.

Usage:
    !python compare_real.py
    !python compare_real.py --model model_v2.pt --data synthetic_ode.npz
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import NextDayPredictor, N_FEATURES, SEQ_LEN, FEATURE_INDICES, N_DAYS
from evaluate import rollout

FEATURE_NAMES = [
    'Cell Density', 'Cell Size', 'Titer',
    'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine',
]

REACTOR_IDS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
               'R0006', 'R0008', 'R0010', 'R0011', 'R0012']


def load_data2(data_dir):
    """Load data_2.csv and return per-reactor normalized trajectories."""
    df = pd.read_csv(data_dir / 'data_2.csv', skiprows=1)
    df.columns = (
        ['Vessel', 'Time', 'Phase']
        + [f'C{i}' for i in range(25)]
    )
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])

    real = {}
    for reactor in df['Vessel'].dropna().unique():
        rdf = df[df['Vessel'] == reactor].sort_values('Time')
        days = rdf['Time'].values.astype(float)
        vals = np.array([rdf[f'C{i}'].values.astype(float)
                         for i in FEATURE_INDICES]).T
        real[str(reactor)] = {'days': days, 'vals': vals}
    return real


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(here / 'model_v2.pt'))
    parser.add_argument('--data',  default=str(here / 'synthetic_ode.npz'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt    = torch.load(args.model, map_location=device, weights_only=False)
    doe_min = ckpt.get('doe_min', None)
    doe_max = ckpt.get('doe_max', None)

    if 'scaler' in ckpt:
        scaler      = ckpt['scaler']
        feature_min = scaler.data_min_.astype(np.float32)
        scale       = scaler.data_range_.astype(np.float32)
    else:
        feature_min = ckpt['feature_min']
        scale       = ckpt['feature_max'] - feature_min
    scale[scale == 0] = 1.0

    n_input_features = ckpt.get('n_input_features', N_FEATURES)
    use_time         = n_input_features > N_FEATURES
    seq_len          = ckpt.get('seq_len', SEQ_LEN)

    model = NextDayPredictor(hidden=ckpt.get('hidden', 64),
                             n_doe=ckpt.get('n_doe', 0),
                             n_input_features=n_input_features).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load ODE data
    npz          = np.load(args.data, allow_pickle=True)
    trajectories = npz['trajectories'].astype(np.float32)
    doe_params   = npz['doe_params'].astype(np.float32)
    n_original   = int(npz['n_original']) if 'n_original' in npz else 10

    ode_trajs = trajectories[:n_original]
    ode_sub   = ode_trajs[:, :, FEATURE_INDICES]
    n_reactors, n_days, _ = ode_trajs.shape

    # Load real data
    real = load_data2(here / 'data')

    # Run model rollouts
    model_preds = []
    for i in range(n_reactors):
        seed_raw   = ode_sub[i, :seq_len, :]
        seed_feats = (seed_raw - feature_min) / scale
        if use_time:
            time_col  = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
            seed_norm = np.concatenate([seed_feats, time_col], axis=1)
        else:
            seed_norm = seed_feats
        doe_raw   = doe_params[i]
        if doe_min is not None:
            doe_sc = doe_max - doe_min
            doe_sc[doe_sc == 0] = 1.0
            doe_raw = (doe_raw - doe_min) / doe_sc
        preds = rollout(model, seed_norm,
                        n_steps=n_days - seq_len, device=device, doe=doe_raw)
        model_preds.append(preds)

    # Plot: one row per reactor, one column per feature
    fig, axes = plt.subplots(n_reactors, N_FEATURES,
                             figsize=(3 * N_FEATURES, 2.5 * n_reactors))
    days_all  = np.arange(n_days)
    days_pred = np.arange(seq_len, n_days)

    for i, reactor_id in enumerate(REACTOR_IDS):
        for f in range(N_FEATURES):
            ax = axes[i, f]

            # ODE trajectory (physical units)
            ode_vals = ode_sub[i, :, f]
            ax.plot(days_all, ode_vals, 'k-', lw=1.5, label='ODE')

            # Real data (denormalized using ODE per-reactor max)
            if reactor_id in real:
                rd = real[reactor_id]
                ode_max_val = ode_vals.max()
                real_phys = rd['vals'][:, f] * ode_max_val if ode_max_val > 0 else rd['vals'][:, f]
                ax.plot(rd['days'], real_phys, 'b.--', lw=0.8, ms=4,
                        alpha=0.7, label='Real (approx)')

            # Model predictions (unscaled to physical)
            pred_phys = model_preds[i][:, f] * scale[f] + feature_min[f]
            ax.plot(days_pred, pred_phys, 'r--', lw=1.2, label='Model')

            if i == 0:
                ax.set_title(FEATURE_NAMES[f], fontsize=9)
            if f == 0:
                ax.set_ylabel(reactor_id, fontsize=9)
            ax.tick_params(labelsize=6)

    axes[0, -1].legend(fontsize=6, loc='upper left')
    fig.suptitle('Real (blue) vs ODE Synthetic (black) vs Model Predictions (red)',
                 y=1.01, fontsize=11)
    fig.tight_layout()
    out = here / 'compare_real.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')

    # Per-reactor ODE vs real RMSE (normalized space)
    print(f'\n--- ODE vs Real fit quality (normalized space) ---')
    print(f'  {"Reactor":<10} {"Feature":<18} {"RMSE":>8}')
    print(f'  {"-"*38}')
    for i, reactor_id in enumerate(REACTOR_IDS):
        if reactor_id not in real:
            continue
        rd = real[reactor_id]
        for f, name in enumerate(FEATURE_NAMES):
            ode_max_val = ode_sub[i, :, f].max()
            if ode_max_val == 0:
                continue
            ode_norm = ode_sub[i, :, f] / ode_max_val
            real_norm = rd['vals'][:len(ode_norm), f]
            if len(real_norm) < len(ode_norm):
                ode_norm = ode_norm[:len(real_norm)]
            rmse = np.sqrt(((ode_norm - real_norm) ** 2).mean())
            if rmse > 0.1:
                print(f'  {reactor_id:<10} {name:<18} {rmse:>8.4f}  !')
    print()


if __name__ == '__main__':
    main()
