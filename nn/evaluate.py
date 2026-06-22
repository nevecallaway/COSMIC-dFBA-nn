#!/usr/bin/env python3
"""
Evaluation script for COSMIC-dFBA surrogate v2 (NextDayPredictor).

Runs autoregressive rollout per reactor and reports metrics against
paper benchmarks.

Paper benchmarks:
  f(t) accuracy +/-0.1 : 72.3%   (94/130 time points)
  f(t) accuracy +/-0.2 : 90.8%   (118/130 time points)
  Specificity          : 0.780
  Sensitivity          : 0.681
  F1                   : 0.731
  MCC                  : 0.454
  Titer within 10%     : 10/10

Usage:
    !python evaluate.py
    !python evaluate.py --model model_v2.pt --data synthetic_ode.npz
    !python evaluate.py --shuffled-model shuffled_v2.pt   # with baseline
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from model import NextDayPredictor, N_FEATURES, SEQ_LEN, FEATURE_INDICES
from utils import load_experimental_data

IDX_TITER = 2   # index of titer within the 8-feature vector

FEATURE_NAMES = [
    'Cell Density', 'Cell Size', 'Titer',
    'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine',
]

PAPER = {
    'f_acc_01':   0.723,
    'f_acc_02':   0.908,
    'specificity': 0.780,
    'sensitivity': 0.681,
    'f1':          0.731,
    'mcc':         0.454,
    'titer_within_10': '10/10',
}


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(model, seed_norm, feature_min, feature_max, n_steps, device, doe=None):
    """Autoregressive rollout. Returns normalized [0,1] predictions."""
    window = seed_norm.copy()
    doe_t  = torch.from_numpy(doe).unsqueeze(0).float().to(device) if doe is not None else None
    preds  = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            x         = torch.from_numpy(window).unsqueeze(0).float().to(device)
            mu, _     = model(x, doe_t)          # discard log_var at inference
            pred_norm = np.clip(mu.squeeze(0).cpu().numpy(), 0.0, 1.0)
            preds.append(pred_norm)
            window = np.vstack([window[1:], pred_norm])

    return np.array(preds)   # (n_steps, N_FEATURES) normalized [0,1]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(titer_within_10, n_reactors, shuffled_within_10=None):
    w10_str  = f'{titer_within_10}/{n_reactors}'
    shuf_str = f'{shuffled_within_10}/{n_reactors}' if shuffled_within_10 is not None else ''

    has_shuffled = shuffled_within_10 is not None
    col = f'{"Shuffled":>10}' if has_shuffled else ''

    print()
    print('=' * (55 + (11 if has_shuffled else 0)))
    print(f'{"Metric":<28} {"Ours":>8}  {col}  {"Paper":>10}')
    print('-' * (55 + (11 if has_shuffled else 0)))

    def row(name, ours, shuf, paper):
        s = f'{shuf:>10}  ' if has_shuffled else ''
        print(f'{name:<28} {ours:>8}  {s}{paper:>10}')

    row('f(t) acc +/-0.1', 'N/A', 'N/A', f'{PAPER["f_acc_01"]:.1%}')
    row('f(t) acc +/-0.2', 'N/A', 'N/A', f'{PAPER["f_acc_02"]:.1%}')
    row('Specificity',      'N/A', 'N/A', f'{PAPER["specificity"]:.3f}')
    row('Sensitivity',      'N/A', 'N/A', f'{PAPER["sensitivity"]:.3f}')
    row('F1',               'N/A', 'N/A', f'{PAPER["f1"]:.3f}')
    row('MCC',              'N/A', 'N/A', f'{PAPER["mcc"]:.3f}')
    row('Titer within 10%', w10_str, shuf_str, PAPER['titer_within_10'])

    print('=' * (55 + (11 if has_shuffled else 0)))
    print('N/A: phase prediction not yet implemented in v2')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here   = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          default=str(here / 'model_v2.pt'))
    parser.add_argument('--shuffled-model', default=None,
                        help='Path to model trained on shuffled data for baseline comparison')
    parser.add_argument('--data',           default=str(here / 'synthetic_ode.npz'))
    parser.add_argument('--real-data',      default=str(here / 'data' / 'data_2.csv'),
                        help='Path to real experimental data for validation')
    parser.add_argument('--n-eval',         type=int, default=None,
                        help='Evaluate only the first N reactors (default: n_original from npz, or all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt        = torch.load(args.model, map_location=device, weights_only=False)
    feature_min = ckpt['feature_min']
    feature_max = ckpt['feature_max']
    doe_min     = ckpt.get('doe_min', None)
    doe_max     = ckpt.get('doe_max', None)

    model = NextDayPredictor(hidden=ckpt.get('hidden', 64),
                             n_doe=ckpt.get('n_doe', 0)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded {args.model}')

    npz          = np.load(args.data, allow_pickle=True)
    trajectories = npz['trajectories'].astype(np.float32)
    doe_params   = npz['doe_params'].astype(np.float32)   # (N, 3)

    # Limit evaluation to original real-data reactors by default.
    # n_original is saved by generate_synthetic_ode.py; fall back to all.
    n_original = int(npz['n_original']) if 'n_original' in npz else len(trajectories)
    n_eval     = args.n_eval if args.n_eval is not None else n_original
    trajectories = trajectories[:n_eval]
    doe_params   = doe_params[:n_eval]
    n_reactors, n_days, _ = trajectories.shape
    print(f'Evaluating {n_reactors} reactors (n_original={n_original}, total in npz={len(npz["trajectories"])})')
    sub   = trajectories[:, :, FEATURE_INDICES]
    scale = feature_max - feature_min
    scale[scale == 0] = 1.0

    # Load shuffled model if provided
    shuffled_model = None
    if args.shuffled_model and Path(args.shuffled_model).exists():
        shuf_ckpt     = torch.load(args.shuffled_model, map_location=device,
                                   weights_only=False)
        shuffled_model = NextDayPredictor(hidden=shuf_ckpt.get('hidden', 64)).to(device)
        shuffled_model.load_state_dict(shuf_ckpt['model_state'])
        shuffled_model.eval()
        print(f'Loaded shuffled baseline from {args.shuffled_model}')

    n_pred          = n_days - SEQ_LEN
    titer_within_10 = 0
    shuf_within_10  = 0
    errors          = []

    has_shuf = shuffled_model is not None
    shuf_col = f'{"Shuf pred":>12} {"Shuf err":>9}' if has_shuf else ''
    print(f'\n{"Reactor":<10} {"Actual":>10} {"Predicted":>12} {"Error":>8}  {shuf_col}')
    print('-' * (48 + (23 if has_shuf else 0)))

    for i in range(n_reactors):
        seed_raw  = sub[i, :SEQ_LEN, :]
        seed_norm = np.clip((seed_raw - feature_min) / scale, 0.0, 1.0)
        doe_raw = doe_params[i]
        if doe_min is not None:
            doe_scale = doe_max - doe_min
            doe_scale[doe_scale == 0] = 1.0
            doe_raw = (doe_raw - doe_min) / doe_scale
        preds     = rollout(model, seed_norm, feature_min, feature_max,
                            n_steps=n_pred, device=device, doe=doe_raw)

        actual_titer = sub[i, -1, IDX_TITER]   # raw
        pred_titer   = preds[-1, IDX_TITER] * scale[IDX_TITER] + feature_min[IDX_TITER]  # denormalized
        err = abs(pred_titer - actual_titer) / actual_titer if actual_titer > 0 else float('nan')
        errors.append(err)
        if not np.isnan(err) and err <= 0.10:
            titer_within_10 += 1

        shuf_str = ''
        if has_shuf:
            shuf_preds = rollout(shuffled_model, seed_norm, feature_min, feature_max,
                                 n_steps=n_pred, device=device, doe=doe_raw)
            shuf_titer = shuf_preds[-1, IDX_TITER]
            shuf_err   = abs(shuf_titer - actual_titer) / actual_titer if actual_titer > 0 else float('nan')
            if not np.isnan(shuf_err) and shuf_err <= 0.10:
                shuf_within_10 += 1
            shuf_str = f'{shuf_titer:>12.3f} {shuf_err:>8.1%}'

        flag = 'OK' if (not np.isnan(err) and err <= 0.10) else ''
        print(f'R{i:04d}     {actual_titer:>10.3f} {pred_titer:>12.3f} {err:>7.1%}  {flag:<4}  {shuf_str}')

    print(f'\nMean titer error: {np.nanmean(errors):.1%}')

    # ------------------------------------------------------------------
    # Real data validation (if available)
    # ------------------------------------------------------------------
    if Path(args.real_data).exists():
        print(f'\n--- Real data validation ({args.real_data}) ---')
        print('Note: real data is per-reactor normalized [0,1]. '
              'Predictions normalized per-reactor for comparison.')

        real_trajs, _, _, real_meta = load_experimental_data(args.real_data)
        # real_trajs: (N, T, 25) already per-reactor-per-component normalized

        real_sub = real_trajs[:, :, FEATURE_INDICES].astype(np.float32)
        real_doe = real_meta.get('doe_params')   # (N, 3) or None
        n_real, t_real, _ = real_sub.shape
        n_real_pred = t_real - SEQ_LEN

        sq_err_real = np.zeros(N_FEATURES)
        n_pts = 0

        for i in range(n_real):
            seed     = real_sub[i, :SEQ_LEN, :]   # already in [0,1]
            doe_i    = real_doe[i].astype(np.float32) if real_doe is not None else None
            preds_raw = rollout(model, seed, feature_min, feature_max,
                                n_steps=n_real_pred, device=device, doe=doe_i)

            # Normalize predictions per-reactor to match real data scale
            actual = real_sub[i, SEQ_LEN:, :]
            for f in range(N_FEATURES):
                mx = preds_raw[:, f].max()
                if mx > 0:
                    preds_raw[:, f] /= mx

            sq_err_real += ((preds_raw - actual) ** 2).sum(axis=0)
            n_pts += n_real_pred

        rmse_real = np.sqrt(sq_err_real / n_pts)
        print(f'\n  Per-feature RMSE vs real trajectories (normalized space):')
        for name, r in zip(FEATURE_NAMES, rmse_real):
            print(f'    {name:<18}: {r:.4f}')
        print(f'    {"Mean":<18}: {rmse_real.mean():.4f}')
    else:
        print(f'\nReal data not found at {args.real_data} -- skipping real validation')

    print_summary(titer_within_10, n_reactors,
                  shuffled_within_10=shuf_within_10 if has_shuf else None)


if __name__ == '__main__':
    main()
