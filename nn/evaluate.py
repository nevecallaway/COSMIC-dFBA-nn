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
from scipy.stats import spearmanr
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc)

from model import NextDayPredictor, N_FEATURES, SEQ_LEN, FEATURE_INDICES, N_DAYS
from model_primeur import FluxDecoder

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

FEATURE_NAMES_SHORT = [
    'CellDensity', 'CellSize', 'Titer',
    'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine',
]


def rollout(model, seed_norm, n_steps, device, doe=None, reactor_id=None,
            log_negatives=False, cin=None, is_decoder=False, phase=None):
    """
    Autoregressive rollout. Returns normalized predictions.

    seed_norm: (SEQ_LEN, N_FEATURES) or (SEQ_LEN, N_FEATURES+1) if time-aware.
    If the seed has a time column (last column), it is propagated forward by
    incrementing 1/(N_DAYS-1) per step. Output is always (n_steps, N_FEATURES).

    log_negatives: if True, print a warning when any prediction goes below 0
                   without clipping so the caller can identify the root cause.

    cin/is_decoder: for the flux decoder (FluxDecoder), pass the reactor's
                    physical feed vector (cin) and is_decoder=True. The decoder's
                    forward returns (C_next_norm, v); the fluxes are collected and
                    returned alongside the predictions.
    """
    window   = seed_norm.copy()
    has_time = seed_norm.shape[1] > N_FEATURES
    doe_t    = torch.from_numpy(doe).unsqueeze(0).float().to(device) if doe is not None else None
    cin_t    = torch.from_numpy(cin).unsqueeze(0).float().to(device) if cin is not None else None
    preds    = []
    seq_len  = seed_norm.shape[0]

    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            x         = torch.from_numpy(window).unsqueeze(0).float().to(device)
            if is_decoder:
                # Phase-driven eta: f at the target day (window last day + 1).
                eta_ext = None
                if phase is not None:
                    target_day = int(round(window[-1, N_FEATURES] * (N_DAYS - 1))) + 1
                    eta_ext = float(phase[min(target_day, len(phase) - 1)])
                out, _    = model(x, doe_t, cin_t, eta_ext=eta_ext)
                pred_norm = out.squeeze(0).cpu().numpy()
            else:
                pred_norm = model(x, doe_t).squeeze(0).cpu().numpy()   # (N_FEATURES,)
            if log_negatives:
                for f, val in enumerate(pred_norm):
                    if val < 0.0:
                        day = seq_len + step
                        rid = reactor_id or '?'
                        print(f'  [NEG] reactor={rid}  day={day}  '
                              f'feature={FEATURE_NAMES_SHORT[f]}  value={val:.4f}')
            preds.append(pred_norm)
            if has_time:
                next_time = window[-1, N_FEATURES] + 1.0 / (N_DAYS - 1)
                next_row  = np.append(pred_norm, next_time)
            else:
                next_row = pred_norm
            window = np.vstack([window[1:], next_row])

    return np.array(preds)   # (n_steps, N_FEATURES) normalized


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
    parser.add_argument('--n-eval',         type=int, default=None,
                        help='Evaluate only the first N reactors (default: n_original from npz, or all)')
    parser.add_argument('--eval-reactor',   type=int, default=None,
                        help='Evaluate a single reactor by index (for leave-one-reactor-out)')
    parser.add_argument('--val', action='store_true',
                        help='Evaluate on validation split (held-out training reactors) instead of originals')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match train.py for val split)')
    parser.add_argument('--log-negatives', action='store_true',
                        help='Print a warning for each negative prediction (no clip applied)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip diagnostic plots (useful for single-reactor LORO runs)')
    parser.add_argument('--real-target', action='store_true',
                        help='Score against denormalized data_2 (real measured) instead of '
                             'the ODE trajectory. Use for models trained with train_real.py.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

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

    # Flux decoder checkpoints carry 'n_substeps'; pure-NN ones do not.
    is_decoder = 'n_substeps' in ckpt
    if is_decoder:
        model = FluxDecoder(hidden=ckpt.get('hidden', 64),
                            n_doe=ckpt.get('n_doe', 0),
                            n_input_features=n_input_features,
                            n_substeps=int(ckpt['n_substeps']),
                            integrator=ckpt.get('integrator', 'closed')).to(device)
        model.load_state_dict(ckpt['model_state'])   # restores scaler buffers too
        print(f'Loaded flux decoder {args.model} (substeps={int(ckpt["n_substeps"])})')
    else:
        model = NextDayPredictor(hidden=ckpt.get('hidden', 64),
                                 n_doe=ckpt.get('n_doe', 0),
                                 n_input_features=n_input_features).to(device)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded {args.model}')
    model.eval()

    npz          = np.load(args.data, allow_pickle=True)
    trajectories = npz['trajectories'].astype(np.float32)
    doe_params   = npz['doe_params'].astype(np.float32)   # (N, 3)
    cin_params   = npz['cin_params'].astype(np.float32) if 'cin_params' in npz else None
    if is_decoder and cin_params is None:
        raise SystemExit('cin_params missing from npz: regenerate synthetic_ode.npz '
                         'with the updated generate_synthetic_ode.py.')

    n_original = int(npz['n_original']) if 'n_original' in npz else len(trajectories)

    # Ground truth = denormalized data_2 (real measured) for the real reactors.
    if args.real_target:
        from real_data import denormalize_data2
        real = denormalize_data2(here / 'data' / 'data_2.csv', trajectories, n_original)
        trajectories = trajectories.copy()
        trajectories[:n_original] = real
        print('--real-target: scoring against denormalized data_2 (real measured), '
              'seed and actual both from real')

    if args.val:
        # Reproduce the train/val split from train.py
        reactor_idx  = npz['window_reactor_idx']
        all_reactors = np.unique(reactor_idx)
        rng_split    = np.random.default_rng(args.seed)
        rng_split.shuffle(all_reactors)
        n_val_reactors  = max(1, int(len(all_reactors) * 0.2))
        val_reactor_ids = sorted(all_reactors[:n_val_reactors].tolist())
        # Map back to trajectory indices (extra reactors start at n_original)
        eval_indices = [n_original + r for r in val_reactor_ids]
        trajectories = trajectories[eval_indices]
        doe_params   = doe_params[eval_indices]
        if cin_params is not None:
            cin_params = cin_params[eval_indices]
        print(f'Evaluating {len(eval_indices)} validation reactors '
              f'(indices {eval_indices[:5]}...)')
    elif args.eval_reactor is not None:
        sel = [args.eval_reactor]
        trajectories = trajectories[sel]
        doe_params   = doe_params[sel]
        if cin_params is not None:
            cin_params = cin_params[sel]
        print(f'Evaluating single held-out reactor index {args.eval_reactor}')
    else:
        n_eval     = args.n_eval if args.n_eval is not None else n_original
        trajectories = trajectories[:n_eval]
        doe_params   = doe_params[:n_eval]
        if cin_params is not None:
            cin_params = cin_params[:n_eval]
        print(f'Evaluating {trajectories.shape[0]} reactors '
              f'(n_original={n_original}, total in npz={len(npz["trajectories"])})')

    n_reactors, n_days, _ = trajectories.shape
    sub = trajectories[:, :, FEATURE_INDICES]

    # Load shuffled model if provided
    shuffled_model = None
    if args.shuffled_model and Path(args.shuffled_model).exists():
        shuf_ckpt     = torch.load(args.shuffled_model, map_location=device,
                                   weights_only=False)
        shuffled_model = NextDayPredictor(hidden=shuf_ckpt.get('hidden', 64)).to(device)
        shuffled_model.load_state_dict(shuf_ckpt['model_state'])
        shuffled_model.eval()
        print(f'Loaded shuffled baseline from {args.shuffled_model}')

    n_pred          = n_days - seq_len
    titer_within_10 = 0
    shuf_within_10  = 0
    errors          = []

    # Accumulators
    all_preds_norm        = []
    all_actuals_norm      = []
    all_shuf_preds_norm   = []
    endpoint_within_10      = np.zeros(N_FEATURES, dtype=int)
    shuf_endpoint_within_10 = np.zeros(N_FEATURES, dtype=int)

    has_shuf = shuffled_model is not None
    shuf_col = f'{"Shuf pred":>12} {"Shuf err":>9}' if has_shuf else ''
    print(f'\n{"Reactor":<10} {"Actual":>10} {"Predicted":>12} {"Error":>8}  {shuf_col}')
    print('-' * (48 + (23 if has_shuf else 0)))

    for i in range(n_reactors):
        seed_raw   = sub[i, :seq_len, :]
        seed_feats = np.clip((seed_raw - feature_min) / scale, 0.0, 1.0)
        if use_time:
            time_col  = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
            seed_norm = np.concatenate([seed_feats, time_col], axis=1)
        else:
            seed_norm = seed_feats
        doe_raw   = doe_params[i]
        if doe_min is not None:
            doe_scale = doe_max - doe_min
            doe_scale[doe_scale == 0] = 1.0
            doe_raw = (doe_raw - doe_min) / doe_scale
        cin_i = cin_params[i] if (is_decoder and cin_params is not None) else None
        preds = rollout(model, seed_norm, n_steps=n_pred, device=device, doe=doe_raw,
                        reactor_id=f'R{i:04d}', log_negatives=args.log_negatives,
                        cin=cin_i, is_decoder=is_decoder)

        actual_norm = np.clip((sub[i, seq_len:, :] - feature_min) / scale, 0.0, 1.0)
        all_preds_norm.append(preds)
        all_actuals_norm.append(actual_norm)

        # Titer within 10%
        actual_titer = sub[i, -1, IDX_TITER]
        pred_titer   = preds[-1, IDX_TITER] * scale[IDX_TITER] + feature_min[IDX_TITER]
        err = abs(pred_titer - actual_titer) / actual_titer if actual_titer > 0 else float('nan')
        errors.append(err)
        if not np.isnan(err) and err <= 0.10:
            titer_within_10 += 1

        # Endpoint within 10% for all features
        for f in range(N_FEATURES):
            actual_end = sub[i, -1, f]
            pred_end   = preds[-1, f] * scale[f] + feature_min[f]
            if actual_end > 0 and abs(pred_end - actual_end) / actual_end <= 0.10:
                endpoint_within_10[f] += 1

        shuf_str = ''
        if has_shuf:
            shuf_preds = rollout(shuffled_model, seed_norm,
                                 n_steps=n_pred, device=device, doe=doe_raw)
            all_shuf_preds_norm.append(shuf_preds)
            shuf_titer = shuf_preds[-1, IDX_TITER] * scale[IDX_TITER] + feature_min[IDX_TITER]
            shuf_err   = abs(shuf_titer - actual_titer) / actual_titer if actual_titer > 0 else float('nan')
            if not np.isnan(shuf_err) and shuf_err <= 0.10:
                shuf_within_10 += 1
            shuf_str = f'{shuf_titer:>12.3f} {shuf_err:>8.1%}'
            for f in range(N_FEATURES):
                actual_end = sub[i, -1, f]
                shuf_end   = shuf_preds[-1, f] * scale[f] + feature_min[f]
                if actual_end > 0 and abs(shuf_end - actual_end) / actual_end <= 0.10:
                    shuf_endpoint_within_10[f] += 1

        flag = 'OK' if (not np.isnan(err) and err <= 0.10) else ''
        print(f'R{i:04d}     {actual_titer:>10.3f} {pred_titer:>12.3f} {err:>7.1%}  {flag:<4}  {shuf_str}')

    print(f'\nMean titer error: {np.nanmean(errors):.1%}')

    # ------------------------------------------------------------------
    # R² and endpoint within 10%
    # ------------------------------------------------------------------
    all_preds_norm   = np.array(all_preds_norm)
    all_actuals_norm = np.array(all_actuals_norm)
    has_shuf_preds   = len(all_shuf_preds_norm) == n_reactors
    if has_shuf_preds:
        all_shuf_preds_norm = np.array(all_shuf_preds_norm)

    # Mean trajectory across reactors: what every reactor has in common.
    # Subtracting it isolates the DoE-driven reactor-to-reactor variation.
    mean_traj = all_actuals_norm.mean(axis=0, keepdims=True)  # (1, n_pred, N_FEATURES)
    actual_dev = all_actuals_norm - mean_traj                  # (n_reactors, n_pred, N_FEATURES)
    pred_dev   = all_preds_norm   - mean_traj                  # deviations predicted vs mean

    shuf_header = f'  {"Shuf R²":>7}  {"Shuf E10%":>9}' if has_shuf_preds else ''
    print(f'\n{"Feature":<18} {"R²":>6}  {"DoE R²":>7}  {"End 10%":>8}{shuf_header}')
    print('-' * (44 + (20 if has_shuf_preds else 0)))
    for f, name in enumerate(FEATURE_NAMES):
        y_true = all_actuals_norm[:, :, f].flatten()
        y_pred = all_preds_norm[:, :, f].flatten()

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # DoE R²: R² on deviations from mean trajectory
        d_true   = actual_dev[:, :, f].flatten()
        d_pred   = pred_dev[:, :, f].flatten()
        ss_res_d = ((d_true - d_pred) ** 2).sum()
        ss_tot_d = ((d_true - d_true.mean()) ** 2).sum()
        r2_doe   = 1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else float('nan')

        end_str  = f'{endpoint_within_10[f]}/{n_reactors}'
        shuf_str = ''
        if has_shuf_preds:
            y_shuf   = all_shuf_preds_norm[:, :, f].flatten()
            ss_res_s = ((y_true - y_shuf) ** 2).sum()
            r2_shuf  = 1 - ss_res_s / ss_tot if ss_tot > 0 else float('nan')
            shuf_end = f'{shuf_endpoint_within_10[f]}/{n_reactors}'
            shuf_str = f'  {r2_shuf:>7.3f}  {shuf_end:>9}'

        print(f'{name:<18} {r2:>6.3f}  {r2_doe:>7.3f}  {end_str:>8}{shuf_str}')

    # ------------------------------------------------------------------
    # Spearman rank correlation
    # Two questions:
    #   1. Across reactors: does the model rank reactors correctly by
    #      final titer? (useful for DoE optimization)
    #   2. Within trajectory: does the model capture the temporal shape
    #      of each feature across all reactors?
    # ------------------------------------------------------------------
    print(f'\n--- Spearman rank correlation ---')

    # Across-reactor endpoint ranking (titer)
    actual_end_titers = sub[:, -1, IDX_TITER]
    pred_end_titers   = np.array([
        all_preds_norm[i][-1, IDX_TITER] * scale[IDX_TITER] + feature_min[IDX_TITER]
        for i in range(n_reactors)
    ])
    rho_titer, _ = spearmanr(actual_end_titers, pred_end_titers)
    print(f'  Across-reactor titer ranking: rho = {rho_titer:.3f}')

    # Within-trajectory per feature
    print(f'\n  {"Feature":<18} {"Traj rho":>9}')
    print(f'  {"-"*30}')
    for f, name in enumerate(FEATURE_NAMES):
        rhos = []
        for i in range(n_reactors):
            t = all_actuals_norm[i, :, f]
            p = all_preds_norm[i, :, f]
            if t.std() > 1e-8 and p.std() > 1e-8:
                rho, _ = spearmanr(t, p)
                rhos.append(rho)
        mean_rho = np.nanmean(rhos) if rhos else float('nan')
        print(f'  {name:<18} {mean_rho:>9.3f}')
    print()

    print_summary(titer_within_10, n_reactors,
                  shuffled_within_10=shuf_within_10 if has_shuf else None)

    # ------------------------------------------------------------------
    # Learned sigmas
    # ------------------------------------------------------------------
    if 'log_sigma' in ckpt:
        ls = ckpt['log_sigma'].cpu().numpy()
        sigmas = np.exp(ls)
        weights = np.exp(-2 * ls)
        print(f'--- Learned sigmas (from checkpoint) ---')
        print(f'  {"Feature":<18} {"sigma":>8} {"weight":>8}')
        print(f'  {"-"*36}')
        for f, name in enumerate(FEATURE_NAMES):
            print(f'  {name:<18} {sigmas[f]:>8.4f} {weights[f]:>8.4f}')
        print()

    # ------------------------------------------------------------------
    # Per-feature endpoint error distribution
    # ------------------------------------------------------------------
    print(f'--- Per-feature endpoint errors (%) ---')
    print(f'  {"Feature":<18} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
    print(f'  {"-"*44}')
    for f, name in enumerate(FEATURE_NAMES):
        errs = []
        for i in range(n_reactors):
            actual_end = sub[i, -1, f]
            pred_end   = all_preds_norm[i][-1, f] * scale[f] + feature_min[f]
            if actual_end > 0:
                errs.append(abs(pred_end - actual_end) / actual_end * 100)
        if errs:
            errs = np.array(errs)
            print(f'  {name:<18} {errs.mean():>7.1f}% {errs.std():>7.1f}% '
                  f'{errs.min():>7.1f}% {errs.max():>7.1f}%')
        else:
            print(f'  {name:<18}      N/A')
    print()

    # ------------------------------------------------------------------
    # High/Low binning: confusion matrix, precision, recall, F1
    # Threshold: median of actual values per feature (across all
    # reactors and predicted time steps). Each time point is one sample.
    # ------------------------------------------------------------------
    print(f'--- High/Low Classification (threshold = per-feature median) ---')
    print(f'  {"Feature":<18} {"Prec":>6} {"Recall":>7} {"F1":>6} '
          f'{"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4}  {"Thresh":>10}')
    print(f'  {"-"*72}')

    for f, name in enumerate(FEATURE_NAMES):
        actuals_phys = []
        preds_phys   = []
        for i in range(n_reactors):
            for t in range(all_actuals_norm.shape[1]):
                a = all_actuals_norm[i, t, f] * scale[f] + feature_min[f]
                p = all_preds_norm[i, t, f]   * scale[f] + feature_min[f]
                actuals_phys.append(a)
                preds_phys.append(p)
        actuals_phys = np.array(actuals_phys)
        preds_phys   = np.array(preds_phys)

        if actuals_phys.std() < 1e-10:
            print(f'  {name:<18}   (no variance)')
            continue

        thresh = np.median(actuals_phys)
        y_true = (actuals_phys >= thresh).astype(int)
        y_pred = (preds_phys   >= thresh).astype(int)

        cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print(f'  {name:<18} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} '
              f'{tp:>4} {fp:>4} {fn:>4} {tn:>4}  {thresh:>10.4f}')
    print()

    # Endpoint-only binning (final day per reactor)
    print(f'--- Endpoint High/Low (final day only, median threshold) ---')
    print(f'  {"Feature":<18} {"Prec":>6} {"Recall":>7} {"F1":>6} '
          f'{"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4}  {"Thresh":>10}')
    print(f'  {"-"*72}')

    for f, name in enumerate(FEATURE_NAMES):
        actuals_end = np.array([sub[i, -1, f] for i in range(n_reactors)])
        preds_end   = np.array([all_preds_norm[i][-1, f] * scale[f] + feature_min[f]
                                for i in range(n_reactors)])

        if actuals_end.std() < 1e-10:
            print(f'  {name:<18}   (no variance)')
            continue

        thresh = np.median(actuals_end)
        y_true = (actuals_end >= thresh).astype(int)
        y_pred = (preds_end   >= thresh).astype(int)

        cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print(f'  {name:<18} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} '
              f'{tp:>4} {fp:>4} {fn:>4} {tn:>4}  {thresh:>10.4f}')
    print()

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    if args.no_plots:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        out_dir = here

        # 1. Predicted vs actual trajectories per reactor (all features)
        max_plot_reactors = min(n_reactors, 15)
        fig, axes = plt.subplots(max_plot_reactors, N_FEATURES,
                                 figsize=(3 * N_FEATURES, 2.5 * max_plot_reactors))
        if max_plot_reactors == 1:
            axes = axes[np.newaxis, :]
        days_pred = np.arange(seq_len, n_days)
        for i in range(max_plot_reactors):
            for f in range(N_FEATURES):
                ax = axes[i, f]
                actual_raw = sub[i, seq_len:, f]
                pred_raw   = all_preds_norm[i][:, f] * scale[f] + feature_min[f]
                ax.plot(days_pred, actual_raw, 'k-', lw=1.2, label='Actual')
                ax.plot(days_pred, pred_raw, 'r--', lw=1.2, label='Predicted')
                if i == 0:
                    ax.set_title(FEATURE_NAMES[f], fontsize=8)
                if f == 0:
                    ax.set_ylabel(f'R{i:04d}', fontsize=8)
                ax.tick_params(labelsize=6)
        axes[0, -1].legend(fontsize=6)
        fig.suptitle('Predicted vs Actual Trajectories (physical units)', y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        suffix = f' (showing {max_plot_reactors}/{n_reactors})' if max_plot_reactors < n_reactors else ''
        print(f'Saved {out_dir / "diag_trajectories.png"}{suffix}')

        # 2. Endpoint error box plot per feature
        fig, ax = plt.subplots(figsize=(10, 4))
        err_data = []
        labels = []
        for f, name in enumerate(FEATURE_NAMES):
            errs = []
            for i in range(n_reactors):
                actual_end = sub[i, -1, f]
                pred_end   = all_preds_norm[i][-1, f] * scale[f] + feature_min[f]
                if actual_end > 0:
                    errs.append((pred_end - actual_end) / actual_end * 100)
            if errs:
                err_data.append(errs)
                labels.append(name)
        ax.boxplot(err_data, labels=labels)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.axhline(10, color='r', lw=0.5, ls='--', alpha=0.5)
        ax.axhline(-10, color='r', lw=0.5, ls='--', alpha=0.5)
        ax.set_ylabel('Endpoint Error (%)')
        ax.set_title('Per-feature Endpoint Error Distribution')
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_endpoint_errors.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_dir / "diag_endpoint_errors.png"}')

        # 3. Predicted vs actual scatter (endpoints, all features)
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for f, name in enumerate(FEATURE_NAMES):
            ax = axes[f]
            actuals = [sub[i, -1, f] for i in range(n_reactors)]
            preds   = [all_preds_norm[i][-1, f] * scale[f] + feature_min[f]
                       for i in range(n_reactors)]
            ax.scatter(actuals, preds, s=30, alpha=0.8)
            lo = min(min(actuals), min(preds))
            hi = max(max(actuals), max(preds))
            margin = (hi - lo) * 0.1 if hi > lo else 1.0
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', lw=0.8, alpha=0.5)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel('Actual', fontsize=7)
            ax.set_ylabel('Predicted', fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle('Endpoint: Predicted vs Actual', y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_scatter.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_dir / "diag_scatter.png"}')

        # 4. Confusion matrix heatmaps (trajectory-level, all time steps)
        plot_features = [(f, name) for f, name in enumerate(FEATURE_NAMES)
                         if sub[:, SEQ_LEN:, f].std() > 1e-10]
        n_plot = len(plot_features)
        fig, axes = plt.subplots(2, (n_plot + 1) // 2, figsize=(4 * ((n_plot + 1) // 2), 7))
        axes = axes.flatten()
        for idx, (f, name) in enumerate(plot_features):
            ax = axes[idx]
            actuals_phys = (all_actuals_norm[:, :, f] * scale[f] + feature_min[f]).flatten()
            preds_phys   = (all_preds_norm[:, :, f]   * scale[f] + feature_min[f]).flatten()
            thresh = np.median(actuals_phys)
            y_true = (actuals_phys >= thresh).astype(int)
            y_pred = (preds_phys   >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            im = ax.imshow(cm, cmap='Blues', aspect='equal')
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                            fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Low', 'High'])
            ax.set_yticklabels(['Low', 'High'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            f1 = f1_score(y_true, y_pred, zero_division=0)
            ax.set_title(f'{name}\nF1={f1:.3f}', fontsize=9)
        for idx in range(len(plot_features), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle('High/Low Confusion Matrices (all time steps, median threshold)', y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_confusion.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_dir / "diag_confusion.png"}')

        # 5. ROC curves per feature
        fig, axes = plt.subplots(2, (len(plot_features) + 1) // 2,
                                 figsize=(4 * ((len(plot_features) + 1) // 2), 7))
        axes = axes.flatten()
        for idx, (f, name) in enumerate(plot_features):
            ax = axes[idx]
            actuals_phys = (all_actuals_norm[:, :, f] * scale[f] + feature_min[f]).flatten()
            preds_phys   = (all_preds_norm[:, :, f]   * scale[f] + feature_min[f]).flatten()
            thresh = np.median(actuals_phys)
            y_true = (actuals_phys >= thresh).astype(int)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                ax.text(0.5, 0.5, 'Single class\n(no split)', ha='center',
                        va='center', transform=ax.transAxes)
                ax.set_title(name, fontsize=9)
                continue
            fpr, tpr, _ = roc_curve(y_true, preds_phys)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=8)
        for idx in range(len(plot_features), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle('ROC Curves (High/Low, median threshold)', y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_roc.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_dir / "diag_roc.png"}')

        # 6. Distribution comparison: predicted vs actual per feature
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        for f, name in enumerate(FEATURE_NAMES):
            ax = axes[f]
            actuals_phys = (all_actuals_norm[:, :, f] * scale[f] + feature_min[f]).flatten()
            preds_phys   = (all_preds_norm[:, :, f]   * scale[f] + feature_min[f]).flatten()
            bins = 30
            lo = min(actuals_phys.min(), preds_phys.min())
            hi = max(actuals_phys.max(), preds_phys.max())
            bin_edges = np.linspace(lo, hi, bins + 1)
            ax.hist(actuals_phys, bins=bin_edges, alpha=0.5, label='Actual', density=True)
            ax.hist(preds_phys,   bins=bin_edges, alpha=0.5, label='Predicted', density=True)
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle('Distribution: Predicted vs Actual (all time steps)', y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_distributions.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_dir / "diag_distributions.png"}')

    except ImportError:
        print('matplotlib not available, skipping plots')


if __name__ == '__main__':
    main()
