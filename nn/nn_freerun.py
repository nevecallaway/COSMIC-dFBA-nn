#!/usr/bin/env python3
"""
Free-running (autoregressive-training) pure NN: train the model the same way it
is used at inference.

Contrast with nn_baseline.py, which is TEACHER-FORCED: there, every training
window is real (synthetic-truth) data and the model only ever predicts one step
ahead from ground truth. This script instead rolls the model out over each run
during TRAINING, feeding the model's OWN predictions back in (free-running), and
scores the loss against the real day at every step. Only the first SEQ_LEN days
(the seed) are real; days SEQ_LEN..N_DAYS-1 are produced from the model's own
outputs, exactly as at inference.

Why: teacher forcing never shows the model its own mistakes, so small errors can
compound over an autoregressive rollout (exposure bias). Training the way you
infer (here) can shrink that train/inference gap, at the cost of a harder, less
stable optimization (gradients backpropagate through the whole rollout).

At INFERENCE both models are identical: autoregressive rollout from a real
SEQ_LEN-day seed (evaluate.rollout). So comparing this script's rollout R2 to
nn_baseline's answers "does training the way you infer beat teacher forcing?".

Usage:
    python generate_synthetic_ode.py --n-extra 3000 --fast --output synthetic_ode.npz
    python nn_freerun.py --data synthetic_ode.npz --all-features
    # then compare the R2 table / parity to nn_baseline.py on the same data
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from device_utils import pick_device
from model import NextDayPredictor, N_FEATURES, SEQ_LEN, N_DAYS
from generate_synthetic_ode import WINDOW_FEATURE_INDICES
from evaluate import rollout

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer', 'Glucose',
                 'Glutamine', 'Asparagine', 'Serine', 'Glycine']


def rollout_train(model, seed, doe, n_steps):
    """Differentiable free-running rollout used during TRAINING.

    seed: (B, SEQ_LEN, N_FEATURES+1) normalized window incl. the time column.
    Feeds the model's own prediction back each step (no ground truth), so the
    whole rollout is one autograd graph. Returns (B, n_steps, N_FEATURES).
    """
    window = seed
    dt = 1.0 / (N_DAYS - 1)
    preds = []
    for _ in range(n_steps):
        pred = model(window, doe)                                 # (B, N_FEATURES)
        preds.append(pred)
        last_time = window[:, -1, N_FEATURES:N_FEATURES + 1]      # (B, 1)
        next_row = torch.cat([pred, last_time + dt], dim=1).unsqueeze(1)  # (B,1,F+1)
        window = torch.cat([window[:, 1:, :], next_row], dim=1)   # slide, feed own pred
    return torch.stack(preds, dim=1)                              # (B, n_steps, N_FEATURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='synthetic_ode.npz')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='fraction of extra reactors held out as validation RUNS')
    ap.add_argument('--feature', type=int, default=0, help='scored feature (0 = cell density)')
    ap.add_argument('--all-features', action='store_true',
                    help='score every feature and save the parity plot')
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN,
                    help='seed / input window length in days')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conv-layers', type=int, default=3)
    ap.add_argument('--tag', default='freerun',
                    help='suffix for the saved parity figure (avoids clobbering nn_baseline)')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device()
    here = Path(__file__).parent
    fig_tag = f'_{args.tag}' if args.tag else ''
    seq_len = args.seq_len
    n_pred = N_DAYS - seq_len

    npz = np.load(args.data, allow_pickle=True)
    doe_params = npz['doe_params'].astype(np.float32)
    doe_min = npz['doe_min'].astype(np.float32)
    doe_max = npz['doe_max'].astype(np.float32)
    dsc = doe_max - doe_min; dsc[dsc == 0] = 1.0
    traj = npz['trajectories'].astype(np.float32)
    n_original = int(npz['n_original'])
    sub = traj[:, :, WINDOW_FEATURE_INDICES].astype(np.float32)   # (N, N_DAYS, 8) absolute

    # --- train on the EXTRA reactors' FULL runs (originals reserved for eval);
    #     hold out a few whole runs as validation (run-level, not window-level) ---
    extra = sub[n_original:]                                       # (N_extra, N_DAYS, 8)
    doe_extra = ((doe_params[n_original:] - doe_min) / dsc).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(extra))
    n_val = max(1, int(round(len(extra) * args.val_frac)))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]

    # normalization fit on TRAINING-run concentrations only
    scaler = MinMaxScaler().fit(extra[tr_ids].reshape(-1, N_FEATURES))
    fmin = scaler.data_min_.astype(np.float32)
    fsc = (scaler.data_max_ - scaler.data_min_).astype(np.float32); fsc[fsc == 0] = 1.0

    def to_seed_and_target(runs):
        # (B, N_DAYS, 8) absolute -> seed (B, seq_len, 9) and target (B, n_pred, 8), normalized
        norm = np.clip((runs - fmin) / fsc, 0.0, 1.0).astype(np.float32)
        tcol = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))
        tcol = np.broadcast_to(tcol[None, :, None], (len(runs), seq_len, 1))
        seed = np.concatenate([norm[:, :seq_len, :], tcol], axis=2).astype(np.float32)
        target = norm[:, seq_len:, :].astype(np.float32)
        return seed, target

    tr_seed, tr_tgt = to_seed_and_target(extra[tr_ids])
    vl_seed, vl_tgt = to_seed_and_target(extra[val_ids])
    tr_seed_t = torch.from_numpy(tr_seed); tr_tgt_t = torch.from_numpy(tr_tgt)
    tr_doe_t = torch.from_numpy(doe_extra[tr_ids])
    vl_seed_t = torch.from_numpy(vl_seed).to(device)
    vl_tgt_t = torch.from_numpy(vl_tgt).to(device)
    vl_doe_t = torch.from_numpy(doe_extra[val_ids]).to(device)

    n_doe = doe_params.shape[1]
    model = NextDayPredictor(hidden=args.hidden, n_conv_layers=args.conv_layers,
                             n_doe=n_doe).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f'free-running training: {len(tr_ids)} train runs | {len(val_ids)} val runs | '
          f'seq_len={seq_len} | rollout {n_pred} steps (model feeds its own predictions back)')

    n_tr = len(tr_ids)
    best_val, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_tr)
        tot = 0.0
        for i in range(0, n_tr, args.batch):
            b = order[i:i + args.batch]
            seed = tr_seed_t[b].to(device); tgt = tr_tgt_t[b].to(device); d = tr_doe_t[b].to(device)
            opt.zero_grad()
            pred = rollout_train(model, seed, d, n_pred)          # (B, n_pred, 8)
            loss = ((pred - tgt) ** 2).mean()
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = ((rollout_train(model, vl_seed_t, vl_doe_t, n_pred) - vl_tgt_t) ** 2).mean().item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0 or ep == 1:
            print(f'epoch {ep:4d}  train={tot/n_tr:.5f}  val={vl:.5f}')
    model.load_state_dict(best_state)

    # --- evaluate: autoregressive rollout on the first n_original reactors (== nn_baseline) ---
    all_pred = np.zeros((n_original, n_pred, N_FEATURES), np.float32)
    for i in range(n_original):
        seed = np.clip((sub[i, :seq_len] - fmin) / fsc, 0.0, 1.0)
        tcol = (np.arange(seq_len, dtype=np.float32) / (N_DAYS - 1))[:, None]
        seed = np.concatenate([seed, tcol], axis=1)
        doe_i = (doe_params[i].astype(np.float32) - doe_min) / dsc
        pr = rollout(model, seed, n_steps=n_pred, device=device, doe=doe_i)
        all_pred[i] = pr * fsc + fmin                              # absolute units

    def score(f):
        pf = all_pred[:, :, f].ravel()
        tf = sub[:n_original, seq_len:, f].ravel()
        ss_tot = float(np.sum((tf - tf.mean()) ** 2))
        r2 = 1.0 - float(np.sum((pf - tf) ** 2)) / ss_tot if ss_tot > 0 else float('nan')
        maes, rhos, ranges, ratios = [], [], [], []
        for i in range(n_original):
            pred, real = all_pred[i, :, f], sub[i, seq_len:, f]
            rmax = real.max() if real.max() > 0 else 1.0
            ranges.append(float((real.max() - real.min()) / rmax))
            if np.std(pred) > 0 and np.std(real) > 0:
                rhos.append(float(np.corrcoef(pred, real)[0, 1]))
            maes.append(float(np.mean(np.abs(pred - real)) / rmax))
            ratios.append(float(pred.max() / rmax))
        return r2, np.mean(maes), np.mean(ratios), (np.mean(rhos) if rhos else float('nan')), np.mean(ranges)

    feats = range(N_FEATURES) if args.all_features else [args.feature]
    print('\nfree-running pure NN, autoregressive rollout vs synthetic truth '
          '(range = true signal span; R2/rho not meaningful when range ~ 0):')
    print(f'{"feature":>12} | {"R2":>6} {"peak":>5} {"normMAE":>8} {"rho":>5} {"range":>6}  note')
    print('-' * 60)
    stats = {}
    for f in feats:
        r2, m, p, rho, rng = score(f)
        stats[f] = (r2, m, p, rho, rng)
        flat = rng < 0.1
        r2s = f'{"n/a":>6}' if flat else f'{r2:>6.2f}'
        note = 'flat -> R2/rho N/A' if flat else ''
        print(f'{FEATURE_NAMES[f]:>12} | {r2s} {p:>5.2f} {m:>8.3f} {rho:>5.2f} {rng:>6.2f}  {note}')

    # parity plot (green points; tagged so it does not overwrite nn_baseline's)
    if args.all_features:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.flatten()
        for f in range(N_FEATURES):
            ax = axes[f]
            x = sub[:n_original, seq_len:, f].ravel(); y = all_pred[:, :, f].ravel()
            r2, m, p, rho, rng = stats[f]
            lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0; lo -= pad; hi += pad
            ax.plot([lo, hi], [lo, hi], color='0.5', lw=1, ls='--', zorder=1)
            ax.scatter(x, y, s=9, c='#3A4A2E', alpha=0.30, edgecolors='none', zorder=2)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
            r2str = 'n/a' if rng < 0.1 else f'{r2:.2f}'
            ax.set_title(f'{FEATURE_NAMES[f]}  R2={r2str}', fontsize=10)
            ax.set_xlabel('true (ODE)', fontsize=8); ax.set_ylabel('predicted (NN)', fontsize=8)
        fig.suptitle('Free-running training: predicted vs true, forecast window, all reactors',
                     y=1.0, fontsize=13)
        fig.tight_layout()
        out = here / f'nn_baseline_parity{fig_tag}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
