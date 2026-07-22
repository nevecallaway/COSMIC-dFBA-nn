#!/usr/bin/env python3
"""
Fed-batch training + held-out forecasting, with a data-scaling sweep.

TRAINING AND EVALUATION USE THE SAME ROLLOUT. Each example seeds with the first
seq_len real days and then predicts every remaining day from the model's OWN
previous predictions, so by the last day the input window contains no real data
at all. The loss is taken over the whole predicted trajectory and gradients flow
back through every ODE step (the fed-batch step is exp/multiply/add, so this is
exact autograd, not an approximation).

Why this matters: one-step ("teacher forced") training optimizes a different task
than autoregressive forecasting. A slightly-too-high growth flux is a negligible
one-step error but compounds through exp(v*X) into a large overshoot by day 14.
Training on the rollout puts a gradient on exactly that drift.

The sweep trains on 10, 50, 200, 1000... runs against a FIXED held-out set. Runs
are split disjointly, so a held-out trajectory never appears in training.

Usage:
    python train_fedbatch.py --sweep 10 50 200 1000
    python train_fedbatch.py --horizon 3          # shorter rollout (curriculum)
"""
import argparse

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from fedbatch_data import load_fedbatch
from model_fedbatch import FedBatchDecoder


def build_rollout_data(data, seq_len, runs):
    """
    One example per run: seed on the first seq_len days, supervise every day after.

    Returns seeds (n, seq_len, F), targets (n, H, F), feed_conc (n, F),
            feed_frac (n, H)   where H = D - seq_len
    """
    runs = np.asarray(runs)
    traj = data['traj'][runs]
    return (traj[:, :seq_len].astype(np.float32),
            traj[:, seq_len:].astype(np.float32),
            data['feed_conc'][runs].astype(np.float32),
            data['feed_frac'][runs][:, seq_len:].astype(np.float32))


def rollout(model, seed_norm, feed_conc, feed_frac, horizon, seq_len, n_days):
    """
    Differentiable autoregressive rollout, shared by training and evaluation.

    seed_norm: (B, seq_len, F) normalized real seed days
    returns:   (B, horizon, F) normalized predictions
    """
    B = seed_norm.shape[0]
    dev = seed_norm.device
    win = seed_norm
    preds = []
    for k in range(horizon):
        days = torch.arange(k, k + seq_len, device=dev, dtype=torch.float32) / (n_days - 1)
        x = torch.cat([win, days.view(1, -1, 1).expand(B, -1, 1)], dim=2)
        nxt, _ = model(x, None, feed_conc, feed_frac[:, k:k + 1])
        preds.append(nxt)
        # feed the prediction back in: the window loses its oldest real day
        win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def _scale_arrays(scaler):
    fmin  = scaler.data_min_.astype(np.float32)
    scale = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
    scale[scale == 0] = 1.0
    return fmin, scale


def train_model(data, train_runs, seq_len, hidden, epochs, lr, batch, device,
                horizon=None, seed=0):
    torch.manual_seed(seed)
    n_days = data['traj'].shape[1]
    n_feat = data['traj'].shape[2]
    seeds, targets, fc, ff = build_rollout_data(data, seq_len, train_runs)
    H = targets.shape[1] if horizon is None else min(horizon, targets.shape[1])

    scaler = MinMaxScaler().fit(np.vstack([seeds.reshape(-1, n_feat),
                                           targets.reshape(-1, n_feat)]))
    fmin, scale = _scale_arrays(scaler)
    seeds_n   = ((seeds - fmin) / scale).astype(np.float32)
    targets_n = ((targets[:, :H] - fmin) / scale).astype(np.float32)

    model = FedBatchDecoder(n_features=n_feat, n_input_features=n_feat + 1,
                            hidden=hidden, n_doe=0).to(device)
    model.set_scaler(scaler)

    ds = TensorDataset(torch.from_numpy(seeds_n), torch.from_numpy(fc),
                       torch.from_numpy(ff[:, :H]), torch.from_numpy(targets_n))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for sb, fcb, ffb, yb in loader:
            sb, fcb, ffb, yb = (sb.to(device), fcb.to(device),
                                ffb.to(device), yb.to(device))
            opt.zero_grad()
            pred = rollout(model, sb, fcb, ffb, H, seq_len, n_days)
            loss = ((pred - yb) ** 2).mean()          # loss over the WHOLE rollout
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sb)
        if ep % 20 == 0 or ep == 1:
            print(f'    epoch {ep:4d}  rollout loss={tot / len(ds):.6f}', flush=True)
    return model, scaler


@torch.no_grad()
def evaluate(model, data, holdout_runs, scaler, seq_len, device, feat):
    """Same rollout as training, scored on held-out runs."""
    n_days = data['traj'].shape[1]
    seeds, targets, fc, ff = build_rollout_data(data, seq_len, holdout_runs)
    fmin, scale = _scale_arrays(scaler)
    H = targets.shape[1]

    model.eval()
    pred = rollout(model, torch.from_numpy((seeds - fmin) / scale).to(device),
                   torch.from_numpy(fc).to(device), torch.from_numpy(ff).to(device),
                   H, seq_len, n_days).cpu().numpy()
    pred = np.maximum(pred * scale + fmin, 0.0)       # physical, non-negative

    rhos, maes, ratios, first_err = [], [], [], []
    diverged = 0
    for j in range(len(holdout_runs)):
        p, real = pred[j, :, feat], targets[j, :, feat]
        rmax = max(float(targets[j, :, feat].max()), 1e-12)
        if not np.isfinite(p).all() or p.max() > 100 * rmax:
            diverged += 1
            continue
        first_err.append(abs(p[0] - real[0]) / rmax)
        if np.std(p) > 0 and np.std(real) > 0:
            rhos.append(float(np.corrcoef(p, real)[0, 1]))
        maes.append(float(np.mean(np.abs(p - real)) / rmax))
        ratios.append(float(p.max() / rmax))

    m = lambda a: float(np.mean(a)) if len(a) else float('nan')
    return dict(first=m(first_err), rho=m(rhos), mae=m(maes), peak=m(ratios),
                over=int(sum(r > 1.1 for r in ratios)),
                under=int(sum(r < 0.9 for r in ratios)), diverged=diverged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', type=int, nargs='+', default=None)
    ap.add_argument('--n-train', type=int, default=1000)
    ap.add_argument('--n-holdout', type=int, default=200)
    ap.add_argument('--seq-len', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--horizon', type=int, default=None,
                    help='rollout length during training (default: to the last day)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--feature', type=int, default=1, help='scored feature (1=Titer)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cpu')
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda(); device = torch.device('cuda')
            except RuntimeError as e:
                print(f'CUDA visible but unusable ({e.__class__.__name__}); using CPU.')
    else:
        device = torch.device(args.device)

    data = load_fedbatch()
    R, D, F = data['traj'].shape
    print(f'Loaded {R} runs, features={data["features"]}, device={device}')

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(R)
    holdout, pool = perm[:args.n_holdout], perm[args.n_holdout:]
    sizes = [n for n in (args.sweep or [args.n_train]) if n <= len(pool)]
    H = (D - args.seq_len) if args.horizon is None else args.horizon
    print(f'Held-out runs: {len(holdout)} (fixed)  |  training rollout horizon: {H} days\n')

    print(f'{"n_train":>8} | {"day1 err":>8} {"rho":>6} {"MAE":>7} {"peak":>6} '
          f'{"over":>5} {"under":>6} {"diverged":>9}')
    print('-' * 68)
    results = []
    for n in sizes:
        print(f'  training on {n} runs...', flush=True)
        model, scaler = train_model(data, pool[:n], args.seq_len, args.hidden,
                                    args.epochs, args.lr, args.batch, device,
                                    args.horizon, args.seed)
        r = evaluate(model, data, holdout, scaler, args.seq_len, device, args.feature)
        results.append((n, r))
        print(f'{n:>8} | {r["first"]:>8.3f} {r["rho"]:>6.2f} {r["mae"]:>7.3f} '
              f'{r["peak"]:>6.2f} {r["over"]:>5} {r["under"]:>6} {r["diverged"]:>9}')

    print('\nAll columns are FULL-ROLLOUT forecasts on held-out runs (no teacher '
          'forcing anywhere): seed 6 real days, then predict from own predictions.')
    if len(results) > 1:
        (n0, r0), (n1, r1) = results[0], results[-1]
        print(f'  n={n0:<5} MAE={r0["mae"]:.3f} peak={r0["peak"]:.2f}   ->   '
              f'n={n1:<5} MAE={r1["mae"]:.3f} peak={r1["peak"]:.2f}')


if __name__ == '__main__':
    main()
