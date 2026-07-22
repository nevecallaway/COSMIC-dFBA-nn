#!/usr/bin/env python3
"""
Fed-batch training + held-out forecasting, with a data-scaling sweep.

This is the experiment that separates "our generalization limit is the data" from
"it is the method". The perfusion model had 10 reactors and could not pin down a
held-out reactor's productivity. Here the same architecture (small conv body ->
fluxes -> fixed fed-batch ODE) gets 10, 50, 200, 1000... runs against a FIXED
held-out set. If held-out accuracy improves with training-set size, the wall was
data. If it plateaus at 10, the wall was the method.

Runs are split disjointly, so a held-out run's trajectory never appears in
training: leakage-free by construction.

Evaluation is a true autoregressive forecast (seed the first seq_len days, then
predict from the model's own predictions), scored with the same metrics we used
for perfusion: trajectory correlation (shape), peak-normalized MAE and peak
ratio (magnitude).

Usage:
    python train_fedbatch.py --sweep 10 50 200 1000
    python train_fedbatch.py --n-train 1000 --epochs 60
"""
import argparse

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from fedbatch_data import load_fedbatch, build_fedbatch_windows
from model_fedbatch import FedBatchDecoder


def normalize_windows(win, scaler, n_feat):
    n, s, _ = win.shape
    feats, time = win[:, :, :n_feat], win[:, :, n_feat:]
    scaled = scaler.transform(feats.reshape(-1, n_feat)).reshape(n, s, n_feat)
    return np.concatenate([scaled, time], axis=2).astype(np.float32)


def train_model(data, train_runs, seq_len, hidden, epochs, lr, batch, device, seed=0):
    torch.manual_seed(seed)
    n_feat = data['traj'].shape[2]
    w, y, fc, ff, _ = build_fedbatch_windows(data, seq_len, runs=train_runs)

    scaler = MinMaxScaler().fit(np.vstack([w[:, :, :n_feat].reshape(-1, n_feat), y]))
    wn = normalize_windows(w, scaler, n_feat)
    yn = scaler.transform(y).astype(np.float32)

    model = FedBatchDecoder(n_features=n_feat, n_input_features=n_feat + 1,
                            hidden=hidden, n_doe=0).to(device)
    model.set_scaler(scaler)

    ds = TensorDataset(torch.from_numpy(wn), torch.from_numpy(fc),
                       torch.from_numpy(ff), torch.from_numpy(yn))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for xb, fcb, ffb, yb in loader:
            xb, fcb, ffb, yb = xb.to(device), fcb.to(device), ffb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb, None, fcb, ffb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 20 == 0 or ep == 1:
            print(f'    epoch {ep:4d}  loss={tot / len(ds):.6f}', flush=True)
    return model, scaler


@torch.no_grad()
def forecast_run(model, data, i, scaler, seq_len, device):
    """Autoregressive forecast of run i from its first seq_len days."""
    traj = data['traj'][i]
    D, F = traj.shape
    fmin  = scaler.data_min_.astype(np.float32)
    scale = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
    scale[scale == 0] = 1.0

    win = ((traj[:seq_len] - fmin) / scale).astype(np.float32)
    fc = torch.from_numpy(data['feed_conc'][i][None]).to(device)

    model.eval()
    out = []
    for t in range(seq_len, D):
        tcol = (np.arange(t - seq_len, t, np.float32) / (D - 1))[:, None]
        x = torch.from_numpy(np.concatenate([win, tcol], 1)[None]).to(device)
        ff = torch.tensor([[data['feed_frac'][i, t]]], dtype=torch.float32, device=device)
        nxt, _ = model(x, None, fc, ff)
        nxt = nxt.squeeze(0).cpu().numpy()
        out.append(nxt * scale + fmin)
        win = np.vstack([win[1:], nxt])
    return np.arange(seq_len, D), np.array(out)


def evaluate(model, data, holdout_runs, scaler, seq_len, device, feat):
    """Shape (rho) and magnitude (norm MAE, peak ratio) over held-out runs."""
    rhos, maes, ratios = [], [], []
    for i in holdout_runs:
        days, pred = forecast_run(model, data, i, scaler, seq_len, device)
        real = data['traj'][i][days, feat]
        p = pred[:, feat]
        rmax = data['traj'][i][:, feat].max()
        rmax = rmax if rmax > 0 else 1.0
        if len(p) > 1 and np.std(p) > 0 and np.std(real) > 0:
            rhos.append(float(np.corrcoef(p, real)[0, 1]))
        maes.append(float(np.mean(np.abs(p - real)) / rmax))
        ratios.append(float(p.max() / rmax))
    return (float(np.nanmean(rhos)), float(np.mean(maes)), float(np.mean(ratios)),
            int(sum(r > 1.1 for r in ratios)), int(sum(r < 0.9 for r in ratios)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', type=int, nargs='+', default=None,
                    help='training-set sizes to sweep, e.g. --sweep 10 50 200 1000')
    ap.add_argument('--n-train', type=int, default=1000)
    ap.add_argument('--n-holdout', type=int, default=200)
    ap.add_argument('--seq-len', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--feature', type=int, default=1, help='scored feature (1=Titer)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_fedbatch()
    R = data['traj'].shape[0]
    print(f'Loaded {R} runs, features={data["features"]}, device={device}')

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(R)
    holdout = perm[:args.n_holdout]
    pool    = perm[args.n_holdout:]
    sizes   = args.sweep if args.sweep else [args.n_train]
    sizes   = [n for n in sizes if n <= len(pool)]
    print(f'Held-out runs: {len(holdout)} (fixed across all training sizes)\n')

    print(f'{"n_train":>8} | {"shape rho":>9} {"norm MAE":>9} {"peak ratio":>10} '
          f'{"over":>5} {"under":>6}')
    print('-' * 56)
    results = []
    for n in sizes:
        print(f'  training on {n} runs...', flush=True)
        model, scaler = train_model(data, pool[:n], args.seq_len, args.hidden,
                                    args.epochs, args.lr, args.batch, device, args.seed)
        rho, mae, ratio, over, under = evaluate(model, data, holdout, scaler,
                                                args.seq_len, device, args.feature)
        results.append((n, rho, mae, ratio, over, under))
        print(f'{n:>8} | {rho:>9.2f} {mae:>9.3f} {ratio:>10.2f} {over:>5} {under:>6}')

    print('\nIf shape rho rises and norm MAE falls as n_train grows, the '
          'generalization limit is DATA, not method.')
    if len(results) > 1:
        first, last = results[0], results[-1]
        print(f'  n={first[0]:<5} rho={first[1]:.2f} MAE={first[2]:.3f}   ->   '
              f'n={last[0]:<5} rho={last[1]:.2f} MAE={last[2]:.3f}')


if __name__ == '__main__':
    main()
