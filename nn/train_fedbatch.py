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
        tcol = (np.arange(t - seq_len, t, dtype=np.float32) / (D - 1))[:, None]
        x = torch.from_numpy(np.concatenate([win, tcol], 1)[None]).to(device)
        ff = torch.tensor([[data['feed_frac'][i, t]]], dtype=torch.float32, device=device)
        nxt, _ = model(x, None, fc, ff)
        nxt = nxt.squeeze(0).cpu().numpy()
        phys = nxt * scale + fmin
        # Concentrations and cell density cannot be negative. This is physics, not
        # an arbitrary cap, and it keeps a diverging rollout from going imaginary.
        phys = np.maximum(phys, 0.0)
        out.append(phys)
        if not np.isfinite(phys).all():
            break                                  # diverged; caller detects it
        win = np.vstack([win[1:], (phys - fmin) / scale])
    return np.arange(seq_len, seq_len + len(out)), np.array(out)


@torch.no_grad()
def evaluate_onestep(model, data, holdout_runs, scaler, seq_len, device, feat):
    """
    Teacher-forced one-step error on HELD-OUT runs: predict each day from the
    real previous window. This is pure generalization, with no rollout
    compounding, so it isolates 'did more data help the model learn' from
    'is the autoregressive rollout numerically stable'.
    """
    n_feat = data['traj'].shape[2]
    w, y, fc, ff, _ = build_fedbatch_windows(data, seq_len, runs=holdout_runs)
    wn = normalize_windows(w, scaler, n_feat)
    model.eval()
    pred, _ = model(torch.from_numpy(wn).to(device), None,
                    torch.from_numpy(fc).to(device), torch.from_numpy(ff).to(device))
    fmin  = scaler.data_min_.astype(np.float32)
    scale = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
    scale[scale == 0] = 1.0
    p = pred.cpu().numpy() * scale + fmin
    denom = np.abs(y[:, feat]).mean()
    return float(np.abs(p[:, feat] - y[:, feat]).mean() / max(denom, 1e-12))


def evaluate(model, data, holdout_runs, scaler, seq_len, device, feat):
    """Rollout forecast quality over held-out runs, divergence-aware."""
    rhos, maes, ratios = [], [], []
    diverged = 0
    for i in holdout_runs:
        days, pred = forecast_run(model, data, i, scaler, seq_len, device)
        p = pred[:, feat]
        real = data['traj'][i][days, feat]
        rmax = data['traj'][i][:, feat].max()
        rmax = rmax if rmax > 0 else 1.0
        # A rollout counts as diverged if it went non-finite, stopped early, or
        # ran away past a physically absurd multiple of the real peak.
        if (len(p) != len(data['traj'][i]) - seq_len or not np.isfinite(p).all()
                or p.max() > 100 * rmax):
            diverged += 1
            continue
        if len(p) > 1 and np.std(p) > 0 and np.std(real) > 0:
            rhos.append(float(np.corrcoef(p, real)[0, 1]))
        maes.append(float(np.mean(np.abs(p - real)) / rmax))
        ratios.append(float(p.max() / rmax))

    nanmean = lambda a: float(np.mean(a)) if len(a) else float('nan')
    return (nanmean(rhos), nanmean(maes), nanmean(ratios),
            int(sum(r > 1.1 for r in ratios)), int(sum(r < 0.9 for r in ratios)),
            diverged)


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
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                    help='auto uses cuda when usable; the model is small enough '
                         'that cpu is perfectly fine')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cpu')
        if torch.cuda.is_available():
            try:                       # cuda can be "visible" but not allocated to us
                torch.zeros(1).cuda()
                device = torch.device('cuda')
            except RuntimeError as e:
                print(f'CUDA visible but unusable ({e.__class__.__name__}); using CPU. '
                      f'Request a GPU with --gres=gpu:1 to use it.')
    else:
        device = torch.device(args.device)
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

    print(f'{"n_train":>8} | {"1step MAE":>9} | {"rho":>6} {"MAE":>7} {"peak":>6} '
          f'{"over":>5} {"under":>6} {"diverged":>9}')
    print('-' * 72)
    results = []
    for n in sizes:
        print(f'  training on {n} runs...', flush=True)
        model, scaler = train_model(data, pool[:n], args.seq_len, args.hidden,
                                    args.epochs, args.lr, args.batch, device, args.seed)
        one = evaluate_onestep(model, data, holdout, scaler, args.seq_len,
                               device, args.feature)
        rho, mae, ratio, over, under, div = evaluate(model, data, holdout, scaler,
                                                     args.seq_len, device, args.feature)
        results.append((n, one, rho, mae, ratio, over, under, div))
        print(f'{n:>8} | {one:>9.4f} | {rho:>6.2f} {mae:>7.3f} {ratio:>6.2f} '
              f'{over:>5} {under:>6} {div:>9}')

    print('\n1step MAE = teacher-forced held-out error (pure generalization, no '
          'rollout compounding).')
    print('The rollout columns additionally require the autoregressive forecast '
          'to stay stable.')
    if len(results) > 1:
        f, l = results[0], results[-1]
        print(f'\n  n={f[0]:<5} 1step={f[1]:.4f}   ->   n={l[0]:<5} 1step={l[1]:.4f}')
        if l[1] < f[1]:
            print('  Held-out one-step error FELL with more data: the '
                  'generalization limit is DATA, not method.')
        else:
            print('  Held-out one-step error did NOT fall: more data is not the '
                  'binding constraint.')


if __name__ == '__main__':
    main()
