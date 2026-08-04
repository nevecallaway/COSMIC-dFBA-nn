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
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from device_utils import pick_device
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
                horizon=None, seed=0, val_runs=None, patience=0, curve_csv=None):
    """
    Train on a rollout objective against a FIXED validation set.

    val_runs is a dedicated set of runs, disjoint from both the training pool and
    the evaluation holdout, and identical for every sweep size. That matters: a
    validation fraction carved out of the training runs would be 2 runs at
    n_train=10, far too noisy to early-stop on, and it would also shrink the
    training set so the sweep sizes were not what they claimed.

    patience > 0 enables early stopping on validation loss and restores the best
    weights. curve_csv writes the per-epoch train/val curves for plotting.
    """
    torch.manual_seed(seed)
    n_days = data['traj'].shape[1]
    n_feat = data['traj'].shape[2]

    fit_runs = np.asarray(train_runs)
    n_val = 0 if val_runs is None else len(val_runs)

    seeds, targets, fc, ff = build_rollout_data(data, seq_len, fit_runs)
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

    # validation tensors (same rollout objective, unseen runs)
    val_t = None
    if n_val:
        vs, vt, vfc, vff = build_rollout_data(data, seq_len, val_runs)
        val_t = (torch.from_numpy(((vs - fmin) / scale).astype(np.float32)).to(device),
                 torch.from_numpy(vfc).to(device),
                 torch.from_numpy(vff[:, :H]).to(device),
                 torch.from_numpy(((vt[:, :H] - fmin) / scale).astype(np.float32)).to(device))

    curves, best_val, best_state, best_ep, bad = [], float('inf'), None, 0, 0
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
        tr = tot / len(ds)

        vl = float('nan')
        if val_t is not None:
            model.eval()
            with torch.no_grad():
                vp = rollout(model, val_t[0], val_t[1], val_t[2], H, seq_len, n_days)
                vl = float(((vp - val_t[3]) ** 2).mean())
            if vl < best_val - 1e-9:
                best_val, best_ep, bad = vl, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        curves.append((ep, tr, vl))

        if ep % 20 == 0 or ep == 1:
            print(f'    epoch {ep:4d}  train={tr:.6f}  val={vl:.6f}', flush=True)
        if patience and bad >= patience:
            print(f'    early stop at epoch {ep} (best epoch {best_ep}, '
                  f'val={best_val:.6f})', flush=True)
            break

    if patience and best_state is not None:
        model.load_state_dict(best_state)
        print(f'    restored best weights from epoch {best_ep}', flush=True)

    if curve_csv:
        with open(curve_csv, 'w') as fh:
            fh.write('epoch,train_loss,val_loss\n')
            for e, t, v in curves:
                fh.write(f'{e},{t:.8f},{v:.8f}\n')
        print(f'    curves -> {curve_csv}', flush=True)

    return model, scaler, dict(best_epoch=best_ep, best_val=best_val,
                               final_train=curves[-1][1], final_val=curves[-1][2],
                               epochs_run=len(curves))


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
    ap.add_argument('--n-val', type=int, default=200,
                    help='size of the FIXED validation set, shared by every sweep '
                         'size and disjoint from training and the eval holdout')
    ap.add_argument('--patience', type=int, default=0,
                    help='early-stopping patience on val loss (0 = train all epochs)')
    ap.add_argument('--curve-dir', default=None,
                    help='write per-epoch train/val curves as CSV here')
    args = ap.parse_args()

    device = pick_device(args.device)

    data = load_fedbatch()
    R, D, F = data['traj'].shape
    print(f'Loaded {R} runs, features={data["features"]}, device={device}')

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(R)
    holdout = perm[:args.n_holdout]                       # scored, never trained on
    val_runs = perm[args.n_holdout:args.n_holdout + args.n_val]   # early stopping only
    pool = perm[args.n_holdout + args.n_val:]             # training draws from here
    sizes = [n for n in (args.sweep or [args.n_train]) if n <= len(pool)]
    H = (D - args.seq_len) if args.horizon is None else args.horizon
    print(f'Eval holdout: {len(holdout)} runs | validation: {len(val_runs)} runs '
          f'(fixed, shared across sizes) | train pool: {len(pool)}')
    print(f'Training rollout horizon: {H} days\n')

    if args.curve_dir:
        os.makedirs(args.curve_dir, exist_ok=True)

    print(f'{"n_train":>8} | {"day1 err":>8} {"rho":>6} {"MAE":>7} {"peak":>6} '
          f'{"over":>5} {"under":>6} {"div":>4} | {"bestEp":>6} {"ep_run":>6} '
          f'{"val_min":>9} {"val_end":>9}')
    print('-' * 104)
    results = []
    for n in sizes:
        print(f'  training on {n} runs...', flush=True)
        csv = (os.path.join(args.curve_dir, f'curve_n{n}.csv')
               if args.curve_dir else None)
        model, scaler, h = train_model(data, pool[:n], args.seq_len, args.hidden,
                                       args.epochs, args.lr, args.batch, device,
                                       args.horizon, args.seed, val_runs,
                                       args.patience, csv)
        r = evaluate(model, data, holdout, scaler, args.seq_len, device, args.feature)
        results.append((n, r, h))
        print(f'{n:>8} | {r["first"]:>8.3f} {r["rho"]:>6.2f} {r["mae"]:>7.3f} '
              f'{r["peak"]:>6.2f} {r["over"]:>5} {r["under"]:>6} {r["diverged"]:>4} | '
              f'{h["best_epoch"]:>6} {h["epochs_run"]:>6} {h["best_val"]:>9.5f} '
              f'{h["final_val"]:>9.5f}')

    # Overtraining readout: if val bottoms out early and then rises, the fixed
    # epoch budget was hurting that row.
    print('\nOvertraining check (val_min vs val_end, and how early val bottomed):')
    for n, _, h in results:
        frac = h['best_epoch'] / max(h['epochs_run'], 1)
        worse = (h['final_val'] - h['best_val']) / max(h['best_val'], 1e-12)
        flag = 'OVERTRAINED' if (frac < 0.6 and worse > 0.05) else 'ok'
        print(f'  n={n:<5} best epoch {h["best_epoch"]:>4}/{h["epochs_run"]:<4} '
              f'({frac:.0%} through)  val rose {worse:+.1%} after best   {flag}')

    print('\nAll columns are FULL-ROLLOUT forecasts on held-out runs (no teacher '
          'forcing anywhere): seed 6 real days, then predict from own predictions.')
    if len(results) > 1:
        (n0, r0, _), (n1, r1, _) = results[0], results[-1]
        print(f'  n={n0:<5} MAE={r0["mae"]:.3f} peak={r0["peak"]:.2f}   ->   '
              f'n={n1:<5} MAE={r1["mae"]:.3f} peak={r1["peak"]:.2f}')


if __name__ == '__main__':
    main()
