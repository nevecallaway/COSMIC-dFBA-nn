#!/usr/bin/env python3
"""
Train the flux decoder on the REAL reactors (Kimberly's #3): fit the decoder to
the measured concentration shapes so the network learns fluxes that reproduce the
real trajectories. The embedded mass balance handles the perfusion correction
automatically, so this is the perfusion-aware version of Sarat's slope-to-rate
method (dC/dt has the F(cin - C) term subtracted before the flux is inferred).

Denormalization: data_2 is measured but normalized (per reactor/component / max).
We recover approximate physical units as

    real_phys = data2_norm * ODE_max

where ODE_max is the synthetic trajectory's peak for that reactor/component. The
SHAPE is real (from data_2); the SCALE is our best guess (raw data_2 is
proprietary). Non-model components keep their ODE values (unused by the model).

Modes:
  --init model_flux.pt   transfer: start from the synthetic-pretrained model (#2)
  (no --init)            from scratch on real data only (#3)

LORO: --holdout excludes reactors from training so held-out ones can be tested
      (evaluate.py --model model_real.pt --data <ode_npz> --eval-reactor H).

Usage:
    python train_real.py --holdout 3 4
    python train_real.py --init model_flux.pt --holdout 3 4
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from device_utils import pick_device
from generate_synthetic_ode import WINDOW_FEATURE_INDICES
from model_primeur import (FluxDecoder, N_FEATURES, N_INPUT_FEATURES, SEQ_LEN,
                           N_DAYS, ETA_SWITCH_DAY)
from generate_synthetic_ode import build_windows
from train_sample import FluxWindowDataset
from real_data import denormalize_data2


def perfusion_rollout(model, seed_norm, doe, cin, etas, horizon, seq_len, n_days):
    """
    Differentiable autoregressive rollout through the perfusion ODE step.

    Seeds on real days and then feeds each prediction back in, so by the last step
    the input window contains no real data. Identical in structure to how the
    forecast is scored, which is the whole point: train on what we measure.

    seed_norm (B, seq_len, F) normalized -> (B, horizon, F) normalized predictions
    """
    B = seed_norm.shape[0]
    dev = seed_norm.device
    win = seed_norm
    preds = []
    for k in range(horizon):
        days = torch.arange(k, k + seq_len, device=dev, dtype=torch.float32) / (n_days - 1)
        x = torch.cat([win, days.view(1, -1, 1).expand(B, -1, 1)], dim=2)
        nxt, _ = model(x, doe, cin, eta_ext=etas[:, k:k + 1])
        preds.append(nxt)
        win = torch.cat([win[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def build_rollout_tensors(real, reactors, doe_n, cin, seq_len, eta_day):
    """
    One rollout example per reactor: seed on the first seq_len days, supervise the
    rest. Returns numpy (seeds, targets, doe, cin, etas).

    etas[k] is the washout for the step ending on day seq_len+k, indexed by the
    SOURCE day to match generate_synthetic_ode and the forward pass.
    """
    sub = real[reactors][:, :, WINDOW_FEATURE_INDICES].astype(np.float32)
    horizon = sub.shape[1] - seq_len
    etas = np.array([[float((seq_len + k - 1) >= eta_day) for k in range(horizon)]],
                    dtype=np.float32).repeat(len(reactors), axis=0)
    return (sub[:, :seq_len], sub[:, seq_len:],
            doe_n[reactors].astype(np.float32), cin[reactors].astype(np.float32), etas)


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--ode-data', default=str(here / 'synthetic_ode.npz'),
                    help='ODE npz for the physical scale, DoE and feed')
    ap.add_argument('--init', default=None,
                    help='Pretrained checkpoint to start from (transfer). Omit = from scratch')
    ap.add_argument('--output', default=str(here / 'model_real.pt'))
    ap.add_argument('--holdout', type=int, nargs='+', default=[])
    ap.add_argument('--epochs', type=int,   default=300)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--batch',  type=int,   default=8,
                    help='small by design: with ~49 training windows, batch 32 gives only '
                         '2 gradient steps per epoch, batch 8 gives 7 (and more '
                         'stochasticity, which regularizes a tiny dataset)')
    ap.add_argument('--hidden', type=int,   default=32)   # small: few real windows
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--substeps', type=int, default=50)
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN,
                    help='Input window length (days) for flux prediction (default 6)')
    ap.add_argument('--freeze-conv', action='store_true',
                    help='Freeze the conv feature-extractor; fine-tune only attention + head '
                         '(prevents overfitting the few real windows, for transfer)')
    ap.add_argument('--stripped', action='store_true',
                    help='Use the low-capacity model_stripped decoder (1 conv layer, '
                         'linear head) instead of the full en Primeur body')
    ap.add_argument('--phase', action='store_true',
                    help='Phase-driven titer washout: eta = production fraction f(t) per '
                         'reactor/day (from data_2 phases) instead of the day-8 switch')
    ap.add_argument('--phase-threshold', type=float, default=None,
                    help='With --phase, step eta = 1 once f crosses this value (per reactor)')
    ap.add_argument('--val-reactors', type=int, default=2,
                    help='reactors held out of the fit for validation (0 = none). '
                         'Reactor-level, so it catches reactor generalization, which '
                         'is the failure mode we care about')
    ap.add_argument('--patience', type=int, default=0,
                    help='early-stopping patience on val loss (0 = train all epochs '
                         'and just log the curve)')
    ap.add_argument('--curve-csv', default=None,
                    help='write per-epoch train/val loss curve here')
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds weight init and batch shuffling. With ~7 training '
                         'reactors the run-to-run spread is large, so comparisons '
                         'between settings are only meaningful across several seeds')
    ap.add_argument('--gap-stop', type=float, default=None,
                    help='stop as soon as val_loss exceeds this multiple of train_loss, '
                         'i.e. when the val curve "lifts off" the train curve. This is a '
                         'gap criterion, not the usual "val stopped improving" one.')
    ap.add_argument('--residual-weight', type=float, default=0.0,
                    help='ODE-relaxation knob: 0 = pure hybrid (physics is a hard '
                         'layer); >0 lets the net add a free correction to bend away '
                         'from the ODE. Sweep it to trade real-data fit vs generalization')
    ap.add_argument('--residual-l2', type=float, default=0.0,
                    help='L2 penalty on the residual correction magnitude. Encourages the '
                         'net to correct the ODE only where it clearly helps, curbing the '
                         'overshoot the free residual introduces')
    ap.add_argument('--rollout', action='store_true',
                    help='Train on the autoregressive forecast instead of one-day-ahead: '
                         'seed the first seq_len real days, predict the rest from the '
                         "model's own predictions, loss over the whole trajectory. "
                         'Makes training, validation and reporting measure the same thing.')
    ap.add_argument('--eta-day', type=int, default=None,
                    help='day the titer washout switches on (default 8). The predicted '
                         'titer peak is pinned to this day by construction, so if the '
                         'real harvest is a day earlier this introduces a systematic lag')
    args = ap.parse_args()

    if args.eta_day is not None:
        import model_primeur, generate_synthetic_ode as _gen
        model_primeur.ETA_SWITCH_DAY = args.eta_day
        _gen.ETA_SWITCH_DAY = args.eta_day
        print(f'Eta switch day set to {args.eta_day} (default 8)')

    if args.stripped:
        from model_stripped import FluxDecoder as ModelClass
        print('Using stripped (low-capacity) decoder')
    else:
        ModelClass = FluxDecoder

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f'Device: {device}  seed: {args.seed}')

    npz = np.load(args.ode_data, allow_pickle=True)
    ode_traj   = npz['trajectories'].astype(np.float32)
    doe_params = npz['doe_params'].astype(np.float32)
    cin_params = npz['cin_params'].astype(np.float32)
    n_original = int(npz['n_original'])
    doe_min, doe_max = npz['doe_min'].astype(np.float32), npz['doe_max'].astype(np.float32)

    # ---- real physical trajectories from data_2 ----
    real = denormalize_data2(here / 'data' / 'data_2.csv', ode_traj, n_original)
    print(f'Denormalized {n_original} real reactors from data_2 (ODE-scaled).')

    # Phase-driven eta uses the real per-reactor production fraction f(t).
    phase_traj = npz['phases'].astype(np.float32)[:n_original] if args.phase else None
    if phase_traj is not None and args.phase_threshold is not None:
        phase_traj = (phase_traj > args.phase_threshold).astype(np.float32)
    windows, targets, wdoe, wcin, weta, ridx = build_windows(
        real, doe_params=doe_params[:n_original], cin_params=cin_params[:n_original],
        phase_traj=phase_traj, seq_len=args.seq_len)

    hold = set(args.holdout)
    keep = np.array([r not in hold for r in ridx])
    print(f'Training on real reactors {[i for i in range(n_original) if i not in hold]}; '
          f'holding out {sorted(hold)}')
    windows, targets, wdoe, wcin, weta = (windows[keep], targets[keep], wdoe[keep],
                                          wcin[keep], weta[keep])
    ridx = ridx[keep]

    # Reactor-level validation split, taken from the TRAINING reactors only.
    # With ~9 training reactors this is a small validation set and early stopping
    # on it is noisy; the per-epoch curve is the reliable output, and --patience
    # is opt-in rather than default for that reason.
    train_reactors = sorted(set(int(r) for r in ridx))
    n_val_r = min(args.val_reactors, max(len(train_reactors) - 1, 0))
    val_reactors = set(train_reactors[:n_val_r])
    fit_mask = np.array([int(r) not in val_reactors for r in ridx])
    val_mask = ~fit_mask
    if n_val_r:
        print(f'Validation reactors (held out of the fit): {sorted(val_reactors)}  '
              f'| fit windows={int(fit_mask.sum())} val windows={int(val_mask.sum())}')

    win_feats, win_time = windows[:, :, :N_FEATURES], windows[:, :, N_FEATURES:]

    # ---- scaler: reuse pretrained (transfer) or fit on real (from scratch) ----
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        scaler = ckpt['scaler']
    else:
        # fit on the FIT windows only, so validation does not leak into normalization
        scaler = MinMaxScaler().fit(
            np.vstack([win_feats[fit_mask].reshape(-1, N_FEATURES), targets[fit_mask]]))

    n, s, f = win_feats.shape
    win_scaled = scaler.transform(win_feats.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    windows_n = np.concatenate([win_scaled, win_time], axis=2)
    targets_n = scaler.transform(targets).astype(np.float32)
    doe_scale = doe_max - doe_min; doe_scale[doe_scale == 0] = 1.0
    wdoe_n = ((wdoe - doe_min) / doe_scale).astype(np.float32)

    ds = FluxWindowDataset(windows_n[fit_mask], wdoe_n[fit_mask], wcin[fit_mask],
                           weta[fit_mask], targets_n[fit_mask])
    val_ds = (FluxWindowDataset(windows_n[val_mask], wdoe_n[val_mask], wcin[val_mask],
                                weta[val_mask], targets_n[val_mask])
              if n_val_r else None)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    print(f'Real training windows: {len(ds)}')

    # ---- model ----
    hidden = args.hidden
    n_doe = wdoe_n.shape[1]
    if args.init:
        hidden = ckpt.get('hidden', hidden)
    model = ModelClass(hidden=hidden, n_doe=n_doe,
                       n_input_features=N_INPUT_FEATURES, n_substeps=args.substeps,
                       residual_weight=args.residual_weight).to(device)
    if args.init:
        model.load_state_dict(ckpt['model_state'])
        print(f'Transfer: initialized from {args.init}')
    model.set_scaler(scaler)

    if args.freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False
        print('Froze conv feature-extractor; fine-tuning attention + head only')

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in trainable)
    print(f'Trainable parameters: {n_params:,}  hidden={hidden}')

    # ---- rollout tensors (only used with --rollout) ----
    if args.rollout:
        eta_day = args.eta_day if args.eta_day is not None else ETA_SWITCH_DAY
        fit_r = [r for r in range(n_original) if r not in hold and r not in val_reactors]
        val_r = sorted(val_reactors)
        wcin_full = cin_params[:n_original]
        doe_full  = ((doe_params[:n_original] - doe_min) / doe_scale).astype(np.float32)

        def _pack(rs):
            s, t, dn, cn, et = build_rollout_tensors(real, rs, doe_full, wcin_full,
                                                     args.seq_len, eta_day)
            fmin  = scaler.data_min_.astype(np.float32)
            sc    = (scaler.data_max_ - scaler.data_min_).astype(np.float32); sc[sc == 0] = 1.0
            to = lambda a: torch.from_numpy(a).to(device)
            return (to(((s - fmin) / sc).astype(np.float32)), to(dn), to(cn), to(et),
                    to(((t - fmin) / sc).astype(np.float32)))

        roll_fit = _pack(fit_r)
        roll_val = _pack(val_r) if val_r else None
        horizon = roll_fit[4].shape[1]
        print(f'Rollout training: {len(fit_r)} reactors, horizon {horizon} days '
              f'({len(fit_r) * horizon} supervised day-predictions)')

    # ---- train (plain MSE; small model + weight decay for the tiny dataset) ----
    val_loader = (DataLoader(val_ds, batch_size=args.batch, shuffle=False)
                  if val_ds is not None and len(val_ds) else None)
    curves, best_val, best_state, best_ep, bad = [], float('inf'), None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0

        if args.rollout:
            s, dn, cn, et, y = roll_fit
            opt.zero_grad()
            pred = perfusion_rollout(model, s, dn, cn, et, horizon, args.seq_len, N_DAYS)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr = loss.item()
            vl = float('nan')
            if roll_val is not None:
                model.eval()
                with torch.no_grad():
                    vs, vd, vc, ve, vy = roll_val
                    vp = perfusion_rollout(model, vs, vd, vc, ve, horizon,
                                           args.seq_len, N_DAYS)
                    vl = float(((vp - vy) ** 2).mean())
                if vl < best_val - 1e-9:
                    best_val, best_ep, bad = vl, epoch, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
            curves.append((epoch, tr, vl))
            if epoch % 25 == 0 or epoch == 1:
                print(f'Epoch {epoch:4d}  train={tr:.5f}  val={vl:.5f}')
            if args.gap_stop and not np.isnan(vl) and vl > args.gap_stop * tr:
                print(f'Gap stop at epoch {epoch}: val={vl:.5f} > '
                      f'{args.gap_stop}x train={tr:.5f}')
                break
            if args.patience and bad >= args.patience:
                print(f'Early stop at epoch {epoch} (best {best_ep}, val={best_val:.5f})')
                break
            continue

        for x, d, cin, eta, y in loader:
            x, d, cin, eta, y = (x.to(device), d.to(device), cin.to(device),
                                 eta.to(device), y.to(device))
            opt.zero_grad()
            pred, _ = model(x, d, cin, eta_ext=eta)
            loss = ((pred - y) ** 2).mean()
            if args.residual_l2 > 0 and model._last_residual is not None:
                loss = loss + args.residual_l2 * (model._last_residual ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        tr = tot / len(ds)

        vl = float('nan')
        if val_loader is not None:
            model.eval(); vtot = 0.0
            with torch.no_grad():
                for x, d, cin, eta, y in val_loader:
                    x, d, cin, eta, y = (x.to(device), d.to(device), cin.to(device),
                                         eta.to(device), y.to(device))
                    pred, _ = model(x, d, cin, eta_ext=eta)
                    vtot += ((pred - y) ** 2).mean().item() * len(x)
            vl = vtot / len(val_ds)
            if vl < best_val - 1e-9:
                best_val, best_ep, bad = vl, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        curves.append((epoch, tr, vl))

        if epoch % 25 == 0 or epoch == 1:
            print(f'Epoch {epoch:4d}  train={tr:.5f}  val={vl:.5f}')
        if args.gap_stop and not np.isnan(vl) and vl > args.gap_stop * tr:
            print(f'Gap stop at epoch {epoch}: val={vl:.5f} > '
                  f'{args.gap_stop}x train={tr:.5f}')
            break
        if args.patience and bad >= args.patience:
            print(f'Early stop at epoch {epoch} (best {best_ep}, val={best_val:.5f})')
            break

    if args.patience and best_state is not None:
        model.load_state_dict(best_state)
        print(f'Restored best weights from epoch {best_ep}')
    if curves and not np.isnan(curves[-1][2]):
        frac = best_ep / max(len(curves), 1)
        worse = (curves[-1][2] - best_val) / max(best_val, 1e-12)
        print(f'Val bottomed at epoch {best_ep}/{len(curves)} ({frac:.0%} through); '
              f'val rose {worse:+.1%} after best'
              + ('   <-- OVERTRAINED' if (frac < 0.6 and worse > 0.05) else ''))
    if args.curve_csv:
        with open(args.curve_csv, 'w') as fh:
            fh.write('epoch,train_loss,val_loss\n')
            for e, t, v in curves:
                fh.write(f'{e},{t:.8f},{v:.8f}\n')
        print(f'Curve -> {args.curve_csv}')

    torch.save({
        'model_state': model.state_dict(), 'scaler': scaler,
        'doe_min': doe_min, 'doe_max': doe_max, 'hidden': hidden,
        'n_features': N_FEATURES, 'n_input_features': N_INPUT_FEATURES,
        'seq_len': args.seq_len, 'n_doe': n_doe, 'n_substeps': args.substeps,
        'arch': 'stripped' if args.stripped else 'primeur',
        'residual_weight': args.residual_weight,
        'phase': args.phase,
    }, args.output)
    print(f'Saved to {args.output}')
    print(f'Evaluate held-out: python evaluate.py --model {Path(args.output).name} '
          f'--data {Path(args.ode_data).name} --eval-reactor {sorted(hold)[0] if hold else 0}')


if __name__ == '__main__':
    main()
