#!/usr/bin/env python3
"""
Speed benchmark: is the NN surrogate actually faster than solving the mechanistic
equations, and how much of that speedup does putting the ODE back in the forward
step give away?

The whole point of a surrogate is speed. This times, for N reactors' worth of
13-day trajectories:

  1. mechanistic ODE, numerical solver (solve_ivp/RK45)  <- what the NN replaces
  2. mechanistic ODE, closed-form analytic step          <- the fast generator
  3. pure NN (NextDayPredictor, NO ODE)                  <- the surrogate
  4. hybrid NN (FluxDecoder, closed-form ODE in forward)
  5. hybrid NN (FluxDecoder, 50-substep Euler in forward)

Reported as total time, per-reactor time, and speedup vs the numerical solver.
The NN also batches: it predicts all N reactors in one vectorized pass, while the
solver is inherently one-reactor-at-a-time.

Usage:
    python speed_benchmark.py --n 1000
    python speed_benchmark.py --n 1000 --device cpu
"""
import argparse
import time

import numpy as np
import torch

from device_utils import pick_device
from generate_synthetic_ode import generate_reactor, N_COMPONENTS, N_DAYS
from model import NextDayPredictor, N_FEATURES, SEQ_LEN
from model_primeur import FluxDecoder, closed_form_step, F_PERFUSION


def make_rates(seed):
    rng = np.random.default_rng(seed)
    vg = rng.uniform(-0.3, 0.3, N_COMPONENTS); vg[0] = 0.30      # cell-density growth
    vp = rng.uniform(-0.3, 0.3, N_COMPONENTS); vp[0] = 0.05
    pm = {d: min(1.0, d / N_DAYS) for d in range(1, N_DAYS + 1)}
    doe = {'O2': 0.0, 'AAs': 0.0, 'Glc': 0.0}
    return vg, vp, pm, doe


def time_mechanistic(n, fast, warmup=2):
    for s in range(warmup):
        vg, vp, pm, doe = make_rates(s)
        generate_reactor('w', vg, vp, pm, doe, fast=fast)
    t0 = time.perf_counter()
    for s in range(n):
        vg, vp, pm, doe = make_rates(s)
        generate_reactor('b', vg, vp, pm, doe, fast=fast)
    return time.perf_counter() - t0


@torch.no_grad()
def time_nn(model, n, device, is_hybrid):
    """Batched autoregressive rollout of n reactors over the full trajectory."""
    steps = N_DAYS - SEQ_LEN
    seed = torch.rand(n, SEQ_LEN, N_FEATURES + 1, device=device)
    doe  = torch.rand(n, 3, device=device)
    cin  = torch.rand(n, N_FEATURES, device=device)
    model.eval()

    def one_pass():
        win = seed.clone()
        for _ in range(steps):
            if is_hybrid:
                out, _ = model(win, doe, cin)
            else:
                out = model(win, doe)
            # out is (N, features); add a time step and carry the time column forward
            nxt = torch.cat([out.unsqueeze(1), win[:, -1:, N_FEATURES:]], dim=2)
            win = torch.cat([win[:, 1:], nxt], dim=1)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    one_pass(); one_pass()                       # warmup
    t0 = time.perf_counter()
    one_pass()
    return time.perf_counter() - t0


def _ode_inputs(n, device):
    """Random but valid initial state, fluxes, feed, eta for a batch of reactors."""
    C = torch.rand(n, N_FEATURES, device=device) * 5 + 0.5
    C[:, 0] = torch.rand(n, device=device) * 2 + 0.5           # cell density = X
    v = (torch.rand(n, N_FEATURES, device=device) - 0.5) * 0.4
    v[:, 0] = torch.rand(n, device=device) * 0.4               # growth rate
    cin = torch.rand(n, N_FEATURES, device=device) * 10
    eta = torch.ones(n, 1, device=device)
    return C, v, cin, eta


@torch.no_grad()
def time_closedform_torch(n, device):
    """GPU-native (batched, torch) closed-form ODE: iterate the exact one-day step
    over the trajectory. Same device as the NN -> isolates the ODE cost."""
    C0, v, cin, eta = _ode_inputs(n, device)

    def one_pass():
        C = C0
        for _ in range(N_DAYS - 1):
            C = closed_form_step(C, v, cin, eta_titer=eta)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    one_pass(); one_pass()                       # warmup
    t0 = time.perf_counter(); one_pass()
    return time.perf_counter() - t0


def time_torchdiffeq(n, device):
    """General differentiable GPU solver (torchdiffeq odeint) on the perfusion RHS.
    Returns None if torchdiffeq is not installed."""
    try:
        from torchdiffeq import odeint
    except ImportError:
        return None
    C0, v, cin, eta = _ode_inputs(n, device)
    F = F_PERFUSION

    def rhs(t, C):
        X = C[:, 0:1]
        d_cd  = v[:, 0:1] * X
        d_sz  = v[:, 1:2] * X
        d_tit = v[:, 2:3] * X - eta * F * C[:, 2:3]
        d_met = F * (cin[:, 3:8] - C[:, 3:8]) + v[:, 3:8] * X
        return torch.cat([d_cd, d_sz, d_tit, d_met], dim=1)

    t = torch.linspace(0, N_DAYS - 1, N_DAYS, device=device)

    def one_pass():
        odeint(rhs, C0, t, method='dopri5')      # adaptive, differentiable
        if device.type == 'cuda':
            torch.cuda.synchronize()

    one_pass()
    t0 = time.perf_counter(); one_pass()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000, help='reactors to simulate/predict')
    ap.add_argument('--hidden', type=int, default=16,
                    help='body width, SAME for pure NN and hybrid so the comparison '
                         'isolates the ODE cost, not the network size')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f'Benchmark: {args.n} reactors, {N_DAYS}-day trajectories, device={device}, '
          f'matched hidden={args.hidden}\n')

    print('timing mechanistic (numerical solve_ivp)...', flush=True)
    t_num = time_mechanistic(args.n, fast=False)
    print('timing mechanistic (closed-form)...', flush=True)
    t_cf = time_mechanistic(args.n, fast=True)

    # Matched architecture: same hidden width and a single conv layer for all,
    # so any timing difference is the ODE step, not the network.
    H = args.hidden
    pure   = NextDayPredictor(hidden=H, n_conv_layers=1, n_doe=3).to(device)
    hyb_cf = FluxDecoder(hidden=H, n_conv_layers=1, n_doe=3, integrator='closed').to(device)
    hyb_eu = FluxDecoder(hidden=H, n_conv_layers=1, n_doe=3,
                         integrator='euler', n_substeps=50).to(device)
    for m in (hyb_cf, hyb_eu):
        class _S:
            data_min_ = np.zeros(N_FEATURES, np.float32)
            data_max_ = np.full(N_FEATURES, 10.0, np.float32)
        m.set_scaler(_S())

    print('timing GPU-native ODE (batched torch closed-form)...', flush=True)
    t_cft = time_closedform_torch(args.n, device)
    print('timing torchdiffeq (if installed)...', flush=True)
    t_tde = time_torchdiffeq(args.n, device)

    print('timing NN variants (batched)...', flush=True)
    t_pure = time_nn(pure,   args.n, device, is_hybrid=False)
    t_hcf  = time_nn(hyb_cf, args.n, device, is_hybrid=True)
    t_heu  = time_nn(hyb_eu, args.n, device, is_hybrid=True)

    rows = [
        ('mechanistic: numerical ODE (solve_ivp, CPU)', t_num),
        ('mechanistic: closed-form ODE (numpy, CPU)',   t_cf),
        (f'mechanistic: closed-form ODE (torch, {device.type})', t_cft),
    ]
    if t_tde is not None:
        rows.append((f'mechanistic: torchdiffeq odeint ({device.type})', t_tde))
    rows += [
        ('pure NN (no ODE)',                        t_pure),
        ('hybrid NN (closed-form ODE in forward)',  t_hcf),
        ('hybrid NN (50-substep Euler in forward)', t_heu),
    ]
    print(f'\n{"method":<42} {"total s":>9} {"ms/reactor":>11} {"vs solve_ivp":>13}')
    print('-' * 78)
    for name, t in rows:
        print(f'{name:<42} {t:>9.3f} {1000*t/args.n:>11.3f} {t_num/t:>12.1f}x')

    print(f'\nHeadline: the pure NN is {t_num/t_pure:,.0f}x faster than the numerical '
          f'solver it replaces.')
    print(f'Putting the ODE back in the forward step costs '
          f'{t_hcf/t_pure:.1f}x (closed-form) to {t_heu/t_pure:.1f}x (Euler) '
          f'of that speed, so a hybrid trades some speed for the mass-balance guarantee.')


if __name__ == '__main__':
    main()
