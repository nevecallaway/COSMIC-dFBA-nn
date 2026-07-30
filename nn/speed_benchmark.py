#!/usr/bin/env python3
"""
Speed benchmark: is the NN surrogate actually faster than solving the mechanistic
equations, and how much of that speedup does putting the ODE back in the forward
step give away?

METHODOLOGY (what to trust, what not to). A single GPU pass over N reactors takes
tens of microseconds, which is dominated by fixed overhead (kernel-launch latency,
the Python loop, cuda.synchronize), NOT by the arithmetic. Timed that way two fast
methods report the same number because both are at the overhead floor, not because
they do the same work. So every method here is timed the RIGHT way:

  * warm up once (load kernels / caches), then
  * run inference in a LOOP for a fixed wall-clock budget (--secs), and
  * amortize: per-reactor time = total_time / (reps * N).

With a big --n and a several-second budget, the timed region is almost entirely
real inference and the per-pass overhead washes out, so genuinely different
amounts of work show up as genuinely different numbers. All timing is in Python
(time.perf_counter around the compute), not wrapped in shell `time`.

Methods, for N reactors' worth of 13-day trajectories:

  1. mechanistic ODE, numerical solver (solve_ivp/RK45, CPU)  <- what the NN replaces
  2. mechanistic ODE, closed-form analytic step (numpy, CPU)  <- the fast generator
  3. mechanistic ODE, closed-form analytic step (torch, GPU)  <- ODE on the NN's device
  4. mechanistic ODE, torchdiffeq odeint (GPU)                <- fair same-device solver
  5. pure NN (NextDayPredictor, NO ODE)                       <- the surrogate
  6. hybrid NN (FluxDecoder, closed-form ODE in forward)
  7. hybrid NN (FluxDecoder, 50-substep Euler in forward)

Two speedup columns: vs the incumbent (scipy, CPU-only) AND vs torchdiffeq (a
general solver on the SAME device as the NN). The second isolates the algorithm
from the hardware -- it is the honest "surrogate vs solver" comparison.

Usage:
    python speed_benchmark.py --n 2000 --secs 5              # quick
    python speed_benchmark.py --n 20000 --secs 45 --device cuda   # real GPU run
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


def sustained(pass_fn, n, secs, warmup=2):
    """Warm up, then call pass_fn (one full pass over n reactors) in a loop until
    `secs` of wall time have elapsed. Returns (seconds_per_reactor, reps, elapsed).
    Amortizing over many reps is what pushes fixed per-pass overhead below the
    real compute, so fast methods stop colliding at the timer floor."""
    for _ in range(warmup):
        pass_fn()
    reps = 0
    t0 = time.perf_counter()
    while True:
        pass_fn()
        reps += 1
        if time.perf_counter() - t0 >= secs:
            break
    elapsed = time.perf_counter() - t0
    return elapsed / (reps * n), reps, elapsed


def mechanistic_pass(n, fast):
    """One pass = simulate n reactors with the mechanistic model (CPU)."""
    def one():
        for s in range(n):
            vg, vp, pm, doe = make_rates(s)
            generate_reactor('b', vg, vp, pm, doe, fast=fast)
    return one


def nn_pass(model, n, device, is_hybrid):
    """One pass = batched autoregressive rollout of n reactors over the trajectory."""
    steps = N_DAYS - SEQ_LEN
    seed = torch.rand(n, SEQ_LEN, N_FEATURES + 1, device=device)
    doe  = torch.rand(n, 3, device=device)
    cin  = torch.rand(n, N_FEATURES, device=device)
    model.eval()

    def one():
        with torch.no_grad():
            win = seed.clone()
            for _ in range(steps):
                if is_hybrid:
                    out, _ = model(win, doe, cin)
                else:
                    out = model(win, doe)
                nxt = torch.cat([out.unsqueeze(1), win[:, -1:, N_FEATURES:]], dim=2)
                win = torch.cat([win[:, 1:], nxt], dim=1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return one


def _ode_inputs(n, device):
    """Random but valid initial state, fluxes, feed, eta for a batch of reactors."""
    C = torch.rand(n, N_FEATURES, device=device) * 5 + 0.5
    C[:, 0] = torch.rand(n, device=device) * 2 + 0.5           # cell density = X
    v = (torch.rand(n, N_FEATURES, device=device) - 0.5) * 0.4
    v[:, 0] = torch.rand(n, device=device) * 0.4               # growth rate
    cin = torch.rand(n, N_FEATURES, device=device) * 10
    eta = torch.ones(n, 1, device=device)
    return C, v, cin, eta


def closedform_torch_pass(n, device):
    """One pass = iterate the exact closed-form one-day step over the trajectory,
    batched in torch on the NN's device -> isolates the ODE arithmetic."""
    C0, v, cin, eta = _ode_inputs(n, device)

    def one():
        with torch.no_grad():
            C = C0
            for _ in range(N_DAYS - 1):
                C = closed_form_step(C, v, cin, eta_titer=eta)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return one


def torchdiffeq_pass(n, device):
    """One pass = torchdiffeq odeint on the perfusion RHS (a general differentiable
    GPU solver). Returns None if torchdiffeq is not installed."""
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

    def one():
        with torch.no_grad():
            odeint(rhs, C0, t, method='dopri5')      # adaptive, differentiable
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000,
                    help='reactors per pass for the fast on-device methods (NN, '
                         'torch/torchdiffeq ODE); bump to 20000+ on a GPU')
    ap.add_argument('--n-solver', type=int, default=300,
                    help='reactors per pass for the CPU solvers (scipy / numpy '
                         'closed-form); small because they run one reactor at a time')
    ap.add_argument('--secs', type=float, default=5.0,
                    help='wall-clock budget PER METHOD; the pass repeats in a loop '
                         'until this elapses, then the time is amortized. Use ~45 '
                         'for a real measurement (spends the run doing inference)')
    ap.add_argument('--hidden', type=int, default=16,
                    help='body width, SAME for pure NN and hybrid so the comparison '
                         'isolates the ODE cost, not the network size')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f'Benchmark: n={args.n} (device methods) / n_solver={args.n_solver} (CPU), '
          f'{N_DAYS}-day trajectories, device={device}, hidden={args.hidden}, '
          f'budget={args.secs}s/method\n')

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

    # (label, pass_fn, n_for_method, warmup). CPU solvers use n_solver + light
    # warmup; on-device methods use n + a GPU warmup to load kernels.
    specs = [
        ('mechanistic: numerical ODE (solve_ivp, CPU)', mechanistic_pass(args.n_solver, False), args.n_solver, 1),
        ('mechanistic: closed-form ODE (numpy, CPU)',   mechanistic_pass(args.n_solver, True),  args.n_solver, 1),
        (f'mechanistic: closed-form ODE (torch, {device.type})', closedform_torch_pass(args.n, device), args.n, 2),
        (f'mechanistic: torchdiffeq odeint ({device.type})',     torchdiffeq_pass(args.n, device),      args.n, 2),
        ('pure NN (no ODE)',                        nn_pass(pure,   args.n, device, False), args.n, 2),
        ('hybrid NN (closed-form ODE in forward)',  nn_pass(hyb_cf, args.n, device, True),  args.n, 2),
        ('hybrid NN (50-substep Euler in forward)', nn_pass(hyb_eu, args.n, device, True),  args.n, 2),
    ]

    results = []                                     # (label, sec_per_reactor)
    for label, fn, nm, warm in specs:
        if fn is None:
            print(f'skip (not installed): {label}')
            continue
        print(f'timing {label} ...', flush=True)
        spr, reps, el = sustained(fn, nm, args.secs, warmup=warm)
        print(f'    {reps} reps x {nm} reactors in {el:.1f}s', flush=True)
        results.append((label, spr))

    spr = dict(results)
    t_num = spr.get('mechanistic: numerical ODE (solve_ivp, CPU)')
    tde_label = f'mechanistic: torchdiffeq odeint ({device.type})'
    t_tde = spr.get(tde_label)

    def col(base, s):
        return f'{base / s:>10.1f}x' if base is not None else f'{"-":>11}'

    print(f'\n{"method":<42} {"ms/reactor":>11} {"reactors/s":>12} '
          f'{"vs solve_ivp":>13} {"vs torchdiffeq":>15}')
    print('-' * 96)
    for label, s in results:
        print(f'{label:<42} {1000*s:>11.4f} {1/s:>12.0f} '
              f'{col(t_num, s):>13} {col(t_tde, s):>15}')

    print('\nBaselines: "vs solve_ivp" is the incumbent (scipy, CPU-only, no GPU '
          'version exists). "vs torchdiffeq" is a general solver on the SAME device '
          'as the NN -> isolates algorithm from hardware.')
    t_pure = spr.get('pure NN (no ODE)')
    if t_pure and t_tde:
        print(f'Same-device: the pure NN is {t_tde/t_pure:.1f}x faster than torchdiffeq '
              f'(the honest surrogate-vs-solver number), vs {t_num/t_pure:,.0f}x faster '
              f'than the CPU solver (which mostly reflects hardware).')


if __name__ == '__main__':
    main()
