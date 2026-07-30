#!/usr/bin/env python3
"""
Generate perturbed synthetic test data to evaluate model generalization.

Varies eta (titer retention) and perturbs growth/production rates to create
reactors the model has never seen. If the model predicts well on these,
it learned the physics. If not, it memorized the training distribution.

Usage:
    !python generate_perturbed_test.py
    !python generate_perturbed_test.py --n-test 50 --rate-noise 0.2
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp

from generate_synthetic_ode import (
    F, N_DAYS, T_EVAL, N_COMPONENTS, IDX_CD, IDX_CV, IDX_TIT, IDX_GLC,
    AAS_INDICES, C_NOMINAL, CIN_NOMINAL,
    load_rates, load_phase_fractions, load_doe, make_cin,
    WINDOW_FEATURE_INDICES, N_WINDOW_FEATURES, SEQ_LEN,
)


def _make_interval_ode_eta(v_net, cin, eta_titer):
    """ODE with configurable eta for titer retention."""
    def ode(t, C):
        X  = max(C[IDX_CD], 0.0)
        dC = np.zeros(N_COMPONENTS)

        dC[IDX_CD]  = v_net[IDX_CD] * X
        dC[IDX_CV]  = v_net[IDX_CV] * X
        dC[IDX_TIT] = v_net[IDX_TIT] * X - eta_titer * F * max(C[IDX_TIT], 0.0)

        for i in range(N_COMPONENTS):
            if i in (IDX_CD, IDX_CV, IDX_TIT):
                continue
            dC[i] = F * (cin[i] - C[i]) + v_net[i] * X

        return dC
    return ode


def generate_reactor_perturbed(reactor_id, v_growth, v_prod, pm_by_day, doe,
                                eta_titer=1.0):
    """Integrate ODE with perturbed eta."""
    cin = make_cin(doe)
    C   = C_NOMINAL.copy()
    trajectory = [C.copy()]

    n_intervals = N_DAYS - 1
    max_day     = max(pm_by_day.keys())
    f_fallback  = pm_by_day[max_day]

    for d in range(n_intervals):
        f_val   = pm_by_day.get(d + 1, f_fallback)
        v_net   = (1.0 - f_val) * v_growth + f_val * v_prod
        ode_fn  = _make_interval_ode_eta(v_net, cin, eta_titer)

        sol = solve_ivp(
            ode_fn, t_span=(0.0, 1.0), y0=C.copy(),
            method='RK45', t_eval=[1.0], max_step=0.1,
            rtol=1e-6, atol=1e-8,
        )

        C_next = sol.y[:, -1].copy() if sol.success else C.copy()
        C_next = np.clip(C_next, 0.0, None)
        trajectory.append(C_next)
        C = C_next

    return np.array(trajectory), T_EVAL


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-test', type=int, default=50)
    parser.add_argument('--rate-noise', type=float, default=0.2,
                        help='Fractional perturbation on rates (default 0.2 = ±20%%)')
    parser.add_argument('--eta-range', type=float, nargs=2, default=[0.5, 1.5],
                        help='Range for titer retention eta (default 0.5 1.5)')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--output', default=str(here / 'perturbed_test.npz'))
    args = parser.parse_args()

    data_dir = here / 'data'
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    rng = np.random.default_rng(args.seed)
    trajs, does, etas_out, rate_scales_out = [], [], [], []

    print(f'Generating {args.n_test} perturbed test reactors...')
    print(f'  Rate noise: ±{args.rate_noise:.0%}')
    print(f'  Eta range: {args.eta_range[0]:.1f} to {args.eta_range[1]:.1f}')

    for k in range(args.n_test):
        doe = {
            'O2':  float(rng.choice([-1, 0, 1])),
            'AAs': float(rng.uniform(-1, 1)),
            'Glc': float(rng.uniform(-1, 1)),
        }

        donor = reactor_ids[rng.integers(len(reactor_ids))]
        v_g = rates_growth[donor].copy()
        v_p = rates_prod[donor].copy()

        # Perturb rates by ±rate_noise
        rate_scale = rng.uniform(1.0 - args.rate_noise, 1.0 + args.rate_noise,
                                 size=N_COMPONENTS)
        v_g *= rate_scale
        v_p *= rate_scale

        # Random eta for titer retention
        eta = rng.uniform(args.eta_range[0], args.eta_range[1])

        traj, _ = generate_reactor_perturbed(
            f'test_{k:04d}', v_g, v_p,
            pm_dict[donor], doe, eta_titer=eta)

        cin = make_cin(doe)
        glc_conc = cin[IDX_GLC]
        aas_conc = float(sum(cin[i] for i in AAS_INDICES))

        trajs.append(traj)
        does.append([doe['O2'], glc_conc, aas_conc])
        etas_out.append(eta)
        rate_scales_out.append(rate_scale)

        if (k + 1) % 10 == 0:
            print(f'  {k + 1}/{args.n_test}')

    trajs = np.array(trajs)
    does  = np.array(does, dtype=np.float32)

    np.savez(
        args.output,
        trajectories=trajs,
        doe_params=does,
        etas=np.array(etas_out),
        rate_scales=np.array(rate_scales_out),
        n_original=0,
    )

    # Summary stats
    sub = trajs[:, :, WINDOW_FEATURE_INDICES]
    final_titers = sub[:, -1, 2]  # titer is index 2 in feature vector
    print(f'\nSaved {args.output}')
    print(f'  {len(trajs)} reactors, {N_DAYS} days each')
    print(f'  Titer range: {final_titers.min():.1f} to {final_titers.max():.1f} mg/L')
    print(f'  Eta range used: {min(etas_out):.3f} to {max(etas_out):.3f}')
    print(f'\nEvaluate with:')
    print(f'  !python evaluate.py --data {args.output} --n-eval {args.n_test}')


if __name__ == '__main__':
    main()
