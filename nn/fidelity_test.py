#!/usr/bin/env python3
"""
Fidelity test for the flux-decoder ODE step (model_primeur.ode_step).

Before training the decoder, confirm its ODE-step layer reproduces the
generator's trajectory when given the TRUE per-cell fluxes. If this fails, the
physics layer is wrong and no training will fix it.

Teacher-forced check: for each 1-day interval of each real reactor, feed the
true current state and the true blended flux v_net = (1-f)*v_growth + f*v_prod
into ode_step, and compare the predicted next-day values against
generate_reactor's own output, on the 8 model features.

The 8 model features form a closed subsystem: each metabolite equation depends
only on itself and cell density (feature 0), which is itself one of the 8. So
the step can be reproduced exactly given the true fluxes; any residual is the
Euler-substep vs generator-RK45 discretization difference.

Note: with the current lean C_NOMINAL the real reactors deplete, so the
generator prints [ODE NEG] lines. Those are expected and irrelevant here; the
comparison is between ode_step and the generator on identical values.

Usage:
    python fidelity_test.py
    python fidelity_test.py --substeps 20     # tighten Euler integration
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from generate_synthetic_ode import (
    load_rates, load_phase_fractions, load_doe,
    generate_reactor, make_cin, N_DAYS,
)
from model import FEATURE_INDICES
from model_primeur import ode_step, closed_form_step

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer', 'Glucose',
                 'Glutamine', 'Asparagine', 'Serine', 'Glycine']


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(here / 'data'))
    parser.add_argument('--substeps', type=int, default=10)
    parser.add_argument('--closed', action='store_true',
                        help='Test the exact closed-form step instead of Euler substeps')
    args = parser.parse_args()

    data_dir = Path(args.data)
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict  = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    fidx = np.array(FEATURE_INDICES)          # component indices for the 8 features
    per_feat_max = np.zeros(len(fidx))
    overall_max  = 0.0

    for reactor in reactor_ids:
        v_growth = rates_growth[reactor]
        v_prod   = rates_prod[reactor]
        pm_by_day = pm_dict[reactor]
        doe = doe_dict.get(reactor, {'O2': 0, 'AAs': 0, 'Glc': 0})
        cin8 = make_cin(doe)[fidx]

        traj, _ = generate_reactor(reactor, v_growth, v_prod, pm_by_day, doe)

        f_fallback = pm_by_day[max(pm_by_day.keys())]
        for d in range(N_DAYS - 1):
            f     = pm_by_day.get(d + 1, f_fallback)
            v_net = (1.0 - f) * v_growth + f * v_prod

            C_curr8      = traj[d,     fidx]
            C_true_next8 = traj[d + 1, fidx]

            C_curr_t = torch.tensor(C_curr8[None], dtype=torch.float64)
            v_t      = torch.tensor(v_net[fidx][None], dtype=torch.float64)
            cin_t    = torch.tensor(cin8[None], dtype=torch.float64)
            if args.closed:
                C_pred = closed_form_step(C_curr_t, v_t, cin_t).numpy()[0]
            else:
                C_pred = ode_step(C_curr_t, v_t, cin_t, n_substeps=args.substeps).numpy()[0]

            denom = np.maximum(np.abs(C_true_next8), 1e-6)
            rel = np.abs(C_pred - C_true_next8) / denom
            per_feat_max = np.maximum(per_feat_max, rel)
            overall_max  = max(overall_max, rel.max())

    method = 'closed-form step' if args.closed else f'{args.substeps} Euler substeps'
    print(f'Fidelity test: {method} vs generator\n')
    print(f'  {"Feature":<14} {"max rel err":>12}')
    print('  ' + '-' * 28)
    for name, e in zip(FEATURE_NAMES, per_feat_max):
        print(f'  {name:<14} {e:>11.3%}')
    print('  ' + '-' * 28)
    print(f'  {"OVERALL":<14} {overall_max:>11.3%}')

    tol = 0.02
    if overall_max <= tol:
        print(f'\nPASS: within {tol:.0%}. ODE-step layer reproduces the generator.')
    else:
        print(f'\nHIGH ({overall_max:.2%} > {tol:.0%}): raise --substeps and re-run. '
              f'If it does not fall, the layer has a bug.')


if __name__ == '__main__':
    main()
