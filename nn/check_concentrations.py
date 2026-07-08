#!/usr/bin/env python3
"""
Diagnostic: find minimum uniform scale factor on all amino acid concentrations
(applied to both initial conditions and feed) such that no concentration goes
negative, across all 10 real reactors and all 3 AA DoE levels (-1, 0, +1).

Usage:
    python check_concentrations.py              # show current min values
    python check_concentrations.py --sweep      # sweep scale factors
    python check_concentrations.py --scale 5.0  # test a specific scale factor
"""

import argparse
import numpy as np
from pathlib import Path

from generate_synthetic_ode import (
    load_rates, load_phase_fractions, load_doe,
    generate_reactor, C_NOMINAL, AAS_INDICES,
)

AA_DOE_LEVELS = [-1, 0, 1]

COMPONENT_NAMES = [
    'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid',
    'L-Serine', 'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine',
    'L-Histidine', 'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine',
    'L-Tyrosine', 'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine',
    'L-Tryptophan',
]


def run_reactors(scale, rates_growth, rates_prod, reactor_ids, pm_dict, doe_dict,
                 nh4_feed=None):
    """
    Run all reactors at each AA DoE level with amino acid concentrations
    scaled uniformly in both C_NOMINAL (initial) and CIN_NOMINAL (feed).
    Optionally override CIN_NOMINAL[NH4] with nh4_feed (mmol/L).
    Returns {(reactor, doe_level): trajectory array}.
    """
    import generate_synthetic_ode as _g

    orig_nom = _g.C_NOMINAL.copy()
    orig_cin = _g.CIN_NOMINAL.copy()

    _g.C_NOMINAL[AAS_INDICES]   = orig_nom[AAS_INDICES] * scale
    _g.CIN_NOMINAL[AAS_INDICES] = orig_cin[AAS_INDICES] * scale

    if nh4_feed is not None:
        _g.CIN_NOMINAL[4] = nh4_feed  # IDX_NH4 = 4

    results = {}
    for reactor in reactor_ids:
        base_doe = doe_dict.get(reactor, {'O2': 0, 'AAs': 0, 'Glc': 0})
        pm_days  = pm_dict.get(reactor)
        if pm_days is None:
            continue
        for aa_level in AA_DOE_LEVELS:
            doe = {**base_doe, 'AAs': float(aa_level)}
            traj, _ = generate_reactor(reactor, rates_growth[reactor],
                                       rates_prod[reactor], pm_days, doe)
            results[(reactor, aa_level)] = traj

    _g.C_NOMINAL[:]   = orig_nom
    _g.CIN_NOMINAL[:] = orig_cin

    return results


def check(results, verbose=True):
    """
    Return True if no concentration goes negative across all reactors and conditions.
    Prints the worst offenders when verbose=True.
    """
    any_neg = False
    for (reactor, aa_level), traj in results.items():
        neg = traj < 0
        if neg.any():
            any_neg = True
            if verbose:
                for idx in np.where(neg.any(axis=0))[0]:
                    worst = traj[:, idx].min()
                    print(f'  NEG  reactor={reactor}  AA_doe={aa_level:+d}  '
                          f'{COMPONENT_NAMES[idx]}  min={worst:.4f} mmol/L')
    return not any_neg


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default=str(here / 'data'))
    parser.add_argument('--scale', type=float, default=None,
                        help='Test a specific uniform AA scale factor')
    parser.add_argument('--nh4', type=float, default=None,
                        help='Override CIN[NH4] feed concentration (mmol/L)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep scale factors to find minimum non-negative value')
    args = parser.parse_args()

    data_dir = Path(args.data)
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict  = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    print(f'Reactors: {reactor_ids}')
    print('Current AA concentrations (DMEM base, mmol/L):')
    for idx in AAS_INDICES:
        print(f'  [{idx:2d}] {COMPONENT_NAMES[idx]:<20} {C_NOMINAL[idx]:.4f}')
    print()

    nh4 = args.nh4
    if nh4 is not None:
        print(f'NH4 feed override: {nh4} mmol/L')

    if args.scale is not None:
        print(f'Testing scale={args.scale:.2f}x ...')
        results = run_reactors(args.scale, rates_growth, rates_prod,
                               reactor_ids, pm_dict, doe_dict, nh4_feed=nh4)
        ok = check(results)
        print('  All non-negative.' if ok else '  Negatives found.')
        return

    # Always show current (scale=1) first
    print('--- Current values (scale=1.0) ---')
    results = run_reactors(1.0, rates_growth, rates_prod,
                           reactor_ids, pm_dict, doe_dict, nh4_feed=nh4)
    check(results, verbose=True)

    if not args.sweep:
        print('\nRun with --sweep to find minimum safe scale factor.')
        print('\nDiagnostic: cell density and first-failure day for scale=1.0:')
        for (reactor, aa_level), traj in sorted(results.items()):
            cd_max = traj[:, 0].max()
            neg_mask = traj < 0
            if neg_mask.any():
                first_day = int(np.argmax(neg_mask.any(axis=1)))
                first_comp = int(np.argmax(neg_mask[first_day]))
                print(f'  {reactor}  AA_doe={aa_level:+d}  '
                      f'CD_max={cd_max:.3f} E9/L  '
                      f'first_neg: day={first_day} {COMPONENT_NAMES[first_comp]}={traj[first_day, first_comp]:.4f}')
        return

    # Sweep uniform scale factors
    print('\n--- Sweeping AA scale factor (applied to initial + feed) ---')
    candidates = [1, 2, 5, 10, 20, 50, 100, 200]
    for scale in candidates:
        results = run_reactors(scale, rates_growth, rates_prod,
                               reactor_ids, pm_dict, doe_dict, nh4_feed=nh4)
        failing = set()
        for (reactor, aa_level), traj in results.items():
            if (traj < 0).any():
                failing.add(reactor)
        n_fail = len(failing)
        status = 'OK' if n_fail == 0 else f'{n_fail} reactors fail'
        fail_str = f'  -> {sorted(failing)}' if 0 < n_fail <= 5 else ''
        print(f'  scale={scale:>4}x  [{status}]{fail_str}')
        if n_fail == 0:
            print(f'\nMinimum safe scale factor: {scale}x')
            print('Scaled AA concentrations (mmol/L):')
            for idx in AAS_INDICES:
                print(f'  {COMPONENT_NAMES[idx]:<20} {C_NOMINAL[idx] * scale:.4f}')
            return

    print('\nNo safe scale factor found within tested range.')
    print('\nDiagnostic at scale=200: cell density and first-failure day for failing reactors:')
    results_200 = run_reactors(200, rates_growth, rates_prod,
                               reactor_ids, pm_dict, doe_dict, nh4_feed=nh4)
    for (reactor, aa_level), traj in sorted(results_200.items()):
        neg_mask = traj < 0
        if not neg_mask.any():
            continue
        cd_max = traj[:, 0].max()
        first_day = int(np.argmax(neg_mask.any(axis=1)))
        first_comp = int(np.argmax(neg_mask[first_day]))
        print(f'  {reactor}  AA_doe={aa_level:+d}  '
              f'CD_max={cd_max:.3f} E9/L  '
              f'first_neg: day={first_day} {COMPONENT_NAMES[first_comp]}={traj[first_day, first_comp]:.4f}')


if __name__ == '__main__':
    main()
