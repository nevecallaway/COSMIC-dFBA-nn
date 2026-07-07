#!/usr/bin/env python3
"""
Diagnostic: find minimum C_NOMINAL for asparagine and serine such that
concentrations never go negative under nominal, high AA (+1), and low AA (-1)
DoE conditions, across all 10 real reactors.

Usage:
    python check_concentrations.py
    python check_concentrations.py --asn 0.5 --ser 1.0   # test specific values
    python check_concentrations.py --sweep                 # sweep a range
"""

import argparse
import numpy as np
from pathlib import Path

from generate_synthetic_ode import (
    load_rates, load_phase_fractions, load_doe,
    generate_reactor, make_cin,
    C_NOMINAL, IDX_ASN, IDX_SER,
)

FEATURE_NAMES = {IDX_ASN: 'Asparagine', IDX_SER: 'Serine'}
AA_DOE_LEVELS = [-1, 0, 1]


MIN_THRESHOLD = 0.01  # concentrations must stay above this (mmol/L)


def run_reactors(asn_conc, ser_conc, rates_growth, rates_prod, reactor_ids,
                 pm_dict, doe_dict):
    """
    Run all reactors at each AA DoE level with the given initial concentrations.
    Returns a dict: {(reactor, doe_level): {feature_idx: min_value}}.
    """
    import generate_synthetic_ode as _g

    c_nom = _g.C_NOMINAL.copy()
    c_nom[IDX_ASN] = asn_conc
    c_nom[IDX_SER] = ser_conc

    # Patch only C_NOMINAL (initial conditions); feed (CIN_NOMINAL) stays unchanged
    orig_nom = _g.C_NOMINAL.copy()
    _g.C_NOMINAL[:] = c_nom

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
            mins = {idx: traj[:, idx].min() for idx in FEATURE_NAMES}
            results[(reactor, aa_level)] = mins

    _g.C_NOMINAL[:] = orig_nom

    return results


def check(asn_conc, ser_conc, results, verbose=True):
    """Return True if all concentrations stay above MIN_THRESHOLD."""
    any_low = False
    for (reactor, aa_level), mins in results.items():
        for idx, name in FEATURE_NAMES.items():
            v = mins[idx]
            if v < MIN_THRESHOLD:
                if verbose:
                    print(f'  LOW  reactor={reactor}  AA_doe={aa_level:+d}  '
                          f'{name}  min={v:.6f} mmol/L')
                any_low = True
    return not any_low


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default=str(here / 'data'))
    parser.add_argument('--asn',   type=float, default=None,
                        help='Test a specific asparagine C_NOMINAL (mmol/L)')
    parser.add_argument('--ser',   type=float, default=None,
                        help='Test a specific serine C_NOMINAL (mmol/L)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep concentrations to find minimum non-negative values')
    args = parser.parse_args()

    data_dir = Path(args.data)
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict  = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    print(f'Reactors: {reactor_ids}')
    print(f'Current C_NOMINAL: Asparagine={C_NOMINAL[IDX_ASN]:.4f}  '
          f'Serine={C_NOMINAL[IDX_SER]:.4f} mmol/L\n')

    if args.asn is not None or args.ser is not None:
        asn = args.asn if args.asn is not None else C_NOMINAL[IDX_ASN]
        ser = args.ser if args.ser is not None else C_NOMINAL[IDX_SER]
        print(f'Testing: Asparagine={asn:.4f}  Serine={ser:.4f} mmol/L')
        results = run_reactors(asn, ser, rates_growth, rates_prod,
                               reactor_ids, pm_dict, doe_dict)
        ok = check(asn, ser, results)
        print('  All non-negative.' if ok else '  Negatives found.')
        return

    # Always show current values first
    print('--- Current values ---')
    results = run_reactors(C_NOMINAL[IDX_ASN], C_NOMINAL[IDX_SER],
                           rates_growth, rates_prod, reactor_ids, pm_dict, doe_dict)
    check(C_NOMINAL[IDX_ASN], C_NOMINAL[IDX_SER], results)

    if not args.sweep:
        # Show the actual min values per reactor per DoE condition
        print('\nMinimum concentrations reached (current C_NOMINAL):')
        print(f'  {"Reactor":<10} {"AA_doe":>6}  ', end='')
        for name in FEATURE_NAMES.values():
            print(f'{name:>14}', end='')
        print()
        print('  ' + '-' * (18 + 14 * len(FEATURE_NAMES)))
        for (reactor, aa_level), mins in sorted(results.items()):
            print(f'  {reactor:<10} {aa_level:>+6}  ', end='')
            for idx in FEATURE_NAMES:
                v = mins[idx]
                flag = ' *' if v < MIN_THRESHOLD else '  '
                print(f'{v:>12.4f}{flag}', end='')
            print()
        print(f'\n  (* = below threshold {MIN_THRESHOLD} mmol/L)')
        print('\nRun with --sweep to find minimum safe C_NOMINAL values.')
        return

    # Sweep: for each (asn, ser) pair report pass/fail and which reactors still fail
    print('\n--- Sweeping concentrations ---')
    candidates = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for asn in candidates:
        for ser in candidates:
            results = run_reactors(asn, ser, rates_growth, rates_prod,
                                   reactor_ids, pm_dict, doe_dict)
            failing = set()
            for (reactor, aa_level), mins in results.items():
                for idx in FEATURE_NAMES:
                    if mins[idx] < MIN_THRESHOLD:
                        failing.add(reactor)
            n_fail = len(failing)
            status = 'OK' if n_fail == 0 else f'{n_fail} reactors fail'
            fail_str = f'  -> {sorted(failing)}' if 0 < n_fail <= 5 else ''
            print(f'  Asparagine={asn:.2f}  Serine={ser:.2f}  [{status}]{fail_str}')
            if n_fail == 0:
                print(f'\nSmallest safe values: '
                      f'Asparagine={asn:.2f}  Serine={ser:.2f} mmol/L')
                return

    print('\nNo fully safe values found. Some reactors deplete regardless of concentration.')


if __name__ == '__main__':
    main()
