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


def run_reactors(asn_conc, ser_conc, rates_growth, rates_prod, reactor_ids,
                 pm_dict, doe_dict):
    """
    Run all reactors at each AA DoE level with the given initial concentrations.
    Returns a dict: {(reactor, doe_level): {feature_idx: min_value}}.
    """
    c_nom = C_NOMINAL.copy()
    c_nom[IDX_ASN] = asn_conc
    c_nom[IDX_SER] = ser_conc

    results = {}
    for reactor in reactor_ids:
        base_doe = doe_dict.get(reactor, {'O2': 0, 'AAs': 0, 'Glc': 0})
        pm_days  = pm_dict.get(reactor)
        if pm_days is None:
            continue
        for aa_level in AA_DOE_LEVELS:
            doe = {**base_doe, 'AAs': float(aa_level)}

            # Temporarily patch C_NOMINAL by overriding make_cin via monkey-patch
            import generate_synthetic_ode as _g
            orig = _g.C_NOMINAL.copy()
            _g.C_NOMINAL[:] = c_nom

            traj, _ = generate_reactor(reactor, rates_growth[reactor],
                                       rates_prod[reactor], pm_days, doe)
            _g.C_NOMINAL[:] = orig

            mins = {idx: traj[:, idx].min() for idx in FEATURE_NAMES}
            results[(reactor, aa_level)] = mins
    return results


def check(asn_conc, ser_conc, results):
    """Return True if no negatives found."""
    any_neg = False
    for (reactor, aa_level), mins in results.items():
        for idx, name in FEATURE_NAMES.items():
            v = mins[idx]
            if v < 0:
                print(f'  NEG  reactor={reactor}  AA_doe={aa_level:+d}  '
                      f'{name}  min={v:.6f} mmol/L')
                any_neg = True
    return not any_neg


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
        # Show the actual min values per feature across all conditions
        print('\nMinimum concentrations reached (current C_NOMINAL):')
        for idx, name in FEATURE_NAMES.items():
            all_mins = [mins[idx] for mins in results.values()]
            print(f'  {name}: min={min(all_mins):.6f}  '
                  f'mean_min={np.mean(all_mins):.6f} mmol/L')
        print('\nRun with --sweep to find minimum safe C_NOMINAL values.')
        return

    # Sweep: find minimum C_NOMINAL that avoids negatives
    print('\n--- Sweeping concentrations ---')
    candidates = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    for asn in candidates:
        for ser in candidates:
            results = run_reactors(asn, ser, rates_growth, rates_prod,
                                   reactor_ids, pm_dict, doe_dict)
            ok = check(asn, ser, results)
            status = 'OK' if ok else 'NEG'
            print(f'  [{status}]  Asparagine={asn:.2f}  Serine={ser:.2f}')
            if ok:
                print(f'\nSmallest safe values found: '
                      f'Asparagine={asn:.2f}  Serine={ser:.2f} mmol/L')
                return

    print('\nNo safe values found in sweep range. '
          'Check consumption rates in data_3.csv.')


if __name__ == '__main__':
    main()
