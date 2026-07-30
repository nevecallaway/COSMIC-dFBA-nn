#!/usr/bin/env python3
"""
Consistency check: compare our data_3 metabolic rates against an independent
CHO study (Antonakoudis & Richelle, npj Syst Biol Appl 2026; MetRaC Bayesian
rate estimates for 12 fed-batch CHO-DG44 cultures).

This is NOT a predictive validation. The external study uses a different clone,
different medium, and fed-batch (not perfusion) with temperature/pH shifts at
day 7. What we can check is whether our rate magnitudes are consistent with an
independent CHO dataset on the overlapping species. Agreement = external
grounding that data_3's rates are physically plausible for CHO.

Units: the external rates are reported per 1e6 cells/mL per hour. Inoculation
is 0.3e6 cells/mL = 0.3e9 cells/L, so the per-1e6-cells/mL basis is numerically
identical to our per-1e9-cells/L basis. Only the time unit differs, so we
convert external rates by x24 (hours -> days) and treat them as per-E9-per-day.

Windowing: external rates are time-resolved. Conditions shift at day 7, so we
compare external time < 7 against our GROWTH-phase rates and external time >= 7
against our PRODUCTION-phase rates.

Caveats printed in the report:
  - vcd is the NET growth rate (death/lysis included); late negatives expected.
  - titer/product rate is molar for IgG and not unit-comparable to data_3 (mg).

Usage:
    python compare_external_rates.py
    python compare_external_rates.py --external data/rate_data_mean.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from generate_synthetic_ode import load_rates

H_TO_DAY = 24.0          # external hours -> days
DAY7     = 7.0           # temperature/pH shift; growth vs production split

# External column abbreviation -> our data_3 component index.
# Overlapping species only; the 7 extras (ac, cit, cys, cyst, form, fum, succ)
# have no counterpart in our 25-component model and are reported as unmatched.
EXT_TO_IDX = {
    'vcd': 0, 'titer': 5, 'glc': 2, 'lac': 3, 'nh4': 4,
    'gln': 6, 'glu': 7, 'asn': 8, 'asp': 9, 'ser': 10, 'gly': 11,
    'ala': 12, 'pro': 13, 'thr': 14, 'his': 15, 'lys': 16, 'val': 17,
    'met': 18, 'arg': 19, 'tyr': 20, 'ile': 21, 'leu': 22, 'phe': 23,
    'trp': 24,
}
EXTRA_EXT = ['ac', 'cit', 'cys', 'cyst', 'form', 'fum', 'succ']

COMPONENT_NAMES = [
    'CellDensity', 'CellVolume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'Asparagine', 'AsparticAcid', 'Serine', 'Glycine',
    'Alanine', 'Proline', 'Threonine', 'Histidine', 'Lysine', 'Valine',
    'Methionine', 'Arginine', 'Tyrosine', 'Isoleucine', 'Leucine',
    'Phenylalanine', 'Tryptophan',
]

# Not unit-comparable / needs a caveat note in the report
NONCOMPARABLE = {5}   # Titer (molar IgG vs mg)


def ranges_overlap(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and b_lo <= a_hi


def our_stats(rates_dict):
    """Per-component min/max/mean across the 10 reactors."""
    mat = np.array([rates_dict[r] for r in rates_dict])  # (10, 25)
    return mat.min(axis=0), mat.max(axis=0), mat.mean(axis=0)


def ext_window_stats(df, species, mask):
    """x24-converted p5/p95/mean of one external species over a time mask."""
    vals = df.loc[mask, species].to_numpy(dtype=float) * H_TO_DAY
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return np.percentile(vals, 5), np.percentile(vals, 95), vals.mean()


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(here / 'data'))
    parser.add_argument('--external', default=str(here / 'data' / 'rate_data_mean.csv'))
    args = parser.parse_args()

    data_dir = Path(args.data)
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    g_min, g_max, g_mean = our_stats(rates_growth)
    p_min, p_max, p_mean = our_stats(rates_prod)

    ext = pd.read_csv(args.external)
    ext.columns = [c.strip().lstrip('﻿') for c in ext.columns]
    t = pd.to_numeric(ext['time'], errors='coerce')
    early = t < DAY7          # growth-phase analog
    late  = t >= DAY7         # production-phase analog

    print('External: Antonakoudis & Richelle 2026, MetRaC rates, 12 CHO-DG44 '
          'fed-batch cultures')
    print('Converted x24 (h->day), treated per-E9 (see header). Consistency '
          'check only, not predictive.\n')

    header = (f'{"Species":<14} | {"our growth":>18} {"ext<d7":>18} {"ok":>3} | '
              f'{"our prod":>18} {"ext>=d7":>18} {"ok":>3}')
    print(header)
    print('-' * len(header))

    n_ok_growth = n_ok_prod = n_checked = 0
    for sp, idx in EXT_TO_IDX.items():
        if sp not in ext.columns:
            continue
        og = f'[{g_min[idx]:+.3f},{g_max[idx]:+.3f}]'
        op = f'[{p_min[idx]:+.3f},{p_max[idx]:+.3f}]'

        e_glo, e_ghi, _ = ext_window_stats(ext, sp, early)
        e_plo, e_phi, _ = ext_window_stats(ext, sp, late)
        eg = f'[{e_glo:+.3f},{e_ghi:+.3f}]'
        ep = f'[{e_plo:+.3f},{e_phi:+.3f}]'

        if idx in NONCOMPARABLE:
            gk = pk = ' . '   # skip overlap test for non-comparable units
        else:
            n_checked += 1
            g_ok = ranges_overlap(g_min[idx], g_max[idx], e_glo, e_ghi)
            p_ok = ranges_overlap(p_min[idx], p_max[idx], e_plo, e_phi)
            n_ok_growth += g_ok
            n_ok_prod   += p_ok
            gk = ' Y ' if g_ok else ' n '
            pk = ' Y ' if p_ok else ' n '

        print(f'{COMPONENT_NAMES[idx]:<14} | {og:>18} {eg:>18} {gk} | '
              f'{op:>18} {ep:>18} {pk}')

    print('-' * len(header))
    print(f'Overlap (excl. non-comparable): growth {n_ok_growth}/{n_checked}, '
          f'production {n_ok_prod}/{n_checked}')
    print(f'\nExternal-only species (no model counterpart): {", ".join(EXTRA_EXT)}')
    print('Caveats: vcd is NET growth (death/lysis -> late negatives expected); '
          'Titer excluded from overlap test (molar IgG vs mg).')


if __name__ == '__main__':
    main()
