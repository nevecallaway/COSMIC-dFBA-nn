#!/usr/bin/env python3
"""
Physiological rate envelope for filtering sampled synthetic reactors.

Builds a per-component plausible band for metabolic rates from the union of:
  - data_3 (Sarat's 10 reactors, growth and production phases), and
  - an independent CHO dataset (Antonakoudis & Richelle 2026, MetRaC rates),
    converted to our per-E9-per-day basis (x24).

When generate_synthetic_ode samples reactor rates from the fitted Gaussian, it
can call `in_envelope` to reject any sample whose rates fall outside this band.
This keeps synthetic reactors physiologically plausible, grounded by two
independent datasets, without importing the external clone's absolute rates as
training data.

Design choices:
  - The band is the widest union of growth/production (data_3) and the external
    range, so it only rejects clearly out-of-range samples (permissive filter).
  - A margin (default 10%) widens the band so borderline-but-plausible samples
    are not over-rejected.
  - Titer (idx 5) takes data_3 bounds only: the external product rate is molar
    IgG and not unit-comparable. Cell volume (idx 1) has no external counterpart.

Usage:
    python rate_envelope.py                 # build and print the envelope
    # in generate_synthetic_ode.py:
    #   from rate_envelope import build_envelope, in_envelope
    #   lo, hi = build_envelope(data_dir)
    #   if not (in_envelope(v_growth, lo, hi) and in_envelope(v_prod, lo, hi)):
    #       reject
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from generate_synthetic_ode import load_rates, N_COMPONENTS

H_TO_DAY = 24.0

# External column abbreviation -> our component index (overlapping species).
EXT_TO_IDX = {
    'vcd': 0, 'glc': 2, 'lac': 3, 'nh4': 4,
    'gln': 6, 'glu': 7, 'asn': 8, 'asp': 9, 'ser': 10, 'gly': 11,
    'ala': 12, 'pro': 13, 'thr': 14, 'his': 15, 'lys': 16, 'val': 17,
    'met': 18, 'arg': 19, 'tyr': 20, 'ile': 21, 'leu': 22, 'phe': 23,
    'trp': 24,
}
# 'titer' (idx 5) deliberately excluded: molar IgG, not unit-comparable.

COMPONENT_NAMES = [
    'CellDensity', 'CellVolume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'Asparagine', 'AsparticAcid', 'Serine', 'Glycine',
    'Alanine', 'Proline', 'Threonine', 'Histidine', 'Lysine', 'Valine',
    'Methionine', 'Arginine', 'Tyrosine', 'Isoleucine', 'Leucine',
    'Phenylalanine', 'Tryptophan',
]


def build_envelope(data_dir, external_name='rate_data_mean.csv', margin=0.10):
    """
    Build per-component (lo, hi) rate bands from data_3 unioned with external.

    Args:
        data_dir: directory holding data_3.csv and the external CSV
        external_name: external rate file (per-1e6/mL/h; converted x24 here)
        margin: fractional widening of each band (0.10 = +/-10% of its width)

    Returns:
        lo, hi : np.ndarray shape (N_COMPONENTS,)  lower/upper rate bounds
    """
    data_dir = Path(data_dir)
    rates_growth, rates_prod, _ = load_rates(data_dir / 'data_3.csv')

    g = np.array([rates_growth[r] for r in rates_growth])   # (10, 25)
    p = np.array([rates_prod[r]   for r in rates_prod])      # (10, 25)
    lo = np.minimum(g.min(axis=0), p.min(axis=0))
    hi = np.maximum(g.max(axis=0), p.max(axis=0))

    ext_path = data_dir / external_name
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        ext.columns = [c.strip().lstrip('﻿') for c in ext.columns]
        for sp, idx in EXT_TO_IDX.items():
            if sp not in ext.columns:
                continue
            vals = pd.to_numeric(ext[sp], errors='coerce').to_numpy() * H_TO_DAY
            vals = vals[~np.isnan(vals)]
            if vals.size:
                lo[idx] = min(lo[idx], vals.min())
                hi[idx] = max(hi[idx], vals.max())
    else:
        print(f'  (external file {ext_path} not found; using data_3 bounds only)')

    # Widen by margin so borderline-plausible samples are not over-rejected.
    width = hi - lo
    lo = lo - margin * width
    hi = hi + margin * width
    return lo, hi


def in_envelope(v, lo, hi):
    """True if every component of rate vector v lies within [lo, hi]."""
    v = np.asarray(v)
    return bool(np.all(v >= lo) and np.all(v <= hi))


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(here / 'data'))
    parser.add_argument('--margin', type=float, default=0.10)
    args = parser.parse_args()

    lo, hi = build_envelope(args.data, margin=args.margin)

    print(f'Rate envelope (per-E9-per-day, margin={args.margin:.0%}):\n')
    print(f'  {"Component":<14} {"lo":>10} {"hi":>10}   external?')
    print('  ' + '-' * 48)
    for i in range(N_COMPONENTS):
        has_ext = i in EXT_TO_IDX.values()
        print(f'  {COMPONENT_NAMES[i]:<14} {lo[i]:>10.3f} {hi[i]:>10.3f}   '
              f'{"yes" if has_ext else "no"}')

    # Sanity: every real data_3 reactor rate must sit inside the envelope.
    rates_growth, rates_prod, reactor_ids = load_rates(Path(args.data) / 'data_3.csv')
    n_bad = 0
    for r in reactor_ids:
        if not in_envelope(rates_growth[r], lo, hi):
            n_bad += 1
        if not in_envelope(rates_prod[r], lo, hi):
            n_bad += 1
    print(f'\n  Sanity: {2 * len(reactor_ids) - n_bad}/{2 * len(reactor_ids)} '
          f'real data_3 phase-rate vectors inside the envelope '
          f'(should be all).')


if __name__ == '__main__':
    main()
