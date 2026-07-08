#!/usr/bin/env python3
"""
Physiological rate envelope for filtering sampled synthetic reactors.

Builds a per-component plausible band for metabolic rates from data_3 (Sarat's
10 reactors, growth and production phases). When generate_synthetic_ode samples
reactor rates from the fitted Gaussian, it calls `in_envelope` to reject any
sample whose rates fall outside this band, keeping synthetic reactors within the
range spanned by the real data.

The band is data_3-only by design. The external CHO dataset (Antonakoudis &
Richelle) is used for validation (see compare_external_rates.py), not as a
sampling bound: its instantaneous fed-batch rates hit transient extremes that
are the wrong yardstick for bounding phase-constant sampled rates.

A margin (default 10%) widens each band so borderline-but-plausible samples are
not over-rejected.

Usage:
    python rate_envelope.py                 # build and print the envelope
    # in generate_synthetic_ode.py:
    #   from rate_envelope import build_envelope_from_rates, in_envelope
    #   lo, hi = build_envelope_from_rates(rates_growth, rates_prod)
    #   if not (in_envelope(v_growth, lo, hi) and in_envelope(v_prod, lo, hi)):
    #       reject
"""

import argparse
import numpy as np
from pathlib import Path

COMPONENT_NAMES = [
    'CellDensity', 'CellVolume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'Asparagine', 'AsparticAcid', 'Serine', 'Glycine',
    'Alanine', 'Proline', 'Threonine', 'Histidine', 'Lysine', 'Valine',
    'Methionine', 'Arginine', 'Tyrosine', 'Isoleucine', 'Leucine',
    'Phenylalanine', 'Tryptophan',
]


def build_envelope_from_rates(rates_growth, rates_prod, margin=0.10):
    """
    Build per-component (lo, hi) rate bands from data_3 growth/production rates.

    Args:
        rates_growth, rates_prod: dicts {reactor_id: rate array (N_COMPONENTS,)}
        margin: fractional widening of each band (0.10 = +/-10% of its width)

    Returns:
        lo, hi : np.ndarray  lower/upper rate bounds, one per component
    """
    g = np.array([rates_growth[r] for r in rates_growth])
    p = np.array([rates_prod[r]   for r in rates_prod])
    lo = np.minimum(g.min(axis=0), p.min(axis=0))
    hi = np.maximum(g.max(axis=0), p.max(axis=0))
    width = hi - lo
    return lo - margin * width, hi + margin * width


def build_envelope(data_dir, margin=0.10):
    """Convenience loader: read data_3 from disk, then build the envelope."""
    from generate_synthetic_ode import load_rates
    rg, rp, _ = load_rates(Path(data_dir) / 'data_3.csv')
    return build_envelope_from_rates(rg, rp, margin)


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

    from generate_synthetic_ode import load_rates
    rates_growth, rates_prod, reactor_ids = load_rates(Path(args.data) / 'data_3.csv')
    lo, hi = build_envelope_from_rates(rates_growth, rates_prod, args.margin)

    print(f'Rate envelope (data_3 bounds, per-E9-per-day, margin={args.margin:.0%}):\n')
    print(f'  {"Component":<14} {"lo":>10} {"hi":>10}')
    print('  ' + '-' * 36)
    for i, name in enumerate(COMPONENT_NAMES):
        print(f'  {name:<14} {lo[i]:>10.3f} {hi[i]:>10.3f}')

    # Sanity: every real data_3 phase-rate vector must sit inside the envelope.
    total = 2 * len(reactor_ids)
    n_in = sum(in_envelope(rates_growth[r], lo, hi) for r in reactor_ids) + \
           sum(in_envelope(rates_prod[r], lo, hi) for r in reactor_ids)
    print(f'\n  Sanity: {n_in}/{total} real data_3 phase-rate vectors inside '
          f'the envelope (should be all).')


if __name__ == '__main__':
    main()
