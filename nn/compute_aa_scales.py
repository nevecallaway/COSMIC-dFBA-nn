#!/usr/bin/env python3
"""
Compute per-amino-acid enrichment scales, replacing the uniform 210x.

Each amino acid's ODE is independent given the cell-density trajectory, and the
concentration is affine in the enrichment scale s:

    C_i(t; s) = s * P_i(t) + Q_i(t)

where P_i is the media part (initial + feed, both scaled by s) and Q_i is the
(negative) consumption part (driven by v * X, unscaled). For non-negativity
C_i(t) >= FLOOR we need

    s >= (FLOOR - Q_i(t)) / P_i(t)   for all t,

so the minimum per-AA scale is the max of that ratio over all reactors and time,
evaluated at the worst DoE for feed (AAs = -1, lowest feed). A margin adds
headroom. P_i is the exact media relaxation P_i(t) = c0 e^{-Ft} + cin0 (1-e^{-Ft});
Q_i is recovered as the generator's trajectory minus P_i.

Saves data/aa_scales.npy (length len(AAS_INDICES)), loaded by
generate_synthetic_ode.py. Re-run whenever data_3 changes.

Usage:
    python compute_aa_scales.py
"""

import numpy as np
from pathlib import Path

import generate_synthetic_ode as g
from generate_synthetic_ode import (
    load_rates, load_phase_fractions, load_doe, generate_reactor,
    AAS_INDICES, DMEM_AA, F, T_EVAL,
)

FLOOR     = 0.05     # mmol/L: keep every AA above this
MARGIN    = 1.15     # 15% headroom above the computed minimum
WORST_AAS = -1.0     # lowest feed multiplier (2^-1) -> largest required scale

COMPONENT_NAMES = [
    'CellDensity', 'CellVolume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'L-Asparagine', 'L-AsparticAcid', 'L-Serine',
    'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine', 'L-Histidine',
    'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine', 'L-Tyrosine',
    'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine', 'L-Tryptophan',
]


def main():
    here = Path(__file__).parent
    data_dir = here / 'data'
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict  = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    aas = list(AAS_INDICES)
    t   = T_EVAL
    feed_mult = 2.0 ** WORST_AAS           # feed at AAs = -1

    # Run each reactor once at DMEM (scale 1), AAs = -1, to recover Q_i.
    orig_C, orig_CIN = g.C_NOMINAL.copy(), g.CIN_NOMINAL.copy()
    g.C_NOMINAL[AAS_INDICES]   = DMEM_AA   # initial = DMEM
    g.CIN_NOMINAL[AAS_INDICES] = DMEM_AA   # feed base = DMEM (make_cin applies 2^AAs)

    req = np.zeros(len(aas))
    for r in reactor_ids:
        doe = {**doe_dict.get(r, {'O2': 0, 'AAs': 0, 'Glc': 0}), 'AAs': WORST_AAS}
        traj, _ = generate_reactor(r, rates_growth[r], rates_prod[r], pm_dict[r], doe)
        for k, j in enumerate(aas):
            c0   = DMEM_AA[k]                 # DMEM initial for this AA
            cin0 = c0 * feed_mult            # feed at AAs = -1, scale 1
            P    = c0 * np.exp(-F * t) + cin0 * (1.0 - np.exp(-F * t))
            Q    = traj[:, j] - P
            ratio = (FLOOR - Q) / P          # required s at each day
            req[k] = max(req[k], ratio.max())

    g.C_NOMINAL[:], g.CIN_NOMINAL[:] = orig_C, orig_CIN

    scales = np.maximum(1.0, req * MARGIN)
    np.save(data_dir / 'aa_scales.npy', scales)

    print(f'Per-AA enrichment scales (FLOOR={FLOOR}, margin={MARGIN:.0%}):\n')
    print(f'  {"Amino acid":<16} {"DMEM":>8} {"scale":>8} {"enriched":>10}')
    print('  ' + '-' * 44)
    for k, j in enumerate(aas):
        print(f'  {COMPONENT_NAMES[j]:<16} {DMEM_AA[k]:>8.3f} '
              f'{scales[k]:>8.1f} {DMEM_AA[k] * scales[k]:>10.3f}')
    print(f'\n  Saved: {data_dir / "aa_scales.npy"}')
    print('  Regenerate synthetic_ode.npz to use these; the old uniform 210x '
          'is replaced.')


if __name__ == '__main__':
    main()
