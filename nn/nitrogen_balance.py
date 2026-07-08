#!/usr/bin/env python3
"""
Test whether NH4 secretion obeys a nitrogen balance against amino acid
consumption, using the per-cell rates in data_3.csv.

Rationale: the ammonium a cell secretes should come from the nitrogen it
strips off consumed amino acids (dominantly glutamine and asparagine, which
each carry 2 N). If NH4 rate is well explained by AA consumption rates, then
a nitrogen-balance term is a real, enforceable constraint for the hybrid loss.

We regress v_NH4 across the 20 (reactor, phase) samples on three increasingly
complete predictors and report R^2 for each:
    A) glutamine only
    B) glutamine + asparagine
    C) full nitrogen-weighted consumption over all amino acids

Sign convention (data_3): consumption is negative, secretion positive.
NH4 is secreted (positive) for most reactors.

Usage:
    python nitrogen_balance.py
"""

import numpy as np
from pathlib import Path

from generate_synthetic_ode import (
    load_rates,
    IDX_NH4, IDX_GLN, IDX_GLU, IDX_ASN, IDX_ASP, IDX_SER, IDX_GLY,
    IDX_ALA, IDX_PRO, IDX_THR, IDX_HIS, IDX_LYS, IDX_VAL, IDX_MET,
    IDX_ARG, IDX_TYR, IDX_ILE, IDX_LEU, IDX_PHE, IDX_TRP,
)

# Nitrogen atoms per amino acid molecule (side chain + backbone amine)
N_ATOMS = {
    IDX_GLN: 2, IDX_GLU: 1, IDX_ASN: 2, IDX_ASP: 1, IDX_SER: 1, IDX_GLY: 1,
    IDX_ALA: 1, IDX_PRO: 1, IDX_THR: 1, IDX_HIS: 3, IDX_LYS: 2, IDX_VAL: 1,
    IDX_MET: 1, IDX_ARG: 4, IDX_TYR: 1, IDX_ILE: 1, IDX_LEU: 1, IDX_PHE: 1,
    IDX_TRP: 2,
}

NAME = {
    IDX_NH4: 'NH4', IDX_GLN: 'Glutamine', IDX_ASN: 'Asparagine',
}


def fit(X, y):
    """OLS with intercept. Returns (coeffs, intercept, r2)."""
    A = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return beta[:-1], beta[-1], r2


def main():
    here = Path(__file__).parent
    rates_growth, rates_prod, reactor_ids = load_rates(here / 'data' / 'data_3.csv')

    # Build the 20-sample matrix: 10 reactors x 2 phases
    v_nh4, v_gln, v_asn, n_total = [], [], [], []
    labels = []
    for reactor in reactor_ids:
        for phase, rates in (('growth', rates_growth), ('prod', rates_prod)):
            r = rates[reactor]
            v_nh4.append(r[IDX_NH4])
            v_gln.append(r[IDX_GLN])
            v_asn.append(r[IDX_ASN])
            # total nitrogen consumed: sum of N_atoms * consumption (only where consumed)
            n_consumed = 0.0
            for idx, n in N_ATOMS.items():
                if r[idx] < 0:                 # consumed
                    n_consumed += n * (-r[idx])
            n_total.append(n_consumed)
            labels.append(f'{reactor}/{phase}')

    v_nh4  = np.array(v_nh4)
    v_gln  = np.array(v_gln)
    v_asn  = np.array(v_asn)
    n_total = np.array(n_total)

    print('Per-sample rates (mmol / E9 cells / day):')
    print(f'  {"sample":<16} {"v_NH4":>8} {"v_Gln":>8} {"v_Asn":>8} {"N_consumed":>11}')
    for i, lab in enumerate(labels):
        print(f'  {lab:<16} {v_nh4[i]:>8.3f} {v_gln[i]:>8.3f} '
              f'{v_asn[i]:>8.3f} {n_total[i]:>11.3f}')

    print('\nNitrogen balance regressions (NH4 secretion vs AA consumption):\n')

    # A) glutamine only (predictor = nitrogen released from Gln = 2 * (-v_gln))
    coef, b, r2 = fit((-v_gln)[:, None], v_nh4)
    print(f'  A) NH4 ~ (-Gln)            slope={coef[0]:+.3f}  '
          f'intercept={b:+.3f}  R^2={r2:.3f}')

    # B) glutamine + asparagine
    X = np.column_stack([-v_gln, -v_asn])
    coef, b, r2 = fit(X, v_nh4)
    print(f'  B) NH4 ~ (-Gln)+(-Asn)     slopes={coef[0]:+.3f},{coef[1]:+.3f}  '
          f'intercept={b:+.3f}  R^2={r2:.3f}')

    # C) full nitrogen-weighted consumption
    coef, b, r2 = fit(n_total[:, None], v_nh4)
    print(f'  C) NH4 ~ total N consumed  slope={coef[0]:+.3f}  '
          f'intercept={b:+.3f}  R^2={r2:.3f}')

    print('\nInterpretation:')
    print('  R^2 near 1 in B or C  -> nitrogen balance is real, worth a loss term.')
    print('  Low R^2 everywhere    -> NH4 not explained by tracked AAs; skip it.')
    print('  Slope in A/B near the N-atom count (2 for Gln/Asn) supports the')
    print('  stoichiometric interpretation directly.')


if __name__ == '__main__':
    main()
