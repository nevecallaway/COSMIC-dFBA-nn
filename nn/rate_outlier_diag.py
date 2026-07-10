#!/usr/bin/env python3
"""
Diagnostic: is LORO failure driven by metabolic outliers?

For each real reactor, measure how far its rates sit from the other 9 (a
leave-one-out z-scored distance in the combined growth+production rate space).
If that distance tracks the held-out LORO titer error, the failures are a
data-coverage problem (the held-out reactor is an outlier the other 9 cannot
span), which points to broadening the real rate diversity rather than changing
the model.

No training; runs in seconds. Compare the printed distances to the LORO errors.

Usage:
    python rate_outlier_diag.py
"""

import numpy as np
from pathlib import Path

from generate_synthetic_ode import load_rates

# LORO held-out titer errors (broad sampling run) keyed by reactor INDEX.
# Edit if you rerun; used only to print side by side.
LORO_ERR = {0: 71.6, 1: 132.1, 2: 6.7, 3: 229.0, 4: 32.3}


def main():
    here = Path(__file__).parent
    rg, rp, reactor_ids = load_rates(here / 'data' / 'data_3.csv')

    # Each reactor -> concatenated growth+production rate vector.
    X = np.array([np.concatenate([rg[r], rp[r]]) for r in reactor_ids])  # (10, 50)

    # Z-score each rate dimension across reactors (guard zero-variance dims).
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    print(f'{"idx":>3} {"reactor":<8} {"LOO dist":>9} {"LORO err":>9}')
    print('-' * 34)
    dists = []
    for i, r in enumerate(reactor_ids):
        others = np.delete(Z, i, axis=0).mean(0)       # centroid of the other 9
        dist   = np.linalg.norm(Z[i] - others)          # distance from that centroid
        dists.append(dist)
        err = LORO_ERR.get(i)
        err_s = f'{err:>8.1f}%' if err is not None else f'{"-":>9}'
        print(f'{i:>3} {r:<8} {dist:>9.2f} {err_s}')

    dists = np.array(dists)
    # Rank correlation between outlierness and LORO error, where both known.
    idx = [i for i in LORO_ERR if i < len(reactor_ids)]
    if len(idx) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr([dists[i] for i in idx], [LORO_ERR[i] for i in idx])
        print(f'\nSpearman(outlier distance, LORO error) over {len(idx)} folds: '
              f'rho={rho:.2f} (p={p:.2f})')
        print('High positive rho -> outliers drive the failures (data-coverage '
              'problem).')


if __name__ == '__main__':
    main()
