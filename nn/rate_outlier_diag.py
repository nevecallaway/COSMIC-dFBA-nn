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

    IDX_CD, IDX_TIT = 0, 5   # cell-density and titer component indices

    # Full 50-dim rate vector, and the titer-relevant subspace (growth+prod of
    # cell density and titer), which is what actually drives titer error.
    X = np.array([np.concatenate([rg[r], rp[r]]) for r in reactor_ids])   # (10, 50)
    tit_dims = [IDX_CD, IDX_TIT, 25 + IDX_CD, 25 + IDX_TIT]               # in the 50-vec
    Xt = X[:, tit_dims]                                                    # (10, 4)

    def loo_distances(M):
        mu, sd = M.mean(0), M.std(0)
        sd[sd == 0] = 1.0
        Z = (M - mu) / sd
        d = np.zeros(len(M))
        for i in range(len(M)):
            d[i] = np.linalg.norm(Z[i] - np.delete(Z, i, axis=0).mean(0))
        return d

    d_all = loo_distances(X)
    d_tit = loo_distances(Xt)

    print(f'{"idx":>3} {"reactor":<8} {"all dist":>9} {"titer dist":>11} '
          f'{"titer_prod":>11} {"LORO err":>9}')
    print('-' * 56)
    for i, r in enumerate(reactor_ids):
        err = LORO_ERR.get(i)
        err_s = f'{err:>8.1f}%' if err is not None else f'{"-":>9}'
        print(f'{i:>3} {r:<8} {d_all[i]:>9.2f} {d_tit[i]:>11.2f} '
              f'{rp[r][IDX_TIT]:>11.2f} {err_s}')

    from scipy.stats import spearmanr
    idx = [i for i in LORO_ERR if i < len(reactor_ids)]
    if len(idx) >= 3:
        errs = [LORO_ERR[i] for i in idx]
        rho_a, _ = spearmanr([d_all[i] for i in idx], errs)
        rho_t, _ = spearmanr([d_tit[i] for i in idx], errs)
        print(f'\nSpearman with LORO error over {len(idx)} folds:')
        print(f'  all-dim distance:            rho = {rho_a:+.2f}')
        print(f'  titer-relevant distance:     rho = {rho_t:+.2f}')
        print('Positive titer-relevant rho -> failures are extrapolation beyond '
              'the observed titer/growth range (coverage in the dims that matter).')


if __name__ == '__main__':
    main()
