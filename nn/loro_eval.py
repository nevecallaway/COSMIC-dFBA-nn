#!/usr/bin/env python3
"""
Leave-one-reactor-out (LORO) generalization test.

The standard eval is optimistic: each real reactor's exact rates appear in the
training set as donors (rate_mix=0.2 -> 80% donor-copied), so predicting a real
reactor is interpolation across DoE, not generalization to unseen metabolism.

For each real reactor h, this driver:
  1. generates synthetic data with h EXCLUDED from the donor/sampling pool,
  2. trains a fresh decoder on it,
  3. evaluates on h, whose metabolism is now unseen.

The held-out titer error per fold is the honest generalization signal; compare
against the in-distribution reference (~1.3% from the standard run). A large gap
confirms the leakage effect; a small gap means the model genuinely interpolates
across the metabolic manifold of the other reactors.

Usage:
    python loro_eval.py                 # all 10 folds (slow: 10 train runs)
    python loro_eval.py --folds 0       # single fold first to gauge time
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print('>>', ' '.join(str(c) for c in cmd), flush=True)
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        sys.exit(f'command failed: {cmd}')
    return out.stdout


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(10)))
    ap.add_argument('--n-extra', type=int, default=3000)
    ap.add_argument('--rate-mix', type=float, default=0.2,
                    help='Fraction sampled vs donor-copied (1.0 = no memorizable copies)')
    ap.add_argument('--rate-scale', type=float, default=0.1,
                    help='Covariance multiplier for sampled rates (higher = more varied)')
    args = ap.parse_args()

    py = sys.executable
    results = {}
    for h in args.folds:
        npz = here / f'loro_{h}.npz'
        pt  = here / f'loro_{h}.pt'
        run([py, here / 'generate_synthetic_ode.py', '--holdout', h,
             '--output', npz, '--n-extra', args.n_extra,
             '--rate-mix', args.rate_mix, '--rate-scale', args.rate_scale])
        run([py, here / 'train_sample.py', '--data', npz, '--output', pt])
        out = run([py, here / 'evaluate.py', '--model', pt, '--data', npz,
                   '--eval-reactor', h, '--no-plots'])
        m = re.search(r'Mean titer error:\s*([\d.]+)%', out)
        results[h] = float(m.group(1)) if m else float('nan')
        print(f'  fold {h}: held-out titer error = {results[h]:.1f}%\n', flush=True)

    print('\n=== Leave-one-reactor-out summary ===')
    for h in sorted(results):
        print(f'  reactor index {h}: {results[h]:.1f}%')
    vals = [v for v in results.values() if v == v]
    if vals:
        print(f'  mean held-out titer error: {sum(vals) / len(vals):.1f}%')
        print(f'  in-distribution reference:  ~1.3%')


if __name__ == '__main__':
    main()
