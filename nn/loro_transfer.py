#!/usr/bin/env python3
"""
Leakage-free transfer-learning evaluation (Kimberly's protocol).

For a held-out set of real reactors, run all three stages so nothing about the
held-out reactors ever touches training:
  1. generate synthetic data with those reactors EXCLUDED from the donor pool,
  2. pretrain the decoder on that synthetic data          -> model_flux.pt,
  3. fine-tune on the NON-held-out real reactors          -> model_real.pt,
  4. evaluate each held-out reactor against the REAL data (--real-target), for
     both the synthetic-only baseline and the transfer model.

Every reported number is honest: held-out rates were never donors and held-out
real data was never in fine-tuning. Reports per-reactor titer error and titer
shape correlation (trajectory rho), baseline vs transfer.

Usage:
    python loro_transfer.py --holdout 1 3 6      # R0002, R0004, R0008 (stratified)
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


def parse(out):
    """(titer error %, titer trajectory rho) from an evaluate.py run."""
    err = re.search(r'Mean titer error:\s*([\d.]+)%', out)
    rho = re.search(r'^\s*Titer\s+([-\d.]+)\s*$', out, re.M)   # the Traj rho line
    return (float(err.group(1)) if err else float('nan'),
            float(rho.group(1)) if rho else float('nan'))


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', type=int, nargs='+', required=True,
                    help='Real reactor indices to hold out from EVERY stage')
    ap.add_argument('--n-extra', type=int, default=3000)
    ap.add_argument('--rate-mix', type=float, default=0.2)
    ap.add_argument('--rate-scale', type=float, default=0.1)
    args = ap.parse_args()

    py = sys.executable
    H  = [str(h) for h in args.holdout]
    npz  = here / 'synth_ho.npz'
    flux = here / 'model_flux.pt'
    real = here / 'model_real.pt'

    # 1. synthetic pretrain with the held-out reactors excluded from donors
    run([py, here / 'generate_synthetic_ode.py', '--holdout', *H, '--output', npz,
         '--n-extra', args.n_extra, '--rate-mix', args.rate_mix,
         '--rate-scale', args.rate_scale])
    run([py, here / 'train_sample.py', '--data', npz, '--output', flux, '--batch', 256])

    # 2. fine-tune on the non-held-out real reactors
    run([py, here / 'train_real.py', '--ode-data', npz, '--init', flux,
         '--output', real, '--holdout', *H])

    # 3. evaluate each held-out reactor: baseline (synthetic only) vs transfer
    results = {}
    for h in args.holdout:
        base = parse(run([py, here / 'evaluate.py', '--model', flux, '--data', npz,
                          '--eval-reactor', h, '--real-target', '--no-plots']))
        tran = parse(run([py, here / 'evaluate.py', '--model', real, '--data', npz,
                          '--eval-reactor', h, '--real-target', '--no-plots']))
        results[h] = (base, tran)

    print('\n=== Leakage-free transfer evaluation (held-out, real target) ===')
    print(f'{"reactor":>8} | {"base err":>9} {"transfer err":>13} | '
          f'{"base rho":>9} {"tran rho":>9}')
    print('-' * 58)
    for h in sorted(results):
        (be, br), (te, tr) = results[h]
        print(f'{h:>8} | {be:>8.1f}% {te:>12.1f}% | {br:>9.2f} {tr:>9.2f}')
    print('\nLower titer error and higher rho are better. Transfer should be at '
          'least as good as the synthetic-only baseline if fine-tuning on the real '
          'reactors helped. Read rho (shape), not just error (scale is a guess).')


if __name__ == '__main__':
    main()
