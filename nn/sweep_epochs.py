#!/usr/bin/env python3
"""
Training-length sweep for the real-reactor LORO, in one figure.

Runs loro_real.py at several epoch budgets, parses the MEAN metric line from each,
and plots shape correlation, error and peak ratio against training length. The
stopping point then picks itself: it is wherever rho peaks and the peak ratio
crosses 1.0, rather than being argued from the loss curves.

Peak ratio doubles as a training-progress readout here. An undertrained model
undershoots (peak < 1) because the flux head has not yet grown into the right
production magnitude, so the curve approaches 1.0 from below.

Usage:
    python sweep_epochs.py                                  # default budgets
    python sweep_epochs.py --epochs 50 100 300 1000 --batch 8
    python sweep_epochs.py --extra --rollout-train          # pass flags through
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

MEAN_RE = re.compile(
    r'MEAN\s*\|\s*([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
    r'(\d+)\s*over\s*/\s*(\d+)\s*under\s*/\s*(\d+)\s*on-target')


def run_one(py, script, epochs, base_args, extra, seed=0):
    cmd = ([str(py), str(script), '--epochs', str(epochs), '--seed', str(seed)]
           + base_args + extra)
    print(f'>> epochs={epochs} seed={seed}', flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-2000:]); print(out.stderr[-2000:])
        raise SystemExit(f'run failed at epochs={epochs}')
    m = MEAN_RE.search(out.stdout)
    if not m:
        print(out.stdout[-2000:])
        raise SystemExit(f'could not parse MEAN line at epochs={epochs}')
    rho, mae, peak, over, under, ontgt = m.groups()
    return dict(epochs=epochs, seed=seed, rho=float(rho), mae=float(mae), peak=float(peak),
                over=int(over), under=int(under), on_target=int(ontgt))


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, nargs='+',
                    default=[50, 100, 200, 300, 600, 1000, 1500])
    ap.add_argument('--seq-len', type=int, default=6)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=3,
                    help='training seeds per setting. With ~7 training reactors the '
                         'run-to-run spread is large, so a single run cannot separate '
                         'two settings; the plot shows mean with min/max band')
    ap.add_argument('--out', default='epoch_sweep')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='everything after --extra is passed through to loro_real.py')
    args = ap.parse_args()

    base = ['--seq-len', str(args.seq_len), '--batch', str(args.batch)]
    raw = [run_one(sys.executable, here / 'loro_real.py', e, base, args.extra, s)
           for e in args.epochs for s in range(args.seeds)]

    csv_path = f'{args.out}.csv'
    with open(csv_path, 'w') as fh:
        fh.write('epochs,seed,rho,mae,peak,over,under,on_target\n')
        for r in raw:
            fh.write(f'{r["epochs"]},{r["seed"]},{r["rho"]:.4f},{r["mae"]:.4f},'
                     f'{r["peak"]:.4f},{r["over"]},{r["under"]},{r["on_target"]}\n')

    # aggregate across seeds
    def agg(e, key):
        vals = [r[key] for r in raw if r['epochs'] == e]
        return sum(vals) / len(vals), min(vals), max(vals)

    rows = []
    for e in args.epochs:
        rows.append(dict(epochs=e,
                         rho=agg(e, 'rho'), mae=agg(e, 'mae'), peak=agg(e, 'peak')))

    print(f'\n{"epochs":>7} | {"rho mean [min,max]":>24} | {"MAE":>22} | {"peak":>22}')
    print('-' * 82)
    for r in rows:
        f = lambda t: f'{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]'
        print(f'{r["epochs"]:>7} | {f(r["rho"]):>24} | {f(r["mae"]):>22} | '
              f'{f(r["peak"]):>22}')
    spread = max(r['rho'][2] - r['rho'][1] for r in rows)
    print(f'\nLargest within-setting rho spread across seeds: {spread:.2f}')
    print('If that spread is comparable to the differences between settings, the '
          'settings are not distinguishable and more seeds are needed.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep = [r['epochs'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    specs = [('rho', 'tab:blue', 'shape correlation (rho)', 'Shape: higher is better'),
             ('mae', 'tab:red', 'peak-normalized MAE', 'Error: lower is better'),
             ('peak', 'tab:green', 'peak ratio (pred / real)',
              'Magnitude: 1.0 is correct\n(<1 undershoot, >1 overshoot)')]
    for ax, (key, col, ylab, title) in zip(axes, specs):
        mean = [r[key][0] for r in rows]
        lo   = [r[key][1] for r in rows]
        hi   = [r[key][2] for r in rows]
        ax.fill_between(ep, lo, hi, color=col, alpha=0.2, lw=0)   # seed spread
        ax.plot(ep, mean, 'o-', color=col)
        ax.set_ylabel(ylab); ax.set_title(title)
    axes[2].axhline(1.0, color='gray', ls='--', lw=1)

    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('training epochs (log scale)')
        ax.grid(alpha=0.3)

    best = max(rows, key=lambda r: r['rho'][0])
    fig.suptitle(f'Real-reactor LORO vs training length (batch {args.batch}, '
                 f'seq_len {args.seq_len}, {args.seeds} seeds; band = min/max across '
                 f'seeds).  Best mean rho at {best["epochs"]} epochs '
                 f'({best["rho"][0]:.2f})', y=1.06)
    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved {args.out}.png and {csv_path}')


if __name__ == '__main__':
    main()
