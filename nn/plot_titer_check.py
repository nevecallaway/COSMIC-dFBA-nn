#!/usr/bin/env python3
"""
Titer shape check: does our perfusion titer model (eta=1 washout) match the real
titer trajectory shape?

With eta=1 (continuous washout) titer sits near a quasi-steady v_titer*X/F, so it
tends to peak and decline as X changes. If the antibody is actually RETAINED in
perfusion (eta~0), the real titer accumulates monotonically. A mismatch here
(synthetic peaks/declines while real keeps rising) means the washout model is
wrong and is what makes titer over-sensitive to v_titer.

Regenerates the 10 real reactors (cheap), normalizes titer to its own max, and
prints the shape (peak day, whether it declines) next to the data_2 measurements.
Also saves titer_check.png if matplotlib is available.

Usage:
    python plot_titer_check.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from generate_synthetic_ode import (
    load_rates, load_phase_fractions, load_doe, generate_reactor,
    N_COMPONENTS, T_EVAL, IDX_TIT,
)

COMPONENT_NAMES = [
    'CellDensity', 'CellVolume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'Asparagine', 'AsparticAcid', 'Serine', 'Glycine',
    'Alanine', 'Proline', 'Threonine', 'Histidine', 'Lysine', 'Valine',
    'Methionine', 'Arginine', 'Tyrosine', 'Isoleucine', 'Leucine',
    'Phenylalanine', 'Tryptophan',
]


def shape_desc(days, vals):
    """Peak day and whether it declines after the peak."""
    vals = np.asarray(vals, dtype=float)
    if vals.max() <= 0:
        return 'flat/zero'
    peak = int(days[np.argmax(vals)])
    end_frac = vals[-1] / vals.max()
    declines = end_frac < 0.9
    tag = f'peak@d{peak}'
    if declines:
        tag += f', declines to {end_frac:.0%} of peak'
    else:
        tag += ', monotonic-ish (ends near peak)'
    return tag


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', type=int, default=IDX_TIT,
                        help=f'Component index to check (default {IDX_TIT}=Titer; 0=CellDensity)')
    args = parser.parse_args()
    comp = args.component
    cname = COMPONENT_NAMES[comp]

    data_dir = here / 'data'
    rg, rp, reactor_ids = load_rates(data_dir / 'data_3.csv')
    pm_dict  = load_phase_fractions(data_dir / 'data_2.csv')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    # Real measured titer (data_2, already normalized), component C5.
    df2 = pd.read_csv(data_dir / 'data_2.csv', skiprows=1)
    df2.columns = ['Vessel', 'Time', 'Phase'] + [f'C{i}' for i in range(N_COMPONENTS)]
    df2['Time'] = pd.to_numeric(df2['Time'], errors='coerce')
    df2 = df2.dropna(subset=['Time'])

    print(f'{cname} shape: SYNTHETIC (our ODE) vs REAL (data_2)\n')
    print(f'  {"reactor":<8} {"synthetic shape":<40} {"real shape":<40}')
    print('  ' + '-' * 88)

    syn_all, real_all = [], []
    for r in reactor_ids:
        doe = doe_dict.get(r, {'O2': 0, 'AAs': 0, 'Glc': 0})
        traj, _ = generate_reactor(r, rg[r], rp[r], pm_dict[r], doe)
        syn_v    = traj[:, comp]
        syn_days = T_EVAL

        rdf = df2[df2['Vessel'] == r].sort_values('Time')
        real_v    = rdf[f'C{comp}'].to_numpy(dtype=float)
        real_days = rdf['Time'].to_numpy(dtype=float)

        syn_all.append(shape_desc(syn_days, syn_v))
        real_all.append(shape_desc(real_days, real_v))
        print(f'  {r:<8} {syn_all[-1]:<40} {real_all[-1]:<40}')

    n_syn_decline  = sum('declines' in s for s in syn_all)
    n_real_decline = sum('declines' in s for s in real_all)
    print(f'\n  Synthetic {cname} declines after peak: {n_syn_decline}/{len(reactor_ids)}')
    print(f'  Real {cname} declines after peak:      {n_real_decline}/{len(reactor_ids)}')
    print('\n  If real declines but synthetic does not, our ODE misses the '
          'post-day-7 dynamics (harvest washout and/or cell-density crash).')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        axes = axes.flatten()
        for i, r in enumerate(reactor_ids):
            doe = doe_dict.get(r, {'O2': 0, 'AAs': 0, 'Glc': 0})
            traj, _ = generate_reactor(r, rg[r], rp[r], pm_dict[r], doe)
            syn = traj[:, comp]
            syn = syn / syn.max() if syn.max() > 0 else syn
            rdf = df2[df2['Vessel'] == r].sort_values('Time')
            ax = axes[i]
            ax.plot(T_EVAL, syn, 'r-', lw=1.8, label='synthetic')
            ax.plot(rdf['Time'], rdf[f'C{comp}'], 'k--', lw=1.8, label='real (data_2)')
            ax.set_title(r, fontsize=9)
            ax.set_ylim(-0.05, 1.15)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f'{cname} shape: synthetic vs real (data_2)', y=1.02)
        fig.tight_layout()
        out = here / f'shape_check_{cname}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\n  Saved {out}')
    except ImportError:
        print('  (matplotlib unavailable; text output only)')


if __name__ == '__main__':
    main()
