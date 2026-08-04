#!/usr/bin/env python3
"""
Work out the fed-batch feeding geometry from the Golzarijalal dataset metadata.

We need the per-feed bolus volume and the initial volume to write the fed-batch
ODE (a feed both ADDS nutrient mass and DILUTES everything already in the vessel).
The tables give the feed composition and which days are fed, but not the volume.

It is recoverable without touching the 13 GB archive, because each run records
its final volume and each strategy has a known number of feed days:

    V_final = V0 + n_feeds * v_bolus

so regressing final volume on the feed count gives v_bolus (slope) and V0
(intercept). A tight fit means constant boluses; a poor fit means the volume
varies with something else (checked per base medium and vs. feed media id).

Usage:
    python diagnose_fedbatch_feed.py                 # uses data/cho_fedbatch.zip
    FEDBATCH_ZIP=/path/to/cho_fedbatch.zip python diagnose_fedbatch_feed.py
"""
import os
import zipfile

import numpy as np
import pandas as pd

ZIP  = os.environ.get('FEDBATCH_ZIP', 'data/cho_fedbatch.zip')
BASE = 'in_silico_fed_batch_CHOK1_cell_culture/'


def main():
    zf = zipfile.ZipFile(ZIP)
    info  = pd.read_csv(zf.open(BASE + 'in_silico_runs_information.csv'))
    strat = pd.read_csv(zf.open(BASE + 'feed_strategy.csv'))

    day_cols = [c for c in strat.columns if c.startswith('day_')]
    strat['n_feeds'] = (strat[day_cols] == 'F').sum(axis=1)
    print(f'Feed days per strategy: min={strat.n_feeds.min()}  '
          f'max={strat.n_feeds.max()}  mean={strat.n_feeds.mean():.2f}')

    df = info.merge(strat[['feed_strategy_id', 'n_feeds']], on='feed_strategy_id')
    vol = 'final_day_volume(L)'

    def fit(sub, label):
        if sub['n_feeds'].nunique() < 2:
            print(f'{label:12s} not enough variation to fit'); return
        slope, intercept = np.polyfit(sub['n_feeds'], sub[vol], 1)
        pred = slope * sub['n_feeds'] + intercept
        ss_res = ((sub[vol] - pred) ** 2).sum()
        ss_tot = ((sub[vol] - sub[vol].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        resid = (sub[vol] - pred).abs()
        print(f'{label:12s} n={len(sub):7d}  V0={intercept:.4f} L  '
              f'v_bolus={slope:.5f} L  R2={r2:.4f}  max|resid|={resid.max():.5f} L')

    print('\n--- V_final = V0 + n_feeds * v_bolus ---')
    fit(df, 'ALL')
    for bm in sorted(df['base_media_version'].unique()):
        fit(df[df.base_media_version == bm], f'base {bm}')

    # Does volume depend on the feed MEDIA (i.e. is the bolus size media-specific)?
    print('\n--- residual volume spread within a fixed feed count ---')
    for n in sorted(df['n_feeds'].unique())[:5]:
        sub = df[df.n_feeds == n]
        print(f'  n_feeds={n:2d}  n={len(sub):6d}  volume min={sub[vol].min():.4f} '
              f'max={sub[vol].max():.4f}  std={sub[vol].std():.5f}')

    # If the bolus is constant, volume is fully determined by the feed count.
    print('\n--- unique final volumes per feed count (first few) ---')
    g = df.groupby('n_feeds')[vol].nunique().head(5)
    print(g.to_string())

    # Decisive: is the bolus a property of the STRATEGY, the MEDIA, or neither?
    # If volume is (near) constant within a feed_strategy_id, the strategy encodes
    # both which days are fed and how much, so we can read the bolus straight off.
    V0 = 0.158
    df['v_bolus'] = (df[vol] - V0) / df['n_feeds']

    for key in ['feed_strategy_id', 'feed_media_concentration_id']:
        spread = df.groupby(key)[vol].agg(['nunique', 'std', 'min', 'max'])
        print(f'\n--- final volume grouped by {key} ---')
        print(f'  mean within-group std : {spread["std"].mean():.6f} L')
        print(f'  mean unique values    : {spread["nunique"].mean():.2f}')
        print(f'  groups with 1 value   : {(spread["nunique"] == 1).sum()} / {len(spread)}')

    print('\n--- implied per-run bolus, v = (V_final - V0)/n_feeds ---')
    print(f'  V0 assumed {V0} L')
    print(f'  v_bolus  min={df.v_bolus.min():.5f}  max={df.v_bolus.max():.5f}  '
          f'mean={df.v_bolus.mean():.5f}  std={df.v_bolus.std():.5f}')
    bs = df.groupby('feed_strategy_id')['v_bolus'].std()
    print(f'  within-strategy std of v_bolus: mean={bs.mean():.6f}  max={bs.max():.6f}')
    print('  (near-zero -> bolus is fixed per strategy: read it off directly)')


if __name__ == '__main__':
    main()
