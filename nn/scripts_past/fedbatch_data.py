#!/usr/bin/env python3
"""
Loader for the fed-batch dataset (Golzarijalal 400k in-silico CHO runs).

Turns the compact npz from prep_fedbatch.py plus the run metadata into the
arrays the fed-batch decoder needs:

    windows    (n, seq_len, F+1)  input days + normalized day index
    targets    (n, F)             next-day concentrations
    feed_conc  (n, F)             the run's feed medium mapped onto our features
    feed_frac  (n, 1)             dilution fraction f for the predicted day
    run_idx    (n,)               which run each window came from (for holdout)

Feeding geometry (see diagnose_fedbatch_feed.py):
    V0 = 0.158 L is constant across runs; the bolus is a per-run design
    parameter recovered exactly from metadata as

        v_bolus = (final_day_volume - V0) / n_feeds

    Before the k-th feed the working volume is V = V0 + k*v_bolus, so the day's
    dilution fraction is f = v_bolus / (V + v_bolus), and 0 on unfed days.

Feed composition: only amino acids and pyruvate are fed. Cell density and titer
are never fed, and glucose is not in the feed table either (it comes from the
base medium), so those entries are 0 and glucose is purely consumed.

Usage:
    python fedbatch_data.py          # prints a summary / sanity check
    from fedbatch_data import load_fedbatch, build_fedbatch_windows
"""
import os
import zipfile

import numpy as np
import pandas as pd

V0_LITRES = 0.158
BASE      = 'in_silico_fed_batch_CHOK1_cell_culture/'

NPZ = os.environ.get('FEDBATCH_OUT', 'data/cho_fedbatch_daily.npz')
ZIP = os.environ.get('FEDBATCH_ZIP', 'data/cho_fedbatch.zip')

# our feature name -> column in feed_media_concentration.csv (None = not fed)
FEED_COL = {
    'CellDensity': None,
    'Titer':       None,
    'Glucose':     None,                    # not in the feed table (base medium only)
    'Asparagine':  'asparagine(mmol/L)',
    'Serine':      'serine(mmol/L)',
    'Glycine':     'glycine(mmol/L)',
}


def load_fedbatch(npz_path=NPZ, zip_path=ZIP, v0=V0_LITRES):
    """
    Returns dict with:
        traj       (R, D, F) daily trajectories
        features   list[str] feature names (index 0 is cell density)
        feed_conc  (R, F)    per-run feed composition on our features
        feed_frac  (R, D)    per-run per-day dilution fraction (0 on unfed days)
        v_bolus    (R,)      per-run bolus volume in L
        n_feeds    (R,)      number of feed days
    """
    d = np.load(npz_path, allow_pickle=True)
    traj     = d['trajectories'].astype(np.float32)          # (R, D, F)
    features = [str(x) for x in d['feature_names']]
    run_ids  = d['bioreactor_id']
    R, D, F  = traj.shape

    fm = pd.DataFrame(d['feed_media'],    columns=[str(c) for c in d['feed_media_cols']])
    fs = pd.DataFrame(d['feed_strategy'], columns=[str(c) for c in d['feed_strategy_cols']])
    # The published tables contain duplicated ids (runs_information has 400,569
    # rows, not exactly 1000*100*4), and .loc on a duplicated index returns extra
    # rows. Keep the first occurrence of each id.
    fm = fm.set_index('feed_media_concentration_id')
    fm = fm[~fm.index.duplicated(keep='first')]
    fs = fs.set_index('feed_strategy_id')
    fs = fs[~fs.index.duplicated(keep='first')]
    day_cols = [c for c in fs.columns if c.startswith('day_')]

    # Per-run metadata comes from runs_information keyed on bioreactor_id, NOT
    # from the npz's id arrays: those were written with a .loc on a duplicated
    # index and can be longer than the trajectory count.
    zf = zipfile.ZipFile(zip_path)
    info = pd.read_csv(zf.open(BASE + 'in_silico_runs_information.csv'))
    info = info.set_index('bioreactor_id')
    info = info[~info.index.duplicated(keep='first')]
    sub = info.reindex(run_ids)
    feed_media_id    = sub['feed_media_concentration_id'].to_numpy()
    feed_strategy_id = sub['feed_strategy_id'].to_numpy()
    v_final          = sub['final_day_volume(L)'].to_numpy(np.float64)
    if np.isnan(v_final).any():
        raise ValueError('runs_information is missing some sampled bioreactor_ids')

    # --- per-run feed composition on our features ---
    feed_conc = np.zeros((R, F), np.float32)
    for j, name in enumerate(features):
        col = FEED_COL.get(name)
        if col is None:
            continue
        feed_conc[:, j] = fm.reindex(feed_media_id)[col].to_numpy(np.float32)

    # --- per-run feed day mask, aligned to trajectory day index ---
    # feed_strategy day_1..day_13 map onto trajectory days 1..13.
    fed = np.zeros((R, D), bool)
    strat_mask = (fs[day_cols] == 'F').to_numpy()            # (100, 13)
    rows = fs.index.get_indexer(feed_strategy_id)
    n_day_cols = min(len(day_cols), D - 1)
    fed[:, 1:1 + n_day_cols] = strat_mask[rows][:, :n_day_cols]

    # --- per-run bolus from the recorded final volume ---
    if np.isnan(feed_conc).any():
        raise ValueError('unmatched feed_media_concentration_id for some runs')

    n_feeds = fed.sum(axis=1).astype(np.float64)
    v_bolus = np.where(n_feeds > 0, (v_final - v0) / np.maximum(n_feeds, 1), 0.0)
    v_bolus = np.clip(v_bolus, 0.0, None)                    # volume rounding can go slightly negative

    # --- dilution fraction per day: V grows by one bolus per preceding feed ---
    feeds_before = np.cumsum(fed, axis=1) - fed              # feeds strictly before this day
    V_before = v0 + feeds_before * v_bolus[:, None]
    denom = V_before + v_bolus[:, None]
    feed_frac = np.where(fed, v_bolus[:, None] / np.maximum(denom, 1e-12), 0.0)

    return dict(traj=traj, features=features, feed_conc=feed_conc,
                feed_frac=feed_frac.astype(np.float32), v_bolus=v_bolus,
                n_feeds=n_feeds, run_ids=run_ids)


def build_fedbatch_windows(data, seq_len=6, runs=None):
    """
    Sliding windows for next-day prediction.

    Args:
        data:    dict from load_fedbatch
        seq_len: input window length in days
        runs:    optional array of run indices to include (for train/holdout splits)

    Returns:
        windows (n, seq_len, F+1), targets (n, F), feed_conc (n, F),
        feed_frac (n, 1), run_idx (n,)
    """
    traj, ff, fc = data['traj'], data['feed_frac'], data['feed_conc']
    R, D, F = traj.shape
    runs = np.arange(R) if runs is None else np.asarray(runs)

    windows, targets, wfc, wff, ridx = [], [], [], [], []
    for i in runs:
        for s in range(D - seq_len):
            t = s + seq_len                                   # target day index
            tcol = (np.arange(s, t, dtype=np.float32) / (D - 1))[:, None]
            windows.append(np.concatenate([traj[i, s:t], tcol], axis=1))
            targets.append(traj[i, t])
            wfc.append(fc[i])
            wff.append(ff[i, t])                              # feed on the predicted day
            ridx.append(i)

    return (np.asarray(windows, np.float32), np.asarray(targets, np.float32),
            np.asarray(wfc, np.float32), np.asarray(wff, np.float32)[:, None],
            np.asarray(ridx, np.int32))


def main():
    d = load_fedbatch()
    R, D, F = d['traj'].shape
    print(f'runs={R}  days={D}  features={F}')
    print('features:', d['features'])

    print(f'\nn_feeds     min={d["n_feeds"].min():.0f}  max={d["n_feeds"].max():.0f}  '
          f'mean={d["n_feeds"].mean():.2f}')
    print(f'v_bolus (L) min={d["v_bolus"].min():.5f}  max={d["v_bolus"].max():.5f}  '
          f'mean={d["v_bolus"].mean():.5f}')

    ff = d['feed_frac']
    fed = ff > 0
    print(f'\nfeed_frac on FED days   mean={ff[fed].mean():.4f}  '
          f'min={ff[fed].min():.4f}  max={ff[fed].max():.4f}')
    print(f'feed_frac on UNFED days max={ff[~fed].max():.6f}   (must be 0)')
    assert ff[~fed].max() == 0, 'unfed days must have zero dilution'

    print('\nfeed composition on our features (mmol/L, 0 = not fed):')
    for j, n in enumerate(d['features']):
        c = d['feed_conc'][:, j]
        print(f'  {n:12s} mean={c.mean():8.3f}  max={c.max():8.3f}')

    w, y, fc, fr, ri = build_fedbatch_windows(d, seq_len=6)
    print(f'\nwindows={w.shape}  targets={y.shape}  feed_conc={fc.shape}  '
          f'feed_frac={fr.shape}  runs={len(np.unique(ri))}')
    print(f'windows per run = {len(w) // R}')


if __name__ == '__main__':
    main()
