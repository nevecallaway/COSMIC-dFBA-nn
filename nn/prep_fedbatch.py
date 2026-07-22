#!/usr/bin/env python3
"""
One-time prep for the Golzarijalal 400k in-silico CHO fed-batch dataset
(Figshare DOI 10.26188/28943096, CC BY 4.0).

Extracts a sample of runs from the nested solid 7z, downsamples the hourly
time-courses to daily, keeps the components our model uses, and saves a compact
npz (plus the feed-media and feed-strategy tables). Idempotent: if the output
npz already exists it does nothing, so it is safe to re-run.

Inputs (env vars, all optional):
    FEDBATCH_ZIP   path to the downloaded cho_fedbatch.zip   (default ./cho_fedbatch.zip)
    FEDBATCH_OUT   output npz path                           (default ./cho_fedbatch_daily.npz)
    FEDBATCH_N     number of runs to sample                  (default 2000)
    FEDBATCH_SEED  sampling seed                             (default 0)

Extraction uses the system `7z` if on PATH (streams, low memory), else falls
back to py7zr (pip install py7zr). Keep the zip on scratch if you may want a
larger sample later; the npz is the working artifact.
"""
import os
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd

ZIP    = os.environ.get('FEDBATCH_ZIP', 'cho_fedbatch.zip')
OUTNPZ = os.environ.get('FEDBATCH_OUT', 'cho_fedbatch_daily.npz')
N      = int(os.environ.get('FEDBATCH_N', '2000'))
SEED   = int(os.environ.get('FEDBATCH_SEED', '0'))

BASE    = 'in_silico_fed_batch_CHOK1_cell_culture/'
INNER7Z = BASE + 'in_silico_runs_compressed.7z'

# Keep the big intermediates (inner 7z, extracted CSVs) beside the zip, not in
# the repo's code directory. Both are gitignored.
WORKDIR = os.path.dirname(os.path.abspath(ZIP)) or '.'
RUNDIR  = os.path.join(WORKDIR, 'fedbatch_runs')
INNER_LOCAL = os.path.join(WORKDIR, 'fedbatch_inner.7z')

# our-model-feature -> column in each run CSV (6 of our 8; no cell size / glutamine here)
FEATS = {
    'CellDensity': 'viable cell density(1e6 cells/mL)',
    'Titer':       'mAb_titre(mmol/L)',
    'Glucose':     'glucose(mmol/L)',
    'Asparagine':  'asparagine(mmol/L)',
    'Serine':      'serine(mmol/L)',
    'Glycine':     'glycine(mmol/L)',
}


def extract_sample(inner_7z_path, targets, outdir):
    """Extract the listed internal files from the solid 7z into outdir."""
    os.makedirs(outdir, exist_ok=True)
    if shutil.which('7z'):
        listfile = os.path.join(outdir, 'fedbatch_extract_list.txt')
        with open(listfile, 'w') as fh:
            fh.write('\n'.join(targets) + '\n')
        subprocess.run(['7z', 'x', inner_7z_path, f'@{listfile}', f'-o{outdir}', '-y'],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        import py7zr
        with py7zr.SevenZipFile(inner_7z_path, 'r') as z:
            z.extract(path=outdir, targets=targets)


def main():
    if os.path.exists(OUTNPZ):
        print(f'{OUTNPZ} already exists, nothing to do.')
        return

    zf = zipfile.ZipFile(ZIP)
    info       = pd.read_csv(zf.open(BASE + 'in_silico_runs_information.csv'))
    feed_media = pd.read_csv(zf.open(BASE + 'feed_media_concentration.csv'))
    feed_strat = pd.read_csv(zf.open(BASE + 'feed_strategy.csv'))

    rng = np.random.default_rng(SEED)
    sample_ids = np.sort(rng.choice(info['bioreactor_id'].values, N, replace=False))

    if not os.path.exists(INNER_LOCAL):
        zf.extract(INNER7Z, WORKDIR)
        os.replace(os.path.join(WORKDIR, INNER7Z), INNER_LOCAL)

    targets = [f'in_silico_runs/bioreactor_id({i}).csv' for i in sample_ids]
    print(f'Extracting {len(targets)} runs from the solid archive '
          f'(this is the slow step)...', flush=True)
    extract_sample(INNER_LOCAL, targets, RUNDIR)

    day_hours = np.arange(0, 15) * 24        # days 0..14
    trajs, ids = [], []
    for i in sample_ids:
        p = os.path.join(RUNDIR, 'in_silico_runs', f'bioreactor_id({i}).csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        idx = [(df['time(h)'] - h).abs().idxmin() for h in day_hours]
        trajs.append(df.iloc[idx][list(FEATS.values())].to_numpy(np.float32))
        ids.append(i)

    trajs = np.stack(trajs)                  # (n, 15, 6)
    sub = info.set_index('bioreactor_id').loc[ids]
    np.savez_compressed(
        OUTNPZ,
        trajectories=trajs,
        feature_names=np.array(list(FEATS.keys())),
        bioreactor_id=np.array(ids),
        feed_media_id=sub['feed_media_concentration_id'].to_numpy(),
        feed_strategy_id=sub['feed_strategy_id'].to_numpy(),
        base_media=sub['base_media_version'].to_numpy(),
        feed_media=feed_media.to_numpy(),
        feed_media_cols=np.array(feed_media.columns, dtype=object),
        feed_strategy=feed_strat.to_numpy(),
        feed_strategy_cols=np.array(feed_strat.columns, dtype=object),
        day_hours=day_hours,
    )
    print(f'Saved {trajs.shape} -> {OUTNPZ}')


if __name__ == '__main__':
    main()
