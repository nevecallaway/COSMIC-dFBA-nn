#!/usr/bin/env python3
"""
Denormalize the measured data_2 onto our physical scale.

data_2 is measured but normalized per reactor/component (divided by its own max).
We recover approximate physical units as

    real_phys = data2_norm * ODE_peak

where ODE_peak is the synthetic trajectory's max for that reactor/component. The
shape is real (from data_2); the scale is our best guess (raw data_2 is
proprietary). Shared by train_real.py (training target) and evaluate.py
(--real-target ground truth) so both use identical denormalization.
"""

import numpy as np
import pandas as pd

from generate_synthetic_ode import WINDOW_FEATURE_INDICES, T_EVAL

REACTOR_IDS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
               'R0006', 'R0008', 'R0010', 'R0011', 'R0012']


def denormalize_data2(data2_csv, ode_traj, n_original):
    """
    data_2 (normalized) -> approximate physical, using the ODE per-reactor
    per-component peak as the scale.

    Args:
        data2_csv:  path to data_2.csv
        ode_traj:   (>=n_original, N_DAYS, 25) synthetic ODE trajectories (physical)
        n_original: number of real reactors

    Returns:
        (n_original, N_DAYS, 25) array; the model-feature components are filled
        from the denormalized real data, the rest keep their ODE values (unused).
    """
    df2 = pd.read_csv(data2_csv, skiprows=1)
    df2.columns = ['Vessel', 'Time', 'Phase'] + [f'C{i}' for i in range(25)]
    df2['Time'] = pd.to_numeric(df2['Time'], errors='coerce')
    df2 = df2.dropna(subset=['Time'])

    real = ode_traj[:n_original].copy()
    for r in range(n_original):
        rdf = df2[df2['Vessel'] == REACTOR_IDS[r]].sort_values('Time')
        rdays = rdf['Time'].to_numpy(float)
        for c in WINDOW_FEATURE_INDICES:
            norm = pd.to_numeric(rdf[f'C{c}'], errors='coerce').to_numpy(float)
            interp = np.interp(T_EVAL, rdays, norm)          # onto integer days 0..12
            omax = ode_traj[r, :, c].max()
            real[r, :, c] = interp * (omax if omax > 0 else 1.0)
    return real
