#!/usr/bin/env python3
"""
Generate synthetic training data to augment small real dataset.

Produces 25-component trajectories matching real data structure:
  [Cell Density, Cell Volume, Glucose, Lactate, NH4, Titer,
   Glutamine, Glutamate, L-Asparagine, L-Aspartic acid, L-Serine,
   Glycine, L-Alanine, L-Proline, L-Threonine, L-Histidine, L-Lysine,
   L-Valine, L-Methionine, L-Arginine, L-Tyrosine, L-Isoleucine,
   L-Leucine, L-Phenylalanine, L-Tryptophan]

ICs are sampled by perturbing real reactor day-0 values.
Phase dynamics use a sigmoid centered at a random switch time (day 4-9)
with random steepness, covering the range from gradual to sharp transitions
observed across the 10 real reactors.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Component indices (match real data column order)
IDX_CD  = 0   # Cell Density
IDX_CV  = 1   # Cell Volume
IDX_GLC = 2   # Glucose
IDX_LAC = 3   # Lactate
IDX_NH4 = 4   # NH4
IDX_TIT = 5   # Titer
IDX_GLN = 6   # Glutamine  (consumed fast during growth)
IDX_GLU = 7   # Glutamate  (produced from Gln breakdown)
IDX_ASN = 8   # L-Asparagine
IDX_ASP = 9   # L-Aspartic acid
IDX_SER = 10  # L-Serine
IDX_GLY = 11  # Glycine     (mild production)
IDX_ALA = 12  # L-Alanine   (mild production)
IDX_PRO = 13  # L-Proline
IDX_THR = 14  # L-Threonine
IDX_HIS = 15  # L-Histidine
IDX_LYS = 16  # L-Lysine
IDX_VAL = 17  # L-Valine
IDX_MET = 18  # L-Methionine
IDX_ARG = 19  # L-Arginine
IDX_TYR = 20  # L-Tyrosine
IDX_ILE = 21  # L-Isoleucine
IDX_LEU = 22  # L-Leucine
IDX_PHE = 23  # L-Phenylalanine
IDX_TRP = 24  # L-Tryptophan

N_COMPONENTS = 25

# AA consumption/production rates per unit CD per day during growth.
# Negative = consumed, positive = produced.
# Calibrated so a reactor starting with IC~0.85 and CD~0.12 depletes
# ~20-40% of each AA over a 6-day growth phase.
_AA_GROWTH_RATES = {
    IDX_GLN: -0.30,   # rapid depletion: key N-source
    IDX_GLU: +0.18,   # produced via GDH from Gln
    IDX_ASN: -0.18,
    IDX_ASP: -0.12,
    IDX_SER: -0.15,
    IDX_GLY: +0.04,   # secreted as overflow
    IDX_ALA: +0.05,   # major secreted amino acid
    IDX_PRO: -0.08,
    IDX_THR: -0.12,
    IDX_HIS: -0.06,
    IDX_LYS: -0.14,
    IDX_VAL: -0.14,
    IDX_MET: -0.08,
    IDX_ARG: -0.10,
    IDX_TYR: -0.08,
    IDX_ILE: -0.13,
    IDX_LEU: -0.16,
    IDX_PHE: -0.10,
    IDX_TRP: -0.04,
}

# During production phase rates are reduced (cells shift metabolism to Titer)
_PRODUCTION_RATE_FACTOR = 0.25


def _sigmoid(t, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (t - x0)))


def load_real_ics(data_file='data_2.csv'):
    """
    Load real reactor day-0 values from CSV.
    Returns array of shape (n_reactors, N_COMPONENTS) and component list.
    """
    df = pd.read_csv(data_file)
    excluded = ['Vessel', 'Time', 'Production phase fraction']
    components = [c for c in df.columns if c not in excluded]

    for col in df.columns:
        if col != 'Vessel':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time'])

    ics = []
    for reactor in sorted(df['Vessel'].dropna().unique()):
        row = df[df['Vessel'] == reactor].sort_values('Time').iloc[0]
        ic = row[components].values.astype(float)
        if not np.any(np.isnan(ic)):
            ics.append(ic)

    return np.array(ics), components


def generate_synthetic_trajectory(real_ics, n_timepoints=13):
    """
    Generate one 25-component trajectory.

    Phase profile: sigmoid from 0 → 1 centered at a random switch_time
    drawn from U[4, 9] days, with steepness drawn from U[0.5, 3.0].
    This spans the range from gradual (R0001, k≈0.8) to near-step-function
    (R0011, k≈3) transitions seen in the 10 real reactors.
    """
    # Sample IC by adding Gaussian noise to a random real reactor's day-0
    base_ic = real_ics[np.random.randint(len(real_ics))].copy()
    noise = np.random.normal(0, 0.05, N_COMPONENTS)
    ic = np.clip(base_ic + noise, 0.01, 1.0)

    # Phase parameters: cover the observed range
    switch_time = np.random.uniform(4.0, 9.0)
    steepness   = np.random.uniform(0.5, 3.0)

    time  = np.linspace(0, 13, n_timepoints)
    phase = _sigmoid(time, switch_time, steepness)  # in [0, 1]

    state = ic.copy()
    trajectory = [state.copy()]

    for t_idx in range(1, n_timepoints):
        p  = phase[t_idx]
        dt = time[t_idx] - time[t_idx - 1]
        cd  = state[IDX_CD]
        glc = state[IDX_GLC]

        # Growth rate: logistic, suppressed in production phase
        mu_growth = 0.15 * cd * (1.0 - cd)
        mu_prod   = 0.02 * cd * (1.0 - cd)
        mu = (1.0 - p) * mu_growth + p * mu_prod

        dstate = np.zeros(N_COMPONENTS)

        # Core metabolites
        dstate[IDX_CD]  = mu
        dstate[IDX_CV]  = 0.04 * mu + np.random.normal(0, 0.003)  # volume tracks mass
        dstate[IDX_GLC] = -(0.30 * (1.0 - p) + 0.10 * p) * cd    # glucose consumed
        dstate[IDX_LAC] = ( 0.08 * (1.0 - p) - 0.04 * p) * cd    # lactate: prod then consumed
        dstate[IDX_NH4] = ( 0.06 * (1.0 - p) + 0.02 * p) * cd    # NH4 from AA catabolism
        dstate[IDX_TIT] = ( 0.01 * (1.0 - p) + 0.18 * p) * cd    # titer: rises in production

        # Amino acids
        for aa_idx, growth_rate in _AA_GROWTH_RATES.items():
            prod_rate = growth_rate * _PRODUCTION_RATE_FACTOR
            effective_rate = (1.0 - p) * growth_rate + p * prod_rate
            dstate[aa_idx] = effective_rate * cd

        # Add small biological noise
        dstate += np.random.normal(0, 0.002, N_COMPONENTS)

        state = np.clip(state + dstate * dt, 0.0, 1.0)
        trajectory.append(state.copy())

    return np.array(trajectory), time, ic, phase


def generate_dataset(n_samples=1000, n_timepoints=13,
                     data_file='data_2.csv',
                     output_file='synthetic_training.npz'):
    """
    Generate synthetic dataset with ICs sampled from real reactor data.

    Output .npz contains:
      trajectories: (n_samples, n_timepoints, N_COMPONENTS)
      times:        (n_samples, n_timepoints)
      ics:          (n_samples, N_COMPONENTS)
      phases:       (n_samples, n_timepoints)
      components:   list of N_COMPONENTS component names
    """
    print(f"\nLoading real ICs from {data_file}...")
    real_ics, components = load_real_ics(data_file)
    print(f"  Found {len(real_ics)} reactors, {len(components)} components")
    assert len(components) == N_COMPONENTS, (
        f"Expected {N_COMPONENTS} components, got {len(components)}: {components}"
    )

    print(f"Generating {n_samples} synthetic trajectories...")
    trajectories, times, ics, phases = [], [], [], []

    for i in range(n_samples):
        traj, t, ic, phase = generate_synthetic_trajectory(real_ics, n_timepoints)
        trajectories.append(traj)
        times.append(t)
        ics.append(ic)
        phases.append(phase)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples}")

    trajectories = np.array(trajectories)
    times        = np.array(times)
    ics          = np.array(ics)
    phases       = np.array(phases)

    np.savez(
        output_file,
        trajectories=trajectories,
        times=times,
        ics=ics,
        phases=phases,
        components=np.array(components, dtype=object),
    )

    print(f"\nSaved to {output_file}")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  ICs:          {ics.shape}")
    _print_stats(trajectories, components)

    return trajectories, times, ics, phases, components


def load_synthetic_dataset(filename='synthetic_training.npz'):
    data = np.load(filename, allow_pickle=True)
    return (
        data['trajectories'],
        data['times'],
        data['ics'],
        data['phases'],
        data['components'].tolist(),
    )


def _print_stats(trajectories, components):
    print(f"\n{'Component':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print('-' * 61)
    for i, name in enumerate(components):
        vals = trajectories[:, :, i]
        print(f"  {name:<23} {vals.mean():8.3f} {vals.std():8.3f} "
              f"{vals.min():8.3f} {vals.max():8.3f}")


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'data_2.csv'
    generate_dataset(
        n_samples=1000,
        n_timepoints=13,
        data_file=data_file,
        output_file='synthetic_training.npz',
    )
