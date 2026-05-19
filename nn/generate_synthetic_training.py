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


def load_real_trajectories(data_file=None):
    """
    Load full trajectories for every reactor.
    Returns:
      trajectories: (n_reactors, n_timepoints, N_COMPONENTS)  – NaN-filled rows dropped
      phases:       (n_reactors, n_timepoints)
      times:        (n_reactors, n_timepoints)
      components:   list of N_COMPONENTS names
    """
    if data_file is None:
        data_file = Path(__file__).parent / 'data' / 'data_2.csv'
    df = pd.read_csv(data_file)
    excluded = ['Vessel', 'Time', 'Production phase fraction']
    components = [c for c in df.columns if c not in excluded]

    for col in df.columns:
        if col != 'Vessel':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time'])

    trajs, phases_list, times_list = [], [], []
    for reactor in sorted(df['Vessel'].dropna().unique()):
        rdf = df[df['Vessel'] == reactor].sort_values('Time')
        traj = rdf[components].values.astype(float)
        # Drop rows with any NaN component
        valid = ~np.any(np.isnan(traj), axis=1)
        traj  = traj[valid]
        ph    = rdf['Production phase fraction'].values[valid] if 'Production phase fraction' in rdf.columns else np.zeros(valid.sum())
        t     = rdf['Time'].values[valid]
        trajs.append(traj)
        phases_list.append(ph)
        times_list.append(t)

    # Pad / truncate to the most common length (all real reactors have 13 points)
    n_tp = max(len(t) for t in times_list)
    n_r  = len(trajs)
    nc   = len(components)
    out_trajs  = np.zeros((n_r, n_tp, nc))
    out_phases = np.zeros((n_r, n_tp))
    out_times  = np.zeros((n_r, n_tp))
    for i, (tr, ph, t) in enumerate(zip(trajs, phases_list, times_list)):
        nt = len(t)
        out_trajs[i, :nt]  = tr
        out_phases[i, :nt] = ph
        out_times[i, :nt]  = t
        if nt < n_tp:                     # pad with last value
            out_trajs[i, nt:]  = tr[-1]
            out_phases[i, nt:] = ph[-1]
            out_times[i, nt:]  = t[-1]

    return out_trajs, out_phases, out_times, components


def generate_synthetic_trajectory(real_ics, n_timepoints=13):
    """
    Generate one 25-component trajectory.

    Phase profile: sigmoid from 0 → 1 centered at a random switch_time
    drawn from U[4, 9] days, with steepness drawn from U[0.5, 3.0].
    This spans the range from gradual (R0001, k≈0.8) to near-step-function
    (R0011, k≈3) transitions seen in the 10 real reactors.

    Titer dynamics: ~60% of trajectories are monotonically increasing;
    ~40% peak then decline (matching the fed-batch dilution effect seen in
    R0002, R0003, R0006, R0011, R0012 where titer drops 30-80% post-peak).

    L-Aspartic acid (comp_9): some trajectories start high and crash sharply,
    matching the unusual kinetics in R0003 and R0011.
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

    # Titer post-peak decline: 40% of runs show peak-then-decline, matching
    # R0002/R0003/R0006/R0011/R0012 which drop 0.10-0.14 units/day post-peak.
    # Peak day is phase switch + 0.5-2 days; before that, normal production.
    titer_decline = np.random.random() < 0.40
    titer_decline_rate = np.random.uniform(0.08, 0.14) if titer_decline else 0.0
    titer_peak_day = switch_time + np.random.uniform(0.5, 2.0)

    # L-Aspartic acid crash: ~20% of runs show IC=high then sharp depletion
    # (matching R0003 IC=1.0 → 0.034 by day 4; R0011 IC=1.0 → -0.188 day 3)
    asp_crash = (np.random.random() < 0.20) and (ic[IDX_ASP] > 0.7)

    # Glucose fed-batch dynamics. Real data shows two patterns:
    #   Low IC (<0.7, 8/10 reactors): glucose is fed over days 0→peak_day,
    #     rising to 1.0, then either stays high (30%) or declines (70%).
    #   High IC (≥0.7, R0003/R0004): already at max, consumption only.
    glc_fed_batch  = ic[IDX_GLC] < 0.7
    glc_peak_day   = np.random.uniform(2.0, 4.0)
    glc_stays_high = np.random.random() < 0.30   # 30% stay high post-peak
    # Constant feed rate that would bring glucose from IC to 1.0 by peak_day
    glc_feed_rate  = (1.0 - ic[IDX_GLC]) / glc_peak_day if glc_fed_batch else 0.0

    state = ic.copy()
    trajectory = [state.copy()]

    for t_idx in range(1, n_timepoints):
        p  = phase[t_idx]
        dt = time[t_idx] - time[t_idx - 1]
        cd  = state[IDX_CD]

        # Growth rate: logistic, suppressed in production phase.
        # Rate 0.25 gives CD ~0.32 at day 5, matching real reactor densities.
        mu = ((1.0 - p) * 0.25 + p * 0.04) * cd * (1.0 - cd)

        dstate = np.zeros(N_COMPONENTS)

        # Core metabolites
        dstate[IDX_CD]  = mu
        dstate[IDX_CV]  = 0.04 * mu + np.random.normal(0, 0.003)

        # Glucose: fed-batch rise to 1.0 over peak_day, then either flat or declining.
        glc_consumption = (0.30 * (1.0 - p) + 0.10 * p) * cd
        if glc_fed_batch and time[t_idx] < glc_peak_day:
            dstate[IDX_GLC] = glc_feed_rate - glc_consumption
        elif glc_stays_high:
            dstate[IDX_GLC] = -0.01 * cd   # feeding ≈ consumption, nearly flat
        else:
            dstate[IDX_GLC] = -glc_consumption
        dstate[IDX_LAC] = ( 0.08 * (1.0 - p) - 0.04 * p) * cd
        dstate[IDX_NH4] = ( 0.06 * (1.0 - p) + 0.02 * p) * cd

        # Titer production rate 0.40 gets synthetic Titer into [0.7, 1.0] range
        # matching real reactors. 40% of runs decline post-peak proportionally
        # (10-14% per day of current value), matching R0002/R0003/R0011.
        if titer_decline and time[t_idx] > titer_peak_day:
            dstate[IDX_TIT] = -titer_decline_rate * state[IDX_TIT]
        else:
            dstate[IDX_TIT] = (0.01 * (1.0 - p) + 0.40 * p) * cd

        # Amino acids
        for aa_idx, growth_rate in _AA_GROWTH_RATES.items():
            prod_rate = growth_rate * _PRODUCTION_RATE_FACTOR
            dstate[aa_idx] = ((1.0 - p) * growth_rate + p * prod_rate) * cd

        # L-Aspartic acid crash: rapid depletion in early growth phase
        if asp_crash and p < 0.3:
            dstate[IDX_ASP] += -0.8 * state[IDX_ASP]  # fast crash toward 0

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


def generate_gaussian_dataset(n_samples=20000, n_timepoints=13,
                               data_file='data/data_2.csv',
                               doe_file='data/data_1.csv',
                               rates_file='data/data_3.csv',
                               output_file='synthetic_training.npz',
                               noise_scale=1.0):
    """
    Generate synthetic data by Gaussian augmentation of real trajectories.

    For each sample:
      1. Pick a random real reactor as the base trajectory.
      2. Add independent Gaussian noise at every (timepoint, component) cell.
      3. Copy the base reactor's DoE params (O2, AAs, Glc) + small noise.
      4. Copy the base reactor's 50 specific rates (25 growth + 25 production)
         + small noise — gives the model per-reactor metabolic rate information
         that's critical for components like L-Arginine where the IC alone
         provides no differentiating signal.
    """
    print(f"\nLoading real trajectories from {data_file}...")
    real_trajs, real_phases, real_times, components = load_real_trajectories(data_file)
    n_reactors, n_tp, nc = real_trajs.shape
    print(f"  {n_reactors} reactors × {n_tp} timepoints × {nc} components")
    assert nc == N_COMPONENTS, f"Expected {N_COMPONENTS} components, got {nc}"

    reactor_order = sorted(set(
        pd.read_csv(data_file)['Vessel'].dropna().unique()
    ))

    # Load DoE parameters (O2, AAs, Glc)
    doe_map = {}
    try:
        df_doe = pd.read_csv(doe_file, header=1)
        df_doe = df_doe[df_doe['Vessel'].str.match(r'^R\d{4}$', na=False)]
        for _, row in df_doe.iterrows():
            doe_map[str(row['Vessel'])] = np.array(
                [pd.to_numeric(row['O2'], errors='coerce'),
                 pd.to_numeric(row['AAs'], errors='coerce'),
                 pd.to_numeric(row['Glc'], errors='coerce')], dtype=float)
        print(f"  DoE params loaded for {len(doe_map)} reactors")
    except FileNotFoundError:
        print(f"  (DoE file not found — doe_params will be zeros)")
    real_doe = np.array([doe_map.get(r, np.zeros(3)) for r in reactor_order])  # (n_r, 3)

    # Load specific rates (25 growth + 25 production per reactor)
    real_rates = np.zeros((n_reactors, 50))
    try:
        from utils import load_specific_rates
        real_rates = load_specific_rates(rates_file, reactor_order)   # (n_r, 50)
        print(f"  Specific rates loaded: {real_rates.shape}")
    except (FileNotFoundError, ImportError):
        print(f"  (Rates file not found — specific_rates will be zeros)")

    # Per-cell std across reactors
    cell_std    = np.clip(np.std(real_trajs, axis=0), 0.02, None)
    noise_sigma = noise_scale * cell_std

    print(f"Generating {n_samples} Gaussian-augmented trajectories "
          f"(noise_scale={noise_scale})...")

    trajectories    = np.empty((n_samples, n_tp, nc))
    times           = np.empty((n_samples, n_tp))
    ics             = np.empty((n_samples, nc))
    phases          = np.empty((n_samples, n_tp))
    doe_params      = np.empty((n_samples, 3))
    specific_rates  = np.empty((n_samples, 50))

    for i in range(n_samples):
        base_idx = np.random.randint(n_reactors)
        noise    = np.random.normal(0.0, noise_sigma)
        traj     = np.clip(real_trajs[base_idx] + noise, 0.0, 1.0)

        trajectories[i]   = traj
        times[i]          = real_times[base_idx]
        ics[i]            = traj[0]
        phases[i]         = real_phases[base_idx]
        doe_params[i]     = real_doe[base_idx] + np.random.normal(0, 0.1, 3)
        specific_rates[i] = real_rates[base_idx] + np.random.normal(0, 0.1, 50)

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_samples}")

    np.savez(
        output_file,
        trajectories=trajectories,
        times=times,
        ics=ics,
        phases=phases,
        doe_params=doe_params,
        specific_rates=specific_rates,
        components=np.array(components, dtype=object),
    )

    print(f"\nSaved to {output_file}")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  DoE params:   {doe_params.shape}")
    _print_stats(trajectories, components)

    return trajectories, times, ics, phases, doe_params, components


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
    _here = Path(__file__).parent           # nn/ directory, works from any cwd
    generate_gaussian_dataset(
        n_samples=20000,
        n_timepoints=13,
        data_file=str(_here / 'data' / 'data_2.csv'),
        doe_file=str(_here / 'data' / 'data_1.csv'),
        rates_file=str(_here / 'data' / 'data_3.csv'),
        output_file=str(_here / 'synthetic_training.npz'),
    )
