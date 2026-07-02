#!/usr/bin/env python3
"""
Physics-based synthetic data generator for COSMIC-dFBA surrogate.

Implements the ODE system specified by Sarat (per-interval integration):

    dX/dt     = v_CD_net * X(t)                              [cell density]
    dX_bm/dt  = v_bm_net * X_bm(t)                          [biomass/cell volume]
    dC_tit/dt = v_tit_net * X(t) - eta * F * C_tit(t)       [titer]
    dC_i/dt   = F * (C_i_in - C_i(t)) + v_i_net * X(t)      [all other metabolites]

    v_net = (1 - f) * v_growth + f * v_prod                  [phase blending]

Integration approach: 1-day intervals (day 0-1, 1-2, ..., 11-12).
Within each interval:
  - f = phase fraction at the END of the interval (from data_2)
  - v_net is constant
  - Final state of interval becomes IC for next interval

Units:
    Cell Density (X)   : E9 cells/L  (magnitude used directly; initial = 0.5)
    Cell Volume (X_bm) : um^3 / 1000 (CHO ~1596 um^3/cell -> initial = 1.6; per Sarat)
    Metabolites        : mmol/L
    Titer              : mg/L

Flux units (data_3, corrected per Sarat):
    Cell Density : 1/day
    Cell Volume  : 1/day
    Metabolites  : mmol / (E9 cells * day)   -- NOTE: table header says E6, actual is E9
    Titer        : mg / (E9 cells * day)

eta = 1 for titer at all times (per Sarat: simplifies dynamics, removes day-8 artifact).
No logistic cap on cell density: phase blending handles the growth-to-stationary reduction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F       = 1.0    # perfusion rate (bioreactor volumes/day, confirmed from paper)
N_DAYS  = 13     # day 0 through day 12 (13 timepoints)
T_EVAL  = np.arange(0, N_DAYS, dtype=float)

# ---------------------------------------------------------------------------
# Window constants -- must match model.py FEATURE_INDICES / SEQ_LEN
# ---------------------------------------------------------------------------
SEQ_LEN = 6   # days per input window
WINDOW_FEATURE_INDICES = [
    0,   # Cell Density
    1,   # Cell Size
    5,   # Titer
    2,   # Glucose
    6,   # Glutamine
    8,   # Asparagine
    10,  # Serine
    11,  # Glycine
]
N_WINDOW_FEATURES = len(WINDOW_FEATURE_INDICES)  # 8

# ---------------------------------------------------------------------------
# Component indices -- match data_2 / data_3 row order
# ---------------------------------------------------------------------------
IDX_CD  = 0   # Cell Density
IDX_CV  = 1   # Cell Volume (biomass)
IDX_GLC = 2   # Glucose
IDX_LAC = 3   # Lactate
IDX_NH4 = 4   # NH4
IDX_TIT = 5   # Titer
IDX_GLN = 6   # Glutamine
IDX_GLU = 7   # Glutamate
IDX_ASN = 8   # L-Asparagine
IDX_ASP = 9   # L-Aspartic acid
IDX_SER = 10  # L-Serine
IDX_GLY = 11  # Glycine
IDX_ALA = 12  # L-Alanine
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
AAS_INDICES  = list(range(6, N_COMPONENTS))

# ---------------------------------------------------------------------------
# Initial conditions and feed concentrations
# Cell density : 0.5 E9 cells/L (Sarat anchor)
# Cell volume  : single-cell volume in um^3, scaled by 1/1000 per Sarat
#                CHO diameter ~14-15 um -> radius ~7.25 um -> V = (4/3)pi*r^3 ~ 1596 um^3
#                scaled initial value: 1596/1000 = 1.596 -> use 1.6
# Metabolites  : DMEM/F12 nominal, mmol/L
# Titer        : 0 (starts at 0)
# ---------------------------------------------------------------------------
C_NOMINAL = np.array([
    0.5,     # Cell Density  (E9 cells/L)
    1.6,     # Cell Volume   (um^3 / 1000, per Sarat: keep initial value between 1-10)
    17.5,    # Glucose       (mmol/L)
    0.0,     # Lactate       (mmol/L)
    0.0,     # NH4           (mmol/L)
    0.0,     # Titer         (mg/L)
    2.5,     # Glutamine     (mmol/L)
    0.05,    # Glutamate     (mmol/L)
    0.05,    # L-Asparagine  (mmol/L)
    0.05,    # L-Aspartic acid (mmol/L)
    0.25,    # L-Serine      (mmol/L)
    0.25,    # Glycine       (mmol/L)
    0.05,    # L-Alanine     (mmol/L)
    0.15,    # L-Proline     (mmol/L)
    0.45,    # L-Threonine   (mmol/L)
    0.15,    # L-Histidine   (mmol/L)
    0.50,    # L-Lysine      (mmol/L)
    0.45,    # L-Valine      (mmol/L)
    0.12,    # L-Methionine  (mmol/L)
    0.70,    # L-Arginine    (mmol/L)
    0.20,    # L-Tyrosine    (mmol/L)
    0.42,    # L-Isoleucine  (mmol/L)
    0.45,    # L-Leucine     (mmol/L)
    0.21,    # L-Phenylalanine (mmol/L)
    0.044,   # L-Tryptophan  (mmol/L)
], dtype=float)

# Perfusion feed: no cells, no biomass, no lactate, no NH4, no titer
CIN_NOMINAL = C_NOMINAL.copy()
CIN_NOMINAL[IDX_CD]  = 0.0
CIN_NOMINAL[IDX_CV]  = 0.0
CIN_NOMINAL[IDX_LAC] = 0.0
CIN_NOMINAL[IDX_NH4] = 0.0
CIN_NOMINAL[IDX_TIT] = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rates(data3_path):
    """
    Load growth and production phase specific rates from data_3.csv.

    Flux units (corrected per Sarat): mmol / (E9 cells * day) for metabolites,
    mg / (E9 cells * day) for titer, 1/day for cell density and volume.

    Returns:
        rates_growth : dict {reactor_id: np.array shape (N_COMPONENTS,)}
        rates_prod   : dict {reactor_id: np.array shape (N_COMPONENTS,)}
        reactor_ids  : list of reactor ID strings
    """
    df = pd.read_csv(data3_path, header=None)

    # Row index 1: reactor IDs in columns 2-11 (growth) and 12-21 (production)
    reactor_ids = [str(df.iloc[1, c]).strip() for c in range(2, 12)]

    rates_growth = {r: np.zeros(N_COMPONENTS) for r in reactor_ids}
    rates_prod   = {r: np.zeros(N_COMPONENTS) for r in reactor_ids}

    for comp_idx in range(N_COMPONENTS):
        row = df.iloc[2 + comp_idx]
        for ri, reactor in enumerate(reactor_ids):
            rates_growth[reactor][comp_idx] = float(row.iloc[2 + ri])
            rates_prod[reactor][comp_idx]   = float(row.iloc[12 + ri])

    return rates_growth, rates_prod, reactor_ids


def load_phase_fractions(data2_path):
    """
    Load production phase fraction (f) per reactor at each integer day from data_2.csv.

    Per Sarat: use the value at the END of each interval as the constant f
    for that interval. Returned as a lookup dict.

    Returns:
        pm_dict : dict {reactor_id: {day_int: float}}
    """
    df = pd.read_csv(data2_path, skiprows=1)
    df.columns = (
        ['Vessel', 'Time', 'Phase'] +
        ['C{}'.format(i) for i in range(N_COMPONENTS)]
    )
    df['Time']  = pd.to_numeric(df['Time'],  errors='coerce')
    df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
    df = df.dropna(subset=['Time', 'Phase'])

    pm_dict = {}
    for reactor in df['Vessel'].dropna().unique():
        rdf = df[df['Vessel'] == reactor].sort_values('Time')
        by_day = {}
        for _, row in rdf.iterrows():
            day = int(round(float(row['Time'])))
            by_day[day] = float(np.clip(row['Phase'], 0.0, 1.0))
        pm_dict[str(reactor)] = by_day

    return pm_dict


def load_doe(data1_path):
    """
    Load DoE coded levels (-1, 0, +1) per reactor from data_1.csv.

    Returns:
        doe_dict : dict {reactor_id: {'O2': float, 'AAs': float, 'Glc': float}}
    """
    df = pd.read_csv(data1_path, skiprows=1)
    doe_dict = {}
    for _, row in df.iterrows():
        vessel = str(row.get('Vessel', '')).strip()
        if not vessel.startswith('R'):
            continue
        doe_dict[vessel] = {
            'O2':  float(row['O2']),
            'AAs': float(row['AAs']),
            'Glc': float(row['Glc']),
        }
    return doe_dict


def make_cin(doe):
    """
    Scale nominal Cin by DoE coded levels.

    level -1 = 0.5x nominal, level 0 = nominal, level +1 = 2x nominal.
    Applied to feed only (base media stays the same).
    """
    cin = CIN_NOMINAL.copy()
    cin[IDX_GLC]      *= 2.0 ** doe['Glc']
    cin[AAS_INDICES]  *= 2.0 ** doe['AAs']
    return cin


# ---------------------------------------------------------------------------
# Interval-based ODE integration
# ---------------------------------------------------------------------------

def _make_interval_ode(v_net, cin):
    """
    Build the ODE function for one 1-day interval with constant v_net and eta.

    Per paper eq. 2 (uniform form for all components):
        dC_i/dt = F * (C_i_in - eta * C_i) + v_i * X

    Applied per component:
        dX/dt     = v_CD * X                           [eta=0, X_in=0]
        dX_bm/dt  = v_bm * X                           [eta=0, X_bm_in=0; driver is X not X_bm]
        dC_tit/dt = v_tit * X - F * C_tit              [eta=1 always]
        dC_i/dt   = F * (Cin_i - C_i) + v_i * X       [eta=1, all other metabolites]
    """
    def ode(t, C):
        X    = max(C[IDX_CD], 0.0)
        dC   = np.zeros(N_COMPONENTS)

        # Cell density: phase blending handles growth-to-stationary transition
        dC[IDX_CD] = v_net[IDX_CD] * X

        # Cell size: driven by cell density (per paper eq. 2, C_1 = X for all components)
        dC[IDX_CV] = v_net[IDX_CV] * X

        # Titer: production minus washout (eta=1 always, per Sarat)
        dC[IDX_TIT] = v_net[IDX_TIT] * X - F * max(C[IDX_TIT], 0.0)

        # All other metabolites
        for i in range(N_COMPONENTS):
            if i in (IDX_CD, IDX_CV, IDX_TIT):
                continue
            dC[i] = F * (cin[i] - C[i]) + v_net[i] * X

        return dC

    return ode


def generate_reactor(reactor_id, v_growth, v_prod, pm_by_day, doe):
    """
    Integrate the ODE interval by interval (1 day each) per Sarat's specification.

    For interval [d, d+1]:
      - f  = pm_by_day[d+1]  (phase fraction at end of interval)
      - v_net = (1-f)*v_growth + f*v_prod  (constant within interval)
      - eta_titer = 1 always (per Sarat)

    Returns:
        trajectory : np.array shape (N_DAYS, N_COMPONENTS)
        times      : np.array shape (N_DAYS,)
    """
    cin        = make_cin(doe)
    C          = C_NOMINAL.copy()
    trajectory = [C.copy()]

    n_intervals = N_DAYS - 1   # 12 intervals for days 0-12

    # Fallback f if a day is missing from data_2
    max_day    = max(pm_by_day.keys())
    f_fallback = pm_by_day[max_day]

    for d in range(n_intervals):
        f       = pm_by_day.get(d + 1, f_fallback)
        v_net   = (1.0 - f) * v_growth + f * v_prod


        ode_fn  = _make_interval_ode(v_net, cin)

        sol = solve_ivp(
            ode_fn,
            t_span=(0.0, 1.0),
            y0=C.copy(),
            method='RK45',
            t_eval=[1.0],
            max_step=0.1,
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            print(f'  WARNING: {reactor_id} interval [{d},{d+1}]: {sol.message}')
            C_next = C.copy()
        else:
            C_next = sol.y[:, -1].copy()

        C_next = np.clip(C_next, 0.0, None)
        trajectory.append(C_next)
        C = C_next

    return np.array(trajectory), T_EVAL


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------

def build_windows(trajectories, doe_params=None):
    """
    Build sliding windows from trajectories for next-day prediction.

    For each reactor and each starting day d in [0, N_DAYS - SEQ_LEN):
        x:   days [d, d+SEQ_LEN)  raw physical units
        y:   day  d+SEQ_LEN       raw physical units
        doe: DoE vector for that reactor (O2 coded, Glc mmol/L, AAs mmol/L)

    Normalization is NOT applied here. train.py fits a MinMaxScaler on
    training-only windows and applies it there so the scaler is never
    contaminated with validation data.

    The last column of each window step is the normalized day index
    (day / (N_DAYS - 1)), already in [0, 1]. train.py keeps this column
    out of the MinMaxScaler and passes it through as-is.

    Returns:
        windows:          np.ndarray (n_obs, SEQ_LEN, N_WINDOW_FEATURES+1)  raw feats + time
        targets:          np.ndarray (n_obs, N_WINDOW_FEATURES)              raw (no time)
        window_doe:       np.ndarray (n_obs, 3) or None
        reactor_indices:  np.ndarray (n_obs,)  which reactor each window came from
    """
    sub = trajectories[:, :, WINDOW_FEATURE_INDICES].astype(np.float32)
    N, T, _ = sub.shape
    n_days_total = T  # should be N_DAYS = 13

    windows, targets, window_doe, reactor_indices = [], [], [], []
    for i in range(N):
        for d in range(T - SEQ_LEN):
            feats = sub[i, d : d + SEQ_LEN, :]
            time_col = (np.arange(d, d + SEQ_LEN, dtype=np.float32) / (n_days_total - 1))[:, None]
            windows.append(np.concatenate([feats, time_col], axis=1))
            targets.append(sub[i, d + SEQ_LEN, :])
            if doe_params is not None:
                window_doe.append(doe_params[i])
            reactor_indices.append(i)

    doe_arr = np.array(window_doe, dtype=np.float32) if window_doe else None

    return (
        np.array(windows, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        doe_arr,
        np.array(reactor_indices, dtype=np.int32),
    )


# Extra reactor generation
# ---------------------------------------------------------------------------

def generate_extra(n_extra, rates_growth, rates_prod, reactor_ids, pm_dict, doe_dict,
                   seed=0, sample_rates=False, rate_mix=0.0, rate_scale=1.0):
    """
    Generate n_extra additional synthetic reactors with randomly sampled DoE.

    Rate generation modes:
      - sample_rates=False, rate_mix=0: all donor-copied (legacy)
      - sample_rates=True, rate_mix=0:  all sampled from Gaussian
      - rate_mix=0.5: 50% donor-copied, 50% sampled from Gaussian
    rate_scale: multiplier on covariance (0.25 = tighter, 1.0 = full variance)

    Returns:
        trajectories: np.ndarray (n_extra, N_DAYS, N_COMPONENTS)
        doe_params:   np.ndarray (n_extra, 3)
    """
    rng = np.random.default_rng(seed)
    trajs, does = [], []

    use_sampling = sample_rates or rate_mix > 0
    if use_sampling:
        g_matrix = np.array([rates_growth[r] for r in reactor_ids])
        p_matrix = np.array([rates_prod[r] for r in reactor_ids])
        g_mean, g_cov = g_matrix.mean(axis=0), np.cov(g_matrix, rowvar=False)
        p_mean, p_cov = p_matrix.mean(axis=0), np.cov(p_matrix, rowvar=False)
        g_cov = g_cov * rate_scale + np.eye(N_COMPONENTS) * 1e-8
        p_cov = p_cov * rate_scale + np.eye(N_COMPONENTS) * 1e-8
        print(f'  Rate sampling: scale={rate_scale}, mix={rate_mix}')

    n_reject = 0
    k = 0
    while k < n_extra:
        doe = {
            'O2':  float(rng.choice([-1, 0, 1])),
            'AAs': float(rng.uniform(-1, 1)),
            'Glc': float(rng.uniform(-1, 1)),
        }
        cin = make_cin(doe)
        glc_conc = cin[IDX_GLC]
        aas_conc = float(sum(cin[i] for i in AAS_INDICES))

        donor = reactor_ids[rng.integers(len(reactor_ids))]

        do_sample = sample_rates or (rate_mix > 0 and rng.random() < rate_mix)
        if do_sample:
            v_growth = rng.multivariate_normal(g_mean, g_cov)
            v_prod   = rng.multivariate_normal(p_mean, p_cov)
        else:
            v_growth = rates_growth[donor]
            v_prod   = rates_prod[donor]

        traj, _ = generate_reactor(
            f'extra_{k:04d}',
            v_growth,
            v_prod,
            pm_dict[donor],
            doe,
        )

        # Reject trajectories with negative concentrations or NaN
        if np.any(np.isnan(traj)) or np.any(traj < -1e-3):
            n_reject += 1
            if n_reject > n_extra * 5:
                print(f'  WARNING: too many rejected samples ({n_reject}), stopping')
                break
            continue

        trajs.append(traj)
        does.append([doe['O2'], glc_conc, aas_conc])
        k += 1
        if k % 10 == 0:
            print(f'  Generated {k}/{n_extra} extra reactors'
                  + (f' ({n_reject} rejected)' if n_reject else ''))

    if n_reject:
        print(f'  Total rejected: {n_reject}')

    return np.array(trajs), np.array(does, dtype=np.float32)


# Main generation
# ---------------------------------------------------------------------------

def generate_all(data_dir=None, output_file=None, n_extra=50,
                 sample_rates=False, rate_mix=0.0, rate_scale=1.0):
    """
    Generate unnormalized trajectories for all 10 reactors plus n_extra
    synthetic reactors with randomly sampled DoE conditions.

    Outputs synthetic_ode.npz and synthetic_ode.xlsx in the same directory.
    Run by Sarat to verify before using for model training.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / 'data'
    else:
        data_dir = Path(data_dir)

    if output_file is None:
        output_file = Path(__file__).parent / 'synthetic_ode.npz'

    print('Loading data_3 rates...')
    rates_growth, rates_prod, reactor_ids = load_rates(data_dir / 'data_3.csv')
    print(f'  {len(reactor_ids)} reactors: {reactor_ids}')

    print('Loading data_2 phase fractions...')
    pm_dict = load_phase_fractions(data_dir / 'data_2.csv')

    print('Loading data_1 DoE levels...')
    doe_dict = load_doe(data_dir / 'data_1.csv')

    component_names = [
        'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
        'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid',
        'L-Serine', 'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine',
        'L-Histidine', 'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine',
        'L-Tyrosine', 'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine',
        'L-Tryptophan',
    ]
    units_list = (
        ['E9 cells/L', 'g_CDW/L', 'mmol/L', 'mmol/L', 'mmol/L', 'mg/L'] +
        ['mmol/L'] * 19
    )

    trajectories = []
    phases_out   = []
    doe_params   = []

    print('\nIntegrating ODEs...')
    for reactor in reactor_ids:
        doe      = doe_dict.get(reactor, {'O2': 0, 'AAs': 0, 'Glc': 0})
        pm_days  = pm_dict.get(reactor)
        if pm_days is None:
            print(f'  WARNING: no phase data for {reactor}, skipping')
            continue

        traj, times = generate_reactor(
            reactor,
            rates_growth[reactor],
            rates_prod[reactor],
            pm_days,
            doe,
        )

        pm_vals = np.array([pm_days.get(int(t), list(pm_days.values())[-1])
                            for t in times])

        cin = make_cin(doe)
        glc_conc = cin[IDX_GLC]
        aas_conc = float(sum(cin[i] for i in AAS_INDICES))

        trajectories.append(traj)
        phases_out.append(pm_vals)
        doe_params.append([doe['O2'], glc_conc, aas_conc])

        print(f'  {reactor} (O2={doe["O2"]:+.0f} Glc={glc_conc:.1f} mmol/L '
              f'AAs={aas_conc:.2f} mmol/L):  '
              f'CD_final={traj[-1, IDX_CD]:.3f} E9/L  '
              f'Titer_final={traj[-1, IDX_TIT]:.2f} mg/L')

    trajectories = np.array(trajectories)
    phases_out   = np.array(phases_out)
    doe_params   = np.array(doe_params)

    if n_extra > 0:
        mode = 'sampled rates' if sample_rates else 'donor rates'
        print(f'\nGenerating {n_extra} extra reactors ({mode})...')
        extra_trajs, extra_doe = generate_extra(
            n_extra, rates_growth, rates_prod, reactor_ids, pm_dict, doe_dict,
            sample_rates=sample_rates, rate_mix=rate_mix, rate_scale=rate_scale)
        # Dummy phases for extras: repeat last phase value from first reactor
        extra_phases = np.tile(phases_out[0], (n_extra, 1))
        trajectories = np.concatenate([trajectories, extra_trajs], axis=0)
        doe_params   = np.concatenate([doe_params, extra_doe], axis=0)
        phases_out   = np.concatenate([phases_out, extra_phases], axis=0)
        print(f'Total reactors: {len(trajectories)}')

    n_original = len(reactor_ids)   # number of real-data reactors (always 10)

    times_out = np.tile(T_EVAL, (len(trajectories), 1))

    print('\nBuilding sliding windows (extra reactors only; originals reserved for eval)...')
    windows, targets, window_doe, reactor_idx = build_windows(
        trajectories[n_original:], doe_params=doe_params[n_original:])
    print(f'  {len(windows)} windows  '
          f'({len(trajectories) - n_original} extra reactors x {N_DAYS - SEQ_LEN} windows each)')

    doe_min = doe_params[n_original:].min(axis=0)
    doe_max = doe_params[n_original:].max(axis=0)

    np.savez(
        output_file,
        trajectories=trajectories,
        times=times_out,
        ics=trajectories[:, 0, :],
        phases=phases_out,
        doe_params=doe_params,
        doe_min=doe_min,
        doe_max=doe_max,
        components=np.array(component_names, dtype=object),
        units=np.array(units_list, dtype=object),
        windows=windows,
        targets=targets,
        window_doe=window_doe,
        window_reactor_idx=reactor_idx,
        n_original=np.array(n_original),
    )
    print(f'\nSaved {trajectories.shape} array + {len(windows)} raw windows (unnormalized)')
    print(f'Output: {output_file}')

    # Excel export -- one sheet per reactor
    xlsx_file = Path(str(output_file).replace('.npz', '.xlsx'))
    col_names = [f'{c} ({u})' for c, u in zip(component_names, units_list)]
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        for i, reactor in enumerate(reactor_ids):
            df = pd.DataFrame(trajectories[i], columns=col_names)
            df.insert(0, 'Day',  np.arange(len(df)))
            df.insert(1, 'O2',   doe_params[i, 0])
            df.insert(2, 'AAs',  doe_params[i, 1])
            df.insert(3, 'Glc',  doe_params[i, 2])
            df.to_excel(writer, sheet_name=reactor, index=False)
    print(f'Output: {xlsx_file}')

    print('\nNominal IC (t=0) used for all reactors:')
    for i, (name, unit, val) in enumerate(zip(component_names, units_list, C_NOMINAL)):
        print(f'  [{i:2d}] {name:<20} {val:.4f}  {unit}')

    return trajectories, times_out, phases_out, doe_params, component_names


def plot_comparison(trajectories, reactor_ids, component_names, data_dir=None):
    """
    Normalize synthetic trajectories the same way as data_2 (per-reactor
    per-component divide by max) and overlay with the real normalized data.

    Plots 5 key components. Saves comparison.png next to the script.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if data_dir is None:
        data_dir = Path(__file__).parent / 'data'

    df2 = pd.read_csv(Path(data_dir) / 'data_2.csv', skiprows=1)
    df2.columns = (
        ['Vessel', 'Time', 'Phase'] +
        ['C{}'.format(i) for i in range(N_COMPONENTS)]
    )
    df2['Time'] = pd.to_numeric(df2['Time'], errors='coerce')
    df2 = df2.dropna(subset=['Time'])

    plot_indices = [IDX_CD, IDX_GLC, IDX_LAC, IDX_TIT, IDX_GLN]
    plot_labels  = ['Cell Density', 'Glucose', 'Lactate', 'Titer', 'Glutamine']

    fig, axes = plt.subplots(1, len(plot_indices), figsize=(18, 4))
    colors = plt.cm.tab10.colors

    for ax, comp_idx, label in zip(axes, plot_indices, plot_labels):
        for ri, reactor in enumerate(reactor_ids):
            color = colors[ri % len(colors)]

            syn     = trajectories[ri, :, comp_idx].astype(float)
            syn_max = syn.max()
            syn_norm = syn / syn_max if syn_max > 0 else syn
            ax.plot(T_EVAL, syn_norm, color=color, lw=1.8,
                    label=reactor if comp_idx == IDX_CD else '')

            rdf       = df2[df2['Vessel'] == reactor].sort_values('Time')
            real_norm = rdf[f'C{comp_idx}'].values.astype(float)
            real_days = rdf['Time'].values.astype(float)
            ax.plot(real_days, real_norm, color=color, lw=1.8,
                    linestyle='--', alpha=0.6)

        ax.set_title(label)
        ax.set_xlabel('Day')
        ax.set_ylim(-0.05, 1.3)
        ax.axhline(0, color='k', lw=0.5, ls=':')

    axes[0].set_ylabel('Normalized concentration')
    axes[0].legend(fontsize=7, ncol=2)

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([0], [0], color='k', lw=1.8, label='Synthetic'),
                 Line2D([0], [0], color='k', lw=1.8, ls='--', alpha=0.6,
                        label='Real (data_2)')],
        loc='upper right', fontsize=9
    )

    fig.suptitle('Synthetic vs Real: normalized trajectories', y=1.02)
    fig.tight_layout()

    out = Path(__file__).parent / 'comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Comparison plot saved: {out}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-extra', type=int, default=50,
                        help='Number of extra reactors with random DoE (default: 50)')
    parser.add_argument('--sample-rates', action='store_true',
                        help='Sample ALL rates from multivariate Gaussian')
    parser.add_argument('--rate-mix', type=float, default=0.0,
                        help='Fraction of reactors with sampled rates (0.5 = 50%% sampled, 50%% donor)')
    parser.add_argument('--rate-scale', type=float, default=1.0,
                        help='Scale factor on covariance (0.25 = tighter sampling)')
    args = parser.parse_args()

    trajs, times, phases, doe_params, component_names = generate_all(
        n_extra=args.n_extra, sample_rates=args.sample_rates,
        rate_mix=args.rate_mix, rate_scale=args.rate_scale)
    reactor_ids = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
                   'R0006', 'R0008', 'R0010', 'R0011', 'R0012']
    plot_comparison(trajs[:10], reactor_ids, component_names)
