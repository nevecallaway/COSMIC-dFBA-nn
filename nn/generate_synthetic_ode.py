#!/usr/bin/env python3
"""
Physics-based synthetic data generator for COSMIC-dFBA surrogate.

Implements Supplementary Methods Equation 2 from Gopalakrishnan et al.:

    dC_i/dt = F * (Cin_i - eta_i * C_i) + v_i(t) * C_1        [Eq. 2]

    v_i(t) = (1 - pm(t)) * v_growth_i + pm(t) * v_stat_i      [Eq. 1]

Inputs:
    data_3.csv  -- growth and production phase specific rates per reactor
    data_2.csv  -- phase fraction pm (column C) per reactor per day
    data_1.csv  -- DoE coded levels per reactor (O2, AAs, Glc)
    DMEM/F12    -- base media concentrations hardcoded below (mM)

Output:
    synthetic_ode.npz -- unnormalized trajectories in physical units

Units:
    Cell Density : 10^6 cells/mL
    Cell Volume  : nL/mL (total biovolume concentration, see NOTE below)
    Glucose, AAs : mM
    Lactate, NH4 : mM
    Titer        : mg/mL

NOTE on Cell Volume: initialized as total biovolume concentration (nL/mL).
Calculation: d_mean = (14.02 + 15.21)/2 = 14.615 um (Harvard BioNumbers for
CHO), V_cell = (4/3)*pi*(7.3075)^3 = 1634.5 um^3 = 1.6345e-6 nL/cell.
At seeding density 0.5e6 cells/mL: C_V0 = 817.3 nL/mL.
The v_V rate from data_3 (units 1/day) is treated as nL/(1e6 cells * day)
so that v_V * C_D [1e6 cells/mL] = dC_V/dt [nL/mL/day]. Confirm with Sarat.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F      = 1.0    # perfusion rate (bioreactor volumes per day)
DAY8   = 8.0    # day when titer washout activates
N_DAYS = 13     # total days (day 0 through day 12)
T_EVAL = np.arange(0, N_DAYS, dtype=float)

# ---------------------------------------------------------------------------
# Component indices -- match data_2 / data_3 column order
# ---------------------------------------------------------------------------
IDX_CD  = 0   # Cell Density   (10^6 cells/mL)
IDX_CV  = 1   # Cell Volume    (dimensionless index)
IDX_GLC = 2   # Glucose        (mM)
IDX_LAC = 3   # Lactate        (mM)
IDX_NH4 = 4   # NH4            (mM)
IDX_TIT = 5   # Titer          (mg/mL)
IDX_GLN = 6   # Glutamine
IDX_GLU = 7   # Glutamate      (secreted)
IDX_ASN = 8   # L-Asparagine
IDX_ASP = 9   # L-Aspartic acid (secreted)
IDX_SER = 10  # L-Serine
IDX_GLY = 11  # Glycine         (secreted)
IDX_ALA = 12  # L-Alanine       (secreted)
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
AAS_INDICES  = list(range(6, N_COMPONENTS))  # all amino acid columns

# ---------------------------------------------------------------------------
# DMEM/F12 base media -- nominal (DoE level 0) concentrations
# Used as both IC and nominal Cin for the perfusion feed
# ---------------------------------------------------------------------------
C_NOMINAL = np.array([
    0.5,     # Cell Density  (10^6 cells/mL, Sarat anchor)
    817.3,   # Cell Volume   (nL/mL: 1634.5 um^3/cell * 0.5e6 cells/mL, see NOTE)
    17.5,    # Glucose       (mM)
    0.0,     # Lactate       (mM, starts at 0 per meeting notes)
    0.0,     # NH4           (mM, starts at 0)
    0.0,     # Titer         (mg/mL, starts at 0)
    2.5,     # Glutamine     (mM)
    0.05,    # Glutamate     (mM, secreted -- DMEM/F12 value)
    0.05,    # L-Asparagine  (mM)
    0.05,    # L-Aspartic acid (mM, secreted)
    0.25,    # L-Serine      (mM)
    0.25,    # Glycine       (mM, secreted)
    0.05,    # L-Alanine     (mM, secreted)
    0.15,    # L-Proline     (mM)
    0.45,    # L-Threonine   (mM)
    0.15,    # L-Histidine   (mM)
    0.50,    # L-Lysine      (mM)
    0.45,    # L-Valine      (mM)
    0.12,    # L-Methionine  (mM)
    0.70,    # L-Arginine    (mM)
    0.20,    # L-Tyrosine    (mM)
    0.42,    # L-Isoleucine  (mM)
    0.45,    # L-Leucine     (mM)
    0.21,    # L-Phenylalanine (mM)
    0.044,   # L-Tryptophan  (mM)
], dtype=float)

# Perfusion feed: same as nominal but no cells, no lactate, no NH4, no titer
CIN_NOMINAL = C_NOMINAL.copy()
CIN_NOMINAL[IDX_CD]  = 0.0
CIN_NOMINAL[IDX_CV]  = 0.0
CIN_NOMINAL[IDX_LAC] = 0.0
CIN_NOMINAL[IDX_NH4] = 0.0
CIN_NOMINAL[IDX_TIT] = 0.0

# eta base: 0 for cells and titer (before day 8), 1 for everything else
ETA_BASE = np.ones(N_COMPONENTS)
ETA_BASE[IDX_CD]  = 0.0
ETA_BASE[IDX_CV]  = 0.0
ETA_BASE[IDX_TIT] = 0.0   # updated to 1 after day 8 in the ODE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rates(data3_path):
    """
    Load growth and production phase specific rates from data_3.csv.

    Returns:
        rates_growth : dict {reactor_id: np.array shape (N_COMPONENTS,)}
        rates_prod   : dict {reactor_id: np.array shape (N_COMPONENTS,)}
        reactor_ids  : list of reactor ID strings
    """
    df = pd.read_csv(data3_path, header=None)

    # Row 1 (index 1): header with reactor IDs starting at column 2
    reactor_ids = [str(df.iloc[1, c]).strip() for c in range(2, 12)]

    rates_growth = {r: np.zeros(N_COMPONENTS) for r in reactor_ids}
    rates_prod   = {r: np.zeros(N_COMPONENTS) for r in reactor_ids}

    # Rows 2+ are components in data_2 order
    for comp_idx in range(N_COMPONENTS):
        row = df.iloc[2 + comp_idx]
        for ri, reactor in enumerate(reactor_ids):
            rates_growth[reactor][comp_idx] = float(row.iloc[2 + ri])
            rates_prod[reactor][comp_idx]   = float(row.iloc[12 + ri])

    return rates_growth, rates_prod, reactor_ids


def load_phase_fractions(data2_path):
    """
    Load pm (production phase fraction) per reactor per day from data_2.csv.

    Returns:
        pm_dict : dict {reactor_id: np.array shape (N_DAYS,)}
        days    : np.array shape (N_DAYS,)
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
        rdf  = df[df['Vessel'] == reactor].sort_values('Time')
        days = rdf['Time'].values.astype(float)
        pm   = rdf['Phase'].values.astype(float)
        # Interpolator: linear, clamped to [0, 1] outside observed range
        pm_dict[str(reactor)] = interp1d(
            days, pm, kind='linear',
            bounds_error=False,
            fill_value=(pm[0], pm[-1])
        )

    return pm_dict


def load_doe(data1_path):
    """
    Load DoE coded levels (-1, 0, +1) per reactor from data_1.csv.

    Returns:
        doe_dict : dict {reactor_id: {'O2': int, 'AAs': int, 'Glc': int}}
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

    DoE convention (meeting notes):
        level -1 -> half the nominal concentration
        level  0 -> nominal
        level +1 -> double the nominal concentration

    DoE AAs factor applies to all amino acid indices (6-24).
    DoE Glc factor applies to glucose index (2) only.
    Cells, lactate, NH4, titer are not in the feed (already 0 in CIN_NOMINAL).
    """
    cin = CIN_NOMINAL.copy()

    # Glucose scaling
    glc_factor = 2.0 ** doe['Glc']   # -1 -> 0.5x, 0 -> 1x, +1 -> 2x
    cin[IDX_GLC] *= glc_factor

    # Amino acid scaling (all AAs including secreted)
    aas_factor = 2.0 ** doe['AAs']
    cin[AAS_INDICES] *= aas_factor

    return cin


# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------

def make_ode(v_growth, v_prod, pm_func, cin, eta_base):
    """
    Return the ODE function for one reactor.

    State vector C has shape (N_COMPONENTS,) in physical units.
    """
    def ode(t, C):
        pm    = float(np.clip(pm_func(t), 0.0, 1.0))
        v     = (1.0 - pm) * v_growth + pm * v_prod   # blended rates

        # eta: titer washout activates at day 8
        eta      = eta_base.copy()
        eta[IDX_TIT] = 1.0 if t >= DAY8 else 0.0

        C_D   = max(C[IDX_CD], 0.0)   # cell density, floor at 0

        dC    = np.zeros(N_COMPONENTS)
        for i in range(N_COMPONENTS):
            washout    = F * (cin[i] - eta[i] * max(C[i], 0.0))
            metabolic  = v[i] * C_D
            # if metabolite is depleted, block further consumption
            if C[i] <= 0.0 and metabolic < 0.0:
                metabolic = 0.0
            dC[i]      = washout + metabolic

        return dC

    return ode


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_reactor(reactor_id, v_growth, v_prod, pm_func, doe,
                     t_eval=T_EVAL):
    """
    Integrate the ODE for one reactor and return the trajectory.

    Returns:
        trajectory : np.array shape (N_DAYS, N_COMPONENTS) in physical units
        times      : np.array shape (N_DAYS,)
    """
    cin = make_cin(doe)
    ode = make_ode(v_growth, v_prod, pm_func, cin, ETA_BASE.copy())

    sol = solve_ivp(
        ode,
        t_span=(t_eval[0], t_eval[-1]),
        y0=C_NOMINAL.copy(),
        method='RK45',
        t_eval=t_eval,
        max_step=0.1,        # max 0.1 day step to handle stiffness
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        print(f'  WARNING: solver failed for {reactor_id}: {sol.message}')

    return sol.y.T, sol.t   # (N_DAYS, N_COMPONENTS), (N_DAYS,)


def generate_all(data_dir=None, output_file=None):
    """
    Generate unnormalized trajectories for all 10 reactors using data_3
    rates, data_2 phase fractions, and DMEM/F12 initial conditions.

    This reproduces the structure of data_2 (Table 2) in physical units.
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

    trajectories = []
    phases_out   = []
    doe_params   = []

    component_names = [
        'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
        'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid',
        'L-Serine', 'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine',
        'L-Histidine', 'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine',
        'L-Tyrosine', 'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine',
        'L-Tryptophan',
    ]

    print('\nIntegrating ODEs...')
    for reactor in reactor_ids:
        doe = doe_dict.get(reactor, {'O2': 0, 'AAs': 0, 'Glc': 0})
        pm_func = pm_dict.get(reactor)
        if pm_func is None:
            print(f'  WARNING: no phase data for {reactor}, skipping')
            continue

        traj, times = generate_reactor(
            reactor,
            rates_growth[reactor],
            rates_prod[reactor],
            pm_func,
            doe,
        )

        # Phase fraction at eval points for reference
        pm_vals = np.array([pm_func(t) for t in times])

        trajectories.append(traj)
        phases_out.append(pm_vals)
        doe_params.append([doe['O2'], doe['AAs'], doe['Glc']])

        cd_final  = traj[-1, IDX_CD]
        tit_final = traj[-1, IDX_TIT]
        print(f'  {reactor} (O2={doe["O2"]:+.0f} AAs={doe["AAs"]:+.0f} '
              f'Glc={doe["Glc"]:+.0f}):  '
              f'CD_final={cd_final:.3f} Titer_final={tit_final:.4f}')

    trajectories = np.array(trajectories)   # (n_reactors, N_DAYS, N_COMPONENTS)
    phases_out   = np.array(phases_out)     # (n_reactors, N_DAYS)
    doe_params   = np.array(doe_params)     # (n_reactors, 3)
    times_out    = np.tile(T_EVAL, (len(trajectories), 1))

    units_list = (
        ['1e6 cells/mL', 'nL/mL', 'mM', 'mM', 'mM', 'mg/mL'] + ['mM'] * 19
    )

    np.savez(
        output_file,
        trajectories=trajectories,
        times=times_out,
        ics=trajectories[:, 0, :],
        phases=phases_out,
        doe_params=doe_params,
        components=np.array(component_names, dtype=object),
        units=np.array(units_list, dtype=object),
    )
    print(f'\nSaved {trajectories.shape[0]} reactors x {trajectories.shape[1]} '
          f'timepoints x {trajectories.shape[2]} components')
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
    for i, (name, val) in enumerate(zip(component_names, C_NOMINAL)):
        print(f'  [{i:2d}] {name:<20} {val:.4f}')

    return trajectories, times_out, phases_out, doe_params, component_names


if __name__ == '__main__':
    generate_all()
