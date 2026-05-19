#!/usr/bin/env python3
"""
Utilities for COSMIC-dFBA: Data loading, experimental analysis, and model diagnostics.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Any
from sklearn.metrics import f1_score, confusion_matrix, r2_score, mean_absolute_percentage_error

def load_doe_parameters(doe_file: str) -> Dict[str, np.ndarray]:
    """
    Load DoE variable levels from data_1.csv.

    The file has a merged "Variable Levels" header in row 0 and the real
    column names (Vessel, O2, AAs, Glc) in row 1, so we read with header=1.

    Returns dict mapping reactor name → np.array([O2, AAs, Glc]) with
    values in {-1, 0, +1}.  Footer/NaN rows are silently dropped.
    """
    df = pd.read_csv(doe_file, header=1)
    # Drop footer rows (NaN vessel or non-reactor strings like the footnote)
    df = df[df['Vessel'].str.match(r'^R\d{4}$', na=False)].copy()
    for col in ['O2', 'AAs', 'Glc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    result = {}
    for _, row in df.iterrows():
        result[str(row['Vessel'])] = np.array([row['O2'], row['AAs'], row['Glc']], dtype=float)
    return result


def load_specific_rates(rates_file: str, reactors: list) -> np.ndarray:
    """
    Load per-reactor phase-specific metabolic rates from data_3.csv.

    Returns array of shape (n_reactors, 50) — 25 growth-phase rates
    followed by 25 production-phase rates — standardised to N(0,1)
    per rate column across reactors so all features are comparable.

    Reactor order matches the `reactors` list passed in (must be sorted).
    """
    df = pd.read_csv(rates_file)
    # Row 0 contains reactor names; rows 1+ are the 25 component rates.
    # Columns 2-11  → growth phase (R0001..R0012 order)
    # Columns 12-21 → production phase (R0001..R0012 order)
    data = df.iloc[1:].reset_index(drop=True)
    # The column order in data_3 is the same sorted reactor order
    data3_reactors = ['R0001','R0002','R0003','R0004','R0005',
                      'R0006','R0008','R0010','R0011','R0012']
    n_r  = len(reactors)
    n_c  = 25   # components
    growth = np.zeros((n_r, n_c))
    prod   = np.zeros((n_r, n_c))
    for i, reactor in enumerate(reactors):
        if reactor in data3_reactors:
            j = data3_reactors.index(reactor)
            growth[i] = data.iloc[:, 2 + j].values.astype(float)
            prod[i]   = data.iloc[:, 12 + j].values.astype(float)

    # Standardise each column to N(0,1) across reactors
    rates = np.concatenate([growth, prod], axis=1)   # (n_r, 50)
    mean  = rates.mean(axis=0, keepdims=True)
    std   = np.clip(rates.std(axis=0, keepdims=True), 1e-6, None)
    return (rates - mean) / std


def load_experimental_data(data_file: str = "data/data_2.csv",
                           doe_file: str = "data/data_1.csv",
                           rates_file: str = "data/data_3.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load real bioprocess data from CSV.

    Args:
        data_file:   Path to data_2.csv with experimental measurements
        doe_file:    Path to data_1.csv with DoE variable levels (O2, AAs, Glc).
        rates_file:  Path to data_3.csv with per-reactor phase-specific rates.
                     Missing files are handled gracefully (params omitted).

    Returns:
        trajectories: (n_reactors, n_timepoints, n_components)
        time_points: (n_reactors, n_timepoints)
        initial_conditions: (n_reactors, n_components)
        metadata: Dict with column info, 'doe_params' (n_reactors, 3),
                  and 'specific_rates' (n_reactors, 50)
    """
    # Read CSV
    df = pd.read_csv(data_file)

    # Convert numeric columns to float (skip unit rows)
    numeric_cols = [col for col in df.columns if col not in ['Vessel']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove header rows with units
    df = df.dropna(subset=['Time'])

    # Key metabolites to track: use all numeric columns except metadata
    excluded_cols = ['Vessel', 'Time', 'Production phase fraction']
    key_components = [col for col in df.columns if col not in excluded_cols]

    # Get unique reactors
    reactors = df['Vessel'].unique()
    reactors = sorted([r for r in reactors if pd.notna(r)])  # Remove NaN

    print(f"\nLoading real experimental data from {data_file}")
    print(f"Found {len(reactors)} reactors: {reactors}")
    print(f"Tracking {len(key_components)} metabolites: {key_components}")

    trajectories_list = []
    times_list = []
    ics_list = []
    phases_list = []  # Track phase transitions

    for reactor in reactors:
        reactor_data = df[df['Vessel'] == reactor].copy()
        reactor_data = reactor_data.sort_values('Time')

        # Extract time points
        times = reactor_data['Time'].values

        # Extract component trajectories
        trajectory = []
        for comp in key_components:
            if comp in reactor_data.columns:
                traj = reactor_data[comp].values
                trajectory.append(traj)
            else:
                print(f"  Warning: {comp} not found in data for {reactor}")
                trajectory.append(np.zeros_like(times))

        trajectory = np.array(trajectory).T  # Shape: (n_timepoints, n_components)

        # Extract phase information (production phase fraction)
        if 'Production phase fraction' in reactor_data.columns:
            phases = reactor_data['Production phase fraction'].values
        else:
            phases = np.zeros_like(times)

        # Initial conditions (first timepoint)
        ic = trajectory[0, :]

        trajectories_list.append(trajectory)
        times_list.append(times)
        ics_list.append(ic)
        phases_list.append(phases)

        print(f"  ✓ {reactor}: {len(times)} timepoints, IC={ic}, Phase range: {phases.min():.3f}-{phases.max():.3f}")

    # Pad to same length
    max_t = max(len(t) for t in times_list)
    n_reactors = len(reactors)
    n_components = trajectory.shape[1]

    trajectories_padded = np.zeros((n_reactors, max_t, n_components))
    times_padded = np.zeros((n_reactors, max_t))
    phases_padded = np.zeros((n_reactors, max_t))

    for i, (traj, times, phases) in enumerate(zip(trajectories_list, times_list, phases_list)):
        nt = len(times)
        trajectories_padded[i, :nt, :] = traj
        times_padded[i, :nt] = times
        phases_padded[i, :nt] = phases

        # Pad with last values
        if nt < max_t:
            trajectories_padded[i, nt:, :] = traj[-1, :]
            times_padded[i, nt:] = times[-1]
            phases_padded[i, nt:] = phases[-1]

    initial_conditions = np.array(ics_list)

    # Load DoE parameters (O2, AAs, Glc) if file exists
    doe_params_array = None
    try:
        doe_map = load_doe_parameters(doe_file)
        doe_params_array = np.array([
            doe_map.get(r, np.zeros(3)) for r in reactors
        ], dtype=float)  # (n_reactors, 3)
        print(f"  DoE params loaded: {doe_params_array.shape} "
              f"[O2, AAs, Glc] from {doe_file}")
    except FileNotFoundError:
        print(f"  (DoE file {doe_file} not found — running without process parameters)")

    # Load phase-specific metabolic rates (25 growth + 25 production) if file exists
    specific_rates_array = None
    try:
        specific_rates_array = load_specific_rates(rates_file, reactors)  # (n_reactors, 50)
        print(f"  Specific rates loaded: {specific_rates_array.shape} "
              f"[25 growth + 25 prod rates] from {rates_file}")
    except FileNotFoundError:
        print(f"  (Rates file {rates_file} not found — running without specific rates)")

    metadata = {
        'components': key_components,
        'n_components': n_components,
        'n_reactors': n_reactors,
        'reactors': reactors,
        'phases': phases_padded,              # Ground truth phase information
        'doe_params': doe_params_array,        # (n_reactors, 3) or None
        'specific_rates': specific_rates_array, # (n_reactors, 50) or None
    }

    print(f"\n✓ Loaded real data:")
    print(f"  Trajectories shape: {trajectories_padded.shape}")
    print(f"  Components: {key_components}")
    print(f"  Time range: {times_padded.min():.4f} - {times_padded.max():.2f} days")
    print(f"  Phase range: {phases_padded.min():.3f} - {phases_padded.max():.3f}")

    return trajectories_padded, times_padded, initial_conditions, metadata


def analyze_phase_transitions(phases: np.ndarray, threshold_growth=0.2, threshold_prod=0.8):
    """
    Analyze bistable phase transitions from ground truth data.
    """
    print(f"\n{'='*70}")
    print("Phase Transition Analysis")
    print(f"{'='*70}")
    print(f"Growth phase: p_m < {threshold_growth}")
    print(f"Transition zone: {threshold_growth} <= p_m <= {threshold_prod}")
    print(f"Production phase: p_m > {threshold_prod}")

    n_reactors, n_timepoints = phases.shape

    for i in range(n_reactors):
        phase_traj = phases[i, :]

        growth_mask = phase_traj < threshold_growth
        prod_mask = phase_traj > threshold_prod
        trans_mask = (phase_traj >= threshold_growth) & (phase_traj <= threshold_prod)

        if np.any(growth_mask) and np.any(prod_mask):
            last_growth = np.where(growth_mask)[0][-1]
            first_prod = np.where(prod_mask)[0][0]

            if first_prod > last_growth:
                print(f"\nReactor {i}:")
                print(f"  Growth phase: 0 → {last_growth} timepoints")
                print(f"  Transition: {last_growth} → {first_prod} ({np.sum(trans_mask)} steps)")
                print(f"  Production phase: {first_prod} → end")
                print(f"  Phase trajectory: {phase_traj[:5]} ... {phase_traj[-5:]}")


def compare_with_synthetic():
    """
    Compare real data characteristics with synthetic data generation.
    """
    print(f"\n{'='*70}")
    print("Real vs Synthetic Data Comparison")
    print(f"{'='*70}")

    comparison = """
    REAL DATA (Gopalakrishnan et al.):
    ✓ Bistable: p_m in {0, 0.5, 1.0} mostly (not continuous)
    ✓ Sharp transitions: Jump from growth (f<0.2) to production (f>0.8) in 1-2 days
    ✓ Substrate-limited: Glucose consumed during growth, then leveled off
    ✓ Product accumulation: Titer increases throughout, but faster in production phase
    ✓ Cell density plateaus: Saturation in production phase
    ✓ Metabolic rewiring: AA uptake/secretion patterns change at transition

    SYNTHETIC DATA (current):
    × Smooth sigmoid: p_m transitions continuously (0→1)
    × Gradual shifts: Phase transition spans multiple days
    × Artificial: Not matching real bistable behavior

    WHAT WE NEED:
    1. Fit real phases to hard thresholds (0 or 1 mostly)
    2. Model sharp transitions as step functions, not sigmoids
    3. Train model to predict discontinuous phase switches
    4. Validate on real data: Can NN learn bistability from smooth input?
    """
    print(comparison)


class ModelDiagnostics:
    """
    Advanced diagnostic suite for COSMIC-dFBA surrogate models.
    """

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # r2_score requires ≥2 samples; return nan gracefully for single-reactor LOO folds
        flat_true, flat_pred = y_true.flatten(), y_pred.flatten()
        r2   = r2_score(flat_true, flat_pred) if len(flat_true) >= 2 else float('nan')
        mape = mean_absolute_percentage_error(flat_true, flat_pred)

        comp_r2 = {}
        for i in range(y_true.shape[-1]):
            t, p = y_true[..., i].flatten(), y_pred[..., i].flatten()
            comp_r2[f"comp_{i}"] = r2_score(t, p) if (len(t) >= 2 and t.var() > 0) else float('nan')

        return {
            "global_r2": r2,
            "global_mape": mape,
            "component_r2": comp_r2
        }

    @staticmethod
    def calculate_phase_metrics(y_true_phase: np.ndarray, y_pred_phase_prob: np.ndarray) -> Dict[str, Any]:
        # f < 0.2 → growth state (0), f > 0.8 → production state (1).
        # Evaluate only on unambiguous timepoints; ignore the 0.2–0.8 transition zone.
        y_true_flat = y_true_phase.flatten()
        y_pred_flat = y_pred_phase_prob.squeeze(-1).flatten()
        mask = (y_true_flat < 0.2) | (y_true_flat > 0.8)
        y_true_binary = (y_true_flat[mask] > 0.5).astype(int)
        y_pred_binary = (y_pred_flat[mask] > 0.5).astype(int)
        f1 = f1_score(y_true_binary, y_pred_binary)
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        return {
            "phase_f1": f1,
            "confusion_matrix": cm
        }

    @staticmethod
    def analyze_modality_dominance(model, ic, time_points, params, device='cpu'):
        model.eval()
        ic = torch.FloatTensor(ic).to(device).requires_grad_(True)
        params = torch.FloatTensor(params).to(device).requires_grad_(True)
        time_tensor = torch.FloatTensor(time_points).unsqueeze(0).to(device)
        outputs = model(ic, time_tensor, params)
        final_titer = outputs['concentrations'][0, -1, -1]
        final_titer.backward()
        ic_saliency = ic.grad.abs().mean().item()
        param_saliency = params.grad.abs().mean().item()
        return {
            "ic_importance": ic_saliency,
            "param_importance": param_saliency,
            "dominance_ratio": ic_saliency / (param_saliency + 1e-6)
        }

    @staticmethod
    def detect_drop_off_rca(concentrations: np.ndarray, time_points: np.ndarray, comp_idx: int = -1):
        c = concentrations[..., comp_idx]
        t = time_points
        dc_dt = np.diff(c, axis=-1) / np.diff(t)
        drop_off_mask = dc_dt < -0.05
        results = []
        for i in range(c.shape[0]):
            drops = np.where(drop_off_mask[i])[0]
            if len(drops) > 0:
                first_drop_t = t[drops[0]]
                decay_rate = np.mean(dc_dt[i, drops])
                results.append({"drop_start_time": first_drop_t, "avg_decay_rate": decay_rate, "is_crashing": True})
            else:
                results.append({"is_crashing": False})
        return results
