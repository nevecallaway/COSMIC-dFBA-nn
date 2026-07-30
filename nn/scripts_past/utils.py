#!/usr/bin/env python3
"""
Utilities for COSMIC-dFBA: Data loading, experimental analysis, and model diagnostics.
"""

import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Any
from sklearn.metrics import (f1_score, confusion_matrix, r2_score,
                             mean_absolute_percentage_error,
                             matthews_corrcoef)
from scipy.stats import spearmanr

def load_specific_rates(data3_file: str) -> Dict[str, np.ndarray]:
    """
    Load phase-specific metabolic rates from data_3.csv.

    data_3.csv layout:
      Row 0: phase headers (Growth Phase, ..., Production Phase, ...)
      Row 1: reactor names (R0001 ... R0012)
      Rows 2+: one row per metabolite component
      Cols 0-1 : component name, unit
      Cols 2-11: growth phase rates per reactor (10 reactors)
      Cols 12-21: production phase rates per reactor

    The values are phase-specific maxima from the dFBA model.  Actual rates
    lie somewhere between 0 and these maxima.  We use their absolute values
    to set per-component v_max ceilings on the rate heads.

    Returns dict with:
      'rates_growth'   : (n_reactors, n_components) -- physical units (see data_3 headers)
      'rates_prod'     : (n_reactors, n_components)
      'v_max_growth'   : (n_components,) -- cross-reactor mean |rate|, normalised
      'v_max_prod'     : (n_components,) -- cross-reactor mean |rate|, normalised
      'v_max_scale'    : float -- global divisor used for normalisation
                         multiply normalised values by this to recover physical units
      'reactors'       : list of reactor names in column order
      'components'     : list of component names
    """
    df = pd.read_csv(data3_file, header=None)

    reactors    = [str(df.iloc[1, c]).strip() for c in range(2, 12)]
    components  = [str(df.iloc[r, 0]).strip() for r in range(2, len(df))]
    n_reactors  = len(reactors)
    n_components = len(components)

    rates_growth = np.zeros((n_reactors, n_components))
    rates_prod   = np.zeros((n_reactors, n_components))

    for ci in range(n_components):
        row = df.iloc[2 + ci]
        for ri in range(n_reactors):
            rates_growth[ri, ci] = float(row.iloc[2 + ri])
            rates_prod[ri, ci]   = float(row.iloc[12 + ri])

    # Cross-reactor mean of absolute rates per component per phase.
    # Using the mean (not max) so outlier reactors don't dominate the ceiling.
    v_max_growth = np.abs(rates_growth).mean(axis=0)   # (n_components,)
    v_max_prod   = np.abs(rates_prod).mean(axis=0)

    # Normalise by the single largest value across both phases so the
    # highest-activity component gets v_max = 1.0 and all others scale
    # proportionally.  The scale factor is stored so callers can convert
    # back to physical units once a physical anchor is known.
    global_scale = float(max(v_max_growth.max(), v_max_prod.max()))
    v_max_growth_norm = v_max_growth / global_scale
    v_max_prod_norm   = v_max_prod   / global_scale

    return {
        'rates_growth': rates_growth,
        'rates_prod':   rates_prod,
        'v_max_growth': v_max_growth_norm,
        'v_max_prod':   v_max_prod_norm,
        'v_max_scale':  global_scale,
        'reactors':     reactors,
        'components':   components,
    }


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




def load_experimental_data(data_file: str = "data/data_2.csv",
                           doe_file: str = "data/data_1.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load real bioprocess data from CSV.

    Args:
        data_file:  Path to data_2.csv with experimental measurements
        doe_file:   Path to data_1.csv with DoE variable levels (O2, AAs, Glc).

    Returns:
        trajectories: (n_reactors, n_timepoints, n_components)
        time_points: (n_reactors, n_timepoints)
        initial_conditions: (n_reactors, n_components)
        metadata: dict with 'doe_params' (n_reactors, 3), 'phases' (n_reactors, T), etc.
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

    metadata = {
        'components': key_components,
        'n_components': n_components,
        'n_reactors': n_reactors,
        'reactors': reactors,
        'phases': phases_padded,      # Ground truth phase information
        'doe_params': doe_params_array,  # (n_reactors, 3) or None
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
    def calculate_spearman_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Within-reactor Spearman correlation across timepoints, per component.

        y_true / y_pred: (n_reactors, n_timepoints, n_components)

        For each reactor × component pair, compute Spearman rank correlation
        of the predicted vs actual trajectory over time.  Returns the mean
        across reactors for each component, plus the titer (comp 5) result.
        """
        n_reactors, n_timepoints, n_components = y_true.shape
        # (n_reactors, n_components)
        rho_matrix = np.full((n_reactors, n_components), np.nan)

        for r in range(n_reactors):
            for c in range(n_components):
                t = y_true[r, :, c]
                p = y_pred[r, :, c]
                # Need variance in both series to compute correlation
                if t.std() > 1e-8 and p.std() > 1e-8:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rho, _ = spearmanr(t, p)
                    rho_matrix[r, c] = rho

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_rho_per_comp = np.nanmean(rho_matrix, axis=0)  # (n_components,)
        comp_spearman = {f"comp_{i}": mean_rho_per_comp[i] for i in range(n_components)}

        return {
            "mean_spearman": float(np.nanmean(rho_matrix)),
            "titer_spearman": float(mean_rho_per_comp[5]),   # comp 5 = Titer
            "component_spearman": comp_spearman,
            "per_reactor_titer_spearman": {
                f"reactor_{r}": float(rho_matrix[r, 5]) for r in range(n_reactors)
            },
        }

    @staticmethod
    def calculate_phase_metrics(y_true_phase: np.ndarray, y_pred_phase_prob: np.ndarray) -> Dict[str, Any]:
        # f < 0.2 → growth state (0), f > 0.8 → production state (1).
        # Evaluate only on unambiguous timepoints; ignore the 0.2–0.8 transition zone.
        # Matches the paper's evaluation protocol (section 2.3).
        y_true_flat = y_true_phase.flatten()
        y_pred_flat = y_pred_phase_prob.squeeze(-1).flatten()
        mask = (y_true_flat < 0.2) | (y_true_flat > 0.8)
        y_true_binary = (y_true_flat[mask] > 0.5).astype(int)
        y_pred_binary = (y_pred_flat[mask] > 0.5).astype(int)

        f1  = f1_score(y_true_binary, y_pred_binary)
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        cm  = confusion_matrix(y_true_binary, y_pred_binary)

        # Specificity = TN / (TN + FP),  Sensitivity (recall) = TP / (TP + FN)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        return {
            "phase_f1":        f1,
            "mcc":             mcc,
            "specificity":     specificity,
            "sensitivity":     sensitivity,
            "confusion_matrix": cm,
        }

    @staticmethod
    def _interp_transition_time(f_traj: np.ndarray, time_points: np.ndarray,
                                threshold: float = 0.5) -> float:
        """Linearly interpolate the day when f(t) first crosses `threshold`.

        If f never reaches the threshold, returns the last timepoint (late
        transition) or first timepoint (already past threshold at t=0).
        """
        for i in range(len(f_traj) - 1):
            f0, f1 = f_traj[i], f_traj[i + 1]
            if (f0 <= threshold <= f1) or (f0 >= threshold >= f1):
                t0, t1 = time_points[i], time_points[i + 1]
                if abs(f1 - f0) < 1e-8:
                    return float(t0)
                return float(t0 + (threshold - f0) * (t1 - t0) / (f1 - f0))
        # f stayed below threshold — transition never happened
        if f_traj[-1] < threshold:
            return float(time_points[-1])
        # f was above threshold from the start — already in production
        return float(time_points[0])

    @staticmethod
    def calculate_transition_metrics(phases_true: np.ndarray,
                                     phases_pred: np.ndarray,
                                     time_points: np.ndarray) -> Dict[str, Any]:
        """Compute transition-timing and f(t) accuracy metrics.

        Matches the COSMIC paper's evaluation (section 2.3):
          - % of timepoints where |f_pred - f_true| < 0.1  (paper: 72%)
          - % of timepoints where |f_pred - f_true| < 0.2  (paper: 91%)
          - Per-reactor transition time (day f first crosses 0.5)
          - MAE of predicted vs actual transition time (in days)

        Args:
            phases_true : (N, T) or (N, T, 1) — actual f(t) per reactor
            phases_pred : (N, T, 1) — predicted f(t) per reactor
            time_points : (N, T) or (T,) — actual day values
        """
        # Normalise shapes
        if phases_true.ndim == 3:
            phases_true = phases_true.squeeze(-1)   # (N, T)
        if phases_pred.ndim == 3:
            phases_pred = phases_pred.squeeze(-1)   # (N, T)
        if time_points.ndim == 1:
            time_points = np.tile(time_points, (phases_true.shape[0], 1))

        N = phases_true.shape[0]
        true_t = np.array([
            ModelDiagnostics._interp_transition_time(phases_true[r], time_points[r])
            for r in range(N)])
        pred_t = np.array([
            ModelDiagnostics._interp_transition_time(phases_pred[r], time_points[r])
            for r in range(N)])

        errors = np.abs(pred_t - true_t)

        # Paper accuracy: % of timepoints within ±0.1 and ±0.2 of measured f
        abs_f_err = np.abs(phases_pred - phases_true)
        acc_01 = float((abs_f_err < 0.1).mean())
        acc_02 = float((abs_f_err < 0.2).mean())

        return {
            'transition_mae_days':        float(errors.mean()),
            'transition_std_days':        float(errors.std()),
            'transition_errors_per_reactor': errors,      # (N,) in days
            'true_transition_days':       true_t,
            'pred_transition_days':       pred_t,
            'f_accuracy_01':              acc_01,   # paper metric (±0.1)
            'f_accuracy_02':              acc_02,   # paper metric (±0.2)
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
