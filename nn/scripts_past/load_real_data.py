#!/usr/bin/env python3
"""
Real data loader for COSMIC-dFBA from experimental CSV files.
Parses Gopalakrishnan et al. (2024) supplementary data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List


def load_experimental_data(data_file: str = "data_2.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load real bioprocess data from CSV.
    
    Args:
        data_file: Path to data_2.csv with experimental measurements
        
    Returns:
        trajectories: (n_reactors, n_timepoints, n_components)
        time_points: (n_reactors, n_timepoints)
        initial_conditions: (n_reactors, n_components)
        metadata: Dict with column info
    """
    
    # Read CSV
    df = pd.read_csv(data_file)
    
    # Convert numeric columns to float (skip unit rows)
    numeric_cols = [col for col in df.columns if col not in ['Vessel']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove header rows with units
    df = df.dropna(subset=['Time'])
    
    # Key metabolites to track (matching paper Figure S2 and S3)
    key_components = [
        'Cell Density',      # Index 0: Biomass proxy
        'Glucose',           # Index 1: Primary substrate
        'Lactate',           # Index 2: Byproduct
        'Titer',             # Index 3: Product (antibody)
    ]
    
    # Optional: add amino acids if needed
    optional_aa = ['Glutamine', 'Glutamate', 'L-Asparagine']  # Top varying AAs per paper
    
    # Get unique reactors
    reactors = df['Vessel'].unique()
    reactors = sorted([r for r in reactors if pd.notna(r)])  # Remove NaN
    
    print(f"\nLoading real experimental data from {data_file}")
    print(f"Found {len(reactors)} reactors: {reactors}")
    
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
    
    metadata = {
        'components': key_components,
        'n_components': n_components,
        'n_reactors': n_reactors,
        'reactors': reactors,
        'phases': phases_padded,  # Ground truth phase information
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
    
    Args:
        phases: Phase fractions array (n_reactors, n_timepoints)
        threshold_growth: Upper bound for growth phase
        threshold_prod: Lower bound for production phase
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
        
        # Find phase boundaries
        growth_mask = phase_traj < threshold_growth
        prod_mask = phase_traj > threshold_prod
        trans_mask = (phase_traj >= threshold_growth) & (phase_traj <= threshold_prod)
        
        # Find transition timepoint
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
    Compare real data characteristics with our synthetic data generation.
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


def main():
    """Load and visualize real experimental data."""
    
    # Try to load from different possible locations
    possible_paths = [
        Path("data_2.csv"),
        Path("nn/data_2.csv"),
        Path("/Users/nevecallaway/Downloads/data_2.csv"),
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = str(p)
            break
    
    if data_file is None:
        print("Error: data_2.csv not found. Please provide the file path.")
        print(f"Searched: {possible_paths}")
        return
    
    # Load data
    trajectories, times, ics, metadata = load_experimental_data(data_file)
    
    # Analyze phases
    analyze_phase_transitions(metadata['phases'])
    
    # Compare with synthetic
    compare_with_synthetic()
    
    print(f"\n✓ Real data ready for training!")
    print(f"  Use: trajectories={trajectories.shape}, times={times.shape}, ics={ics.shape}")


if __name__ == "__main__":
    main()
