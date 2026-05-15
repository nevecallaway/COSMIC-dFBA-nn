#!/usr/bin/env python3
"""
Generate synthetic training data to augment small real dataset.
Uses phase-based dynamics model.
"""

import numpy as np
from pathlib import Path


def sigmoid(x, L=1, k=2, x0=6.5):
    """Smooth sigmoid transition."""
    return L / (1 + np.exp(-k * (x - x0)))


def generate_synthetic_trajectory(n_timepoints=13, seed=None):
    """
    Generate one synthetic trajectory with phase-dependent dynamics.
    
    Model:
    - Growth phase (p < 0.2): Cell density increases, glucose consumed, lactate produced
    - Transition phase (0.2 <= p <= 0.8): Mixed dynamics
    - Production phase (p > 0.8): Titer increases, lactate consumed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random IC
    cd0 = np.random.uniform(0.02, 0.25)  # Cell density
    glc0 = np.random.uniform(0.25, 1.0)  # Glucose
    lac0 = np.random.uniform(0.05, 0.3)  # Lactate
    titer0 = np.random.uniform(0.01, 0.05)  # Titer
    
    ic = np.array([cd0, glc0, lac0, titer0])
    
    # Time points
    time = np.linspace(0, 13, n_timepoints)
    
    # Phase schedule (sigmoid from 0 to 1)
    phase = sigmoid(time, L=1, k=0.8, x0=6.5)
    
    # Dynamics rates
    trajectory = []
    state = ic.copy()
    
    for t_idx, t in enumerate(time):
        p = phase[t_idx]
        
        # Growth phase dynamics (p < 0.2)
        if p < 0.2:
            dcd_dt = 0.15 * state[0] * (1 - state[0])  # Logistic growth
            dglc_dt = -0.3 * state[0]  # Glucose consumption
            dlac_dt = 0.1 * state[0]  # Lactate production
            dtiter_dt = 0.01 * state[0]  # Minimal titer
        
        # Transition phase (0.2 <= p <= 0.8)
        elif p <= 0.8:
            alpha = (p - 0.2) / 0.6  # Blend factor
            dcd_dt = (0.15 * (1 - alpha) + 0.02 * alpha) * state[0] * (1 - state[0])
            dglc_dt = (-0.3 * (1 - alpha) - 0.1 * alpha) * state[0]
            dlac_dt = (0.1 * (1 - alpha) + 0.05 * alpha) * state[0]
            dtiter_dt = (0.01 * (1 - alpha) + 0.15 * alpha) * state[0]
        
        # Production phase (p > 0.8)
        else:
            dcd_dt = 0.02 * state[0] * (1 - state[0])  # Slow growth
            dglc_dt = -0.1 * state[0]  # Slow consumption
            dlac_dt = -0.2 * state[1]  # Lactate consumption (product recycling)
            dtiter_dt = 0.2 * state[0]  # Strong titer production
        
        # Update state (simple Euler)
        dt = time[1] - time[0] if t_idx < len(time) - 1 else time[-1] - time[-2]
        state = state + np.array([dcd_dt, dglc_dt, dlac_dt, dtiter_dt]) * dt
        
        # Bounds
        state = np.clip(state, 0.0, 1.0)
        trajectory.append(state.copy())
    
    trajectory = np.array(trajectory)
    
    return trajectory, time, ic, phase


def generate_dataset(n_samples=50, n_timepoints=13, output_file='synthetic_training.npz'):
    """Generate synthetic dataset."""
    
    print(f"\nGenerating {n_samples} synthetic trajectories...")
    
    trajectories = []
    times = []
    ics = []
    phases = []
    
    for i in range(n_samples):
        trajectory, time, ic, phase = generate_synthetic_trajectory(n_timepoints, seed=i)
        trajectories.append(trajectory)
        times.append(time)
        ics.append(ic)
        phases.append(phase)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_samples}")
    
    trajectories = np.array(trajectories)
    times = np.array(times)
    ics = np.array(ics)
    phases = np.array(phases)
    
    # Save
    np.savez(output_file,
             trajectories=trajectories,
             times=times,
             ics=ics,
             phases=phases)
    
    print(f"\n✓ Saved to {output_file}")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  ICs shape: {ics.shape}")
    print(f"  Phases shape: {phases.shape}")
    
    return trajectories, times, ics, phases


def load_synthetic_dataset(filename='synthetic_training.npz'):
    """Load synthetic dataset."""
    data = np.load(filename)
    return data['trajectories'], data['times'], data['ics'], data['phases']


if __name__ == "__main__":
    trajectories, times, ics, phases = generate_dataset(n_samples=100, n_timepoints=13)
    
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"Cell Density:  mean={trajectories[..., 0].mean():.3f}, std={trajectories[..., 0].std():.3f}")
    print(f"Glucose:       mean={trajectories[..., 1].mean():.3f}, std={trajectories[..., 1].std():.3f}")
    print(f"Lactate:       mean={trajectories[..., 2].mean():.3f}, std={trajectories[..., 2].std():.3f}")
    print(f"Titer:         mean={trajectories[..., 3].mean():.3f}, std={trajectories[..., 3].std():.3f}")
