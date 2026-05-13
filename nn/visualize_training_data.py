#!/usr/bin/env python3
"""
Visualize raw training data: trajectories by reactor, component, and phase.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from load_real_data import load_experimental_data
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def main():
    """Visualize training data."""
    
    print(f"\n{'='*70}")
    print("TRAINING DATA VISUALIZATION")
    print(f"{'='*70}")
    
    # Load data
    possible_paths = [
        Path("data_2.csv"),
        Path("/content/COSMIC-dFBA-nn/nn/data_2.csv"),
        Path("/Users/nevecallaway/Downloads/data_2.csv"),
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = str(p)
            break
    
    if data_file is None:
        print("Error: data_2.csv not found")
        sys.exit(1)
    
    trajectories, time_points, ics, metadata = load_experimental_data(data_file)
    phases = metadata['phases']
    reactor_ids = list(metadata['reactors'])
    component_names = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    print(f"\nVisualizing {len(reactor_ids)} reactors, {trajectories.shape[1]} timepoints, 4 components\n")
    
    # Figure 1: All reactors, all components
    fig, axes = plt.subplots(4, 10, figsize=(20, 12))
    fig.suptitle('Raw Training Data: All Reactors × All Components', fontsize=16, fontweight='bold')
    
    for comp_idx, comp_name in enumerate(component_names):
        for reactor_idx, reactor_id in enumerate(reactor_ids):
            ax = axes[comp_idx, reactor_idx]
            
            time_axis = np.arange(trajectories.shape[1])
            traj = trajectories[reactor_idx, :, comp_idx]
            phase = phases[reactor_idx]
            
            # Color by phase
            scatter = ax.scatter(time_axis, traj, c=phase, cmap='RdYlGn', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.plot(time_axis, traj, 'k-', alpha=0.3, linewidth=1)
            
            ax.set_title(f'{reactor_id}', fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)
            
            if reactor_idx == 0:
                ax.set_ylabel(comp_name, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            if comp_idx == 3:
                ax.set_xlabel('Time')
            else:
                ax.set_xlabel('')
    
    plt.colorbar(scatter, ax=axes, label='Phase (0=Growth, 1=Production)', shrink=0.8)
    plt.tight_layout()
    plt.savefig('training_data_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: training_data_trajectories.png")
    plt.close()
    
    # Figure 2: Component-wise distribution across all reactors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Component Distributions Across All Reactors', fontsize=16, fontweight='bold')
    
    for comp_idx, comp_name in enumerate(component_names):
        ax = axes[comp_idx // 2, comp_idx % 2]
        
        for reactor_idx, reactor_id in enumerate(reactor_ids):
            time_axis = np.arange(trajectories.shape[1])
            traj = trajectories[reactor_idx, :, comp_idx]
            ax.plot(time_axis, traj, 'o-', label=reactor_id, alpha=0.7, markersize=4)
        
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{comp_name}')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig('component_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: component_distributions.png")
    plt.close()
    
    # Figure 3: Phase-annotated trajectories for key components
    fig, axes = plt.subplots(2, 10, figsize=(24, 6))
    fig.suptitle('Phase-Annotated Trajectories: Cell Density & Titer', fontsize=16, fontweight='bold')
    
    for reactor_idx, reactor_id in enumerate(reactor_ids):
        # Cell Density
        ax = axes[0, reactor_idx]
        time_axis = np.arange(trajectories.shape[1])
        phase = phases[reactor_idx]
        cd = trajectories[reactor_idx, :, 0]
        
        scatter = ax.scatter(time_axis, cd, c=phase, cmap='RdYlGn', s=80, alpha=0.8, edgecolors='black', linewidth=1)
        ax.plot(time_axis, cd, 'k-', alpha=0.3, linewidth=1)
        ax.axhline(0.2, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(0.8, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f'{reactor_id}', fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([0, 6, 12])
        
        if reactor_idx == 0:
            ax.set_ylabel('Cell Density', fontweight='bold')
        
        # Titer
        ax = axes[1, reactor_idx]
        titer = trajectories[reactor_idx, :, 3]
        
        scatter = ax.scatter(time_axis, titer, c=phase, cmap='RdYlGn', s=80, alpha=0.8, edgecolors='black', linewidth=1)
        ax.plot(time_axis, titer, 'k-', alpha=0.3, linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Time', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([0, 6, 12])
        
        if reactor_idx == 0:
            ax.set_ylabel('Titer', fontweight='bold')
    
    plt.colorbar(scatter, ax=axes, label='Phase', shrink=0.8)
    plt.tight_layout()
    plt.savefig('phase_annotated_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: phase_annotated_trajectories.png")
    plt.close()
    
    # Figure 4: Initial conditions scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Initial Conditions Across Reactors', fontsize=16, fontweight='bold')
    
    for i in range(3):
        for j in range(2):
            comp1_idx = i + j * 2
            comp2_idx = (i + 1) % 4 + j * 2
            
            if comp1_idx >= 4 or comp2_idx >= 4:
                continue
            
            ax = axes[j, i]
            
            scatter = ax.scatter(ics[:, comp1_idx], ics[:, comp2_idx], s=150, alpha=0.6, edgecolors='black', linewidth=1)
            
            for reactor_idx, reactor_id in enumerate(reactor_ids):
                ax.annotate(reactor_id, (ics[reactor_idx, comp1_idx], ics[reactor_idx, comp2_idx]), 
                           fontsize=8, ha='center', va='center')
            
            ax.set_xlabel(component_names[comp1_idx])
            ax.set_ylabel(component_names[comp2_idx])
            ax.set_title(f'{component_names[comp1_idx]} vs {component_names[comp2_idx]}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('initial_conditions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: initial_conditions.png")
    plt.close()
    
    # Figure 5: Phase distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Phase Distribution Analysis', fontsize=16, fontweight='bold')
    
    # All phases histogram
    ax = axes[0, 0]
    all_phases = phases.flatten()
    ax.hist(all_phases, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0.2, color='green', linestyle='--', linewidth=2, label='Growth/Transition')
    ax.axvline(0.8, color='orange', linestyle='--', linewidth=2, label='Transition/Production')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Count')
    ax.set_title('Overall Phase Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-reactor phase range
    ax = axes[0, 1]
    phase_ranges = [(p.min(), p.max()) for p in phases]
    phase_mins = [r[0] for r in phase_ranges]
    phase_maxs = [r[1] for r in phase_ranges]
    ax.errorbar(range(len(reactor_ids)), phase_maxs, 
               yerr=[np.array(phase_maxs) - np.array(phase_mins), np.zeros(len(reactor_ids))],
               fmt='o', capsize=5, alpha=0.7)
    ax.set_xticks(range(len(reactor_ids)))
    ax.set_xticklabels(reactor_ids, rotation=45, ha='right')
    ax.set_ylabel('Phase Range')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Phase Range by Reactor')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Phase vs Titer
    ax = axes[1, 0]
    for reactor_idx, reactor_id in enumerate(reactor_ids):
        ax.plot(phases[reactor_idx], trajectories[reactor_idx, :, 3], 'o-', label=reactor_id, alpha=0.6)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Titer')
    ax.set_title('Phase vs Titer (Product)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Phase vs Cell Density
    ax = axes[1, 1]
    for reactor_idx, reactor_id in enumerate(reactor_ids):
        ax.plot(phases[reactor_idx], trajectories[reactor_idx, :, 0], 'o-', label=reactor_id, alpha=0.6)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Cell Density')
    ax.set_title('Phase vs Cell Density (Biomass)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: phase_analysis.png")
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DATA STATISTICS")
    print(f"{'='*70}\n")
    
    print("Phase Statistics:")
    print(f"  Overall phase range: [{all_phases.min():.3f}, {all_phases.max():.3f}]")
    print(f"  Mean phase per reactor: {[f'{p.mean():.3f}' for p in phases]}")
    print(f"\n  Growth phase (<0.2):      {(all_phases < 0.2).sum()} points ({100*(all_phases < 0.2).sum()/len(all_phases):.1f}%)")
    print(f"  Transition phase (0.2-0.8): {((all_phases >= 0.2) & (all_phases <= 0.8)).sum()} points ({100*((all_phases >= 0.2) & (all_phases <= 0.8)).sum()/len(all_phases):.1f}%)")
    print(f"  Production phase (>0.8):  {(all_phases > 0.8).sum()} points ({100*(all_phases > 0.8).sum()/len(all_phases):.1f}%)")
    
    print(f"\nComponent Statistics:")
    for comp_idx, comp_name in enumerate(component_names):
        comp_data = trajectories[:, :, comp_idx].flatten()
        print(f"  {comp_name}: mean={comp_data.mean():.4f}, std={comp_data.std():.4f}, "
              f"range=[{comp_data.min():.4f}, {comp_data.max():.4f}]")
    
    print(f"\n✓ Data visualization complete!")


if __name__ == "__main__":
    main()
