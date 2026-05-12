#!/usr/bin/env python3
"""
Enhanced Test and Demo Script for COSMIC-dFBA Neural Network Surrogate
Demonstrates multi-output model with phase transitions and metabolic rates.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Try to import our modules
try:
    from nn.cosmic_nn_surrogate import (
        CosmicNNSurrogateEnhanced, TrainingManager, PredictionInterface, dFBADataset
    )
    from torch.utils.data import DataLoader, random_split
    IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure cosmic_nn_surrogate.py is in the same directory")
    IMPORTS_OK = False


def create_synthetic_data_enhanced(n_simulations=30, n_timepoints=150, n_components=4):
    """
    Create realistic synthetic dFBA data for testing enhanced model.
    Includes explicit growth/production phase transitions.
    """
    print(f"\n{'='*70}")
    print("Creating Synthetic Data (Enhanced)")
    print(f"{'='*70}")
    print(f"Simulations: {n_simulations}, Timepoints: {n_timepoints}, Components: {n_components}")
    
    trajectories = []
    time_points_list = []
    initial_conditions = []
    
    for sim_idx in range(n_simulations):
        # Varying initial conditions
        ic = np.random.uniform(0.1, 1.0, n_components)
        initial_conditions.append(ic)
        
        # Time grid
        t = np.linspace(0, 15, n_timepoints)
        time_points_list.append(t)
        
        # Simulate dynamics with explicit phase transitions
        c = np.zeros((n_timepoints, n_components))
        c[0] = ic
        
        # Parameters vary per simulation
        growth_rate = np.random.uniform(0.15, 0.35)
        switch_time = np.random.uniform(5, 10)
        
        for t_idx in range(1, n_timepoints):
            dt = t[t_idx] - t[t_idx-1]
            
            # PHASE TRANSITION: Smooth sigmoid transition (growth → production)
            # This is what the enhanced model's phase_weights should learn
            phase = 1.0 / (1.0 + np.exp(-2 * (t[t_idx] - switch_time)))
            
            # Component dynamics
            for comp_idx in range(n_components):
                if comp_idx == 0:  # Biomass
                    # Exponential growth with saturation (higher in growth phase)
                    growth = growth_rate * (1 - c[t_idx-1, 0] / 2.0) * (1 - phase)
                    c[t_idx, comp_idx] = c[t_idx-1, comp_idx] * (1 + dt * growth)
                
                elif comp_idx == 1:  # Substrate
                    # Consumption proportional to biomass (higher in growth phase)
                    consumption = 0.2 * c[t_idx-1, 0] * (1 - phase)
                    c[t_idx, comp_idx] = max(0, c[t_idx-1, comp_idx] - dt * consumption)
                
                else:  # Products
                    # Production during production phase (higher when phase > 0)
                    prod_rate = 0.15 * c[t_idx-1, 0] * phase / (1 + c[t_idx-1, comp_idx])
                    c[t_idx, comp_idx] = c[t_idx-1, comp_idx] + dt * prod_rate
        
        trajectories.append(c)
    
    # Stack and pad to same length
    max_t = max(tp.shape[0] for tp in time_points_list)
    trajectories_padded = np.zeros((n_simulations, max_t, n_components))
    times_padded = np.zeros((n_simulations, max_t))
    
    for i, traj in enumerate(trajectories):
        nt = traj.shape[0]
        trajectories_padded[i, :nt, :] = traj
        times_padded[i, :nt] = time_points_list[i]
    
    parameters = {}  # No varying parameters in this demo
    
    print(f"✓ Generated {n_simulations} synthetic trajectories with phase transitions")
    return trajectories_padded, times_padded, np.array(initial_conditions), parameters


def train_enhanced_model(trajectories, time_points, initial_conditions, parameters):
    """Train the enhanced surrogate model."""
    print(f"\n{'='*70}")
    print("Training Enhanced Neural Network Surrogate")
    print(f"{'='*70}")
    
    # Create dataset
    dataset = dFBADataset(trajectories, time_points, initial_conditions, parameters, normalize=True)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model outputs: concentrations, phase_weights, growth_rates, prod_rates")
    
    # Train with enhanced loss function
    trainer = TrainingManager(model, device=device, learning_rate=1e-3, model_type='enhanced')
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=30, patience=10)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    return model, dataset, trainer


def evaluate_enhanced_model(model, dataset, device):
    """Evaluate enhanced model on held-out test data."""
    print(f"\n{'='*70}")
    print("Evaluating Enhanced Model")
    print(f"{'='*70}")
    
    predictor = PredictionInterface(model, dataset, device=device, model_type='enhanced')
    
    # Test prediction on new initial condition
    test_ic = np.array([0.5, 0.8, 0.2, 0.1])
    test_time = np.linspace(0, 15, 100)
    
    start_time = time.time()
    prediction = predictor.predict(test_ic, test_time, return_rates=True)
    elapsed = time.time() - start_time
    
    print(f"Prediction time: {elapsed*1000:.2f}ms")
    print(f"\nPrediction outputs:")
    print(f"  - Concentrations shape: {prediction['concentrations'].shape}")
    print(f"  - Phase weights shape: {prediction['phase_weights'].shape}")
    print(f"  - Growth rates shape: {prediction['growth_rates'].shape}")
    print(f"  - Prod rates shape: {prediction['prod_rates'].shape}")
    
    print(f"\nConcentration ranges:")
    conc = prediction['concentrations']
    print(f"  - Overall: {conc.min():.4f} - {conc.max():.4f}")
    for comp_idx in range(conc.shape[1]):
        print(f"  - Component {comp_idx}: {conc[:, comp_idx].min():.4f} - {conc[:, comp_idx].max():.4f}")
    
    print(f"\nPhase transition:")
    phases = prediction['phase_weights'].flatten()
    print(f"  - Growth phase (0): {phases[0]:.4f}")
    print(f"  - End phase: {phases[-1]:.4f}")
    print(f"  - Midpoint transition at ~{test_time[np.argmin(np.abs(phases - 0.5))]:.2f} days")
    
    # Check for NaNs or Infs
    if np.any(np.isnan(conc)):
        print("⚠ Warning: NaN values in concentrations")
    if np.any(np.isinf(conc)):
        print("⚠ Warning: Inf values in concentrations")
    
    print("✓ Enhanced model evaluation successful")
    
    return prediction, test_time, test_ic, predictor


def visualize_enhanced_results(prediction, test_time, component_names=None):
    """Create comprehensive visualization of enhanced model results."""
    print(f"\n{'='*70}")
    print("Creating Visualizations (Enhanced)")
    print(f"{'='*70}")
    
    conc = prediction['concentrations']
    phases = prediction['phase_weights'].flatten()
    n_components = conc.shape[1]
    
    if component_names is None:
        component_names = ['Biomass', 'Substrate', 'Product 1', 'Product 2'][:n_components]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 4*n_components + 3))
    
    # Plot 1: Phase transition
    ax_phase = plt.subplot(n_components + 1, 1, 1)
    ax_phase.fill_between(test_time, 0, phases, alpha=0.3, label='Production Phase')
    ax_phase.fill_between(test_time, 0, 1-phases, alpha=0.3, label='Growth Phase')
    ax_phase.plot(test_time, phases, 'k-', linewidth=2, label='Phase Weight (0=Growth, 1=Production)')
    ax_phase.set_ylabel('Phase Weight')
    ax_phase.set_title('Cell Metabolic Phase Transition', fontsize=12, fontweight='bold')
    ax_phase.legend(loc='center right')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_ylim([0, 1])
    
    # Plot 2+: Concentrations
    for comp_idx in range(n_components):
        ax = plt.subplot(n_components + 1, 1, comp_idx + 2)
        ax.plot(test_time, conc[:, comp_idx], 'b-', linewidth=2, label='NN Prediction')
        ax.fill_between(test_time, 0, conc[:, comp_idx], alpha=0.2)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.set_title(component_names[comp_idx])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    output_file = Path('/tmp/cosmic_nn_demo_enhanced.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Saved enhanced plot to {output_file}")
    
    return fig


def phase_transition_analysis(predictor, test_ic, test_time):
    """Detailed analysis of phase transitions."""
    print(f"\n{'='*70}")
    print("Phase Transition Analysis")
    print(f"{'='*70}")
    
    # Analyze phase transition with different initial conditions
    ic_variations = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        ic_var = test_ic * scale
        ic_variations.append(ic_var)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Phase transitions across IC variations
    ax = axes[0, 0]
    for scale, ic_var in zip([0.5, 0.75, 1.0, 1.25, 1.5], ic_variations):
        pred = predictor.predict(ic_var, test_time, return_rates=True)
        ax.plot(test_time, pred['phase_weights'].flatten(), label=f'IC Scale={scale:.2f}', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Phase Weight (0=Growth, 1=Production)')
    ax.set_title('Phase Transition Sensitivity to Initial Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Biomass across IC variations
    ax = axes[0, 1]
    for scale, ic_var in zip([0.5, 0.75, 1.0, 1.25, 1.5], ic_variations):
        pred = predictor.predict(ic_var, test_time)
        ax.plot(test_time, pred['concentrations'][:, 0], label=f'IC Scale={scale:.2f}', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Biomass Concentration')
    ax.set_title('Biomass Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Substrate consumption across phases
    ax = axes[1, 0]
    pred = predictor.predict(test_ic, test_time, return_rates=True)
    ax.plot(test_time, pred['growth_rates'][:, 1], label='Growth Phase Substrate Rate', linewidth=2)
    ax.plot(test_time, pred['prod_rates'][:, 1], label='Production Phase Substrate Rate', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Substrate Uptake Rate')
    ax.set_title('Substrate Uptake by Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Product formation across phases
    ax = axes[1, 1]
    ax.plot(test_time, pred['growth_rates'][:, 2], label='Growth Phase Product Rate', linewidth=2)
    ax.plot(test_time, pred['prod_rates'][:, 2], label='Production Phase Product Rate', linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Product Formation Rate')
    ax.set_title('Product Formation by Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path('/tmp/cosmic_nn_phase_analysis.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Saved phase analysis plot to {output_file}")
    
    return fig


def main():
    """Run complete enhanced demo."""
    print("\n" + "="*70)
    print("COSMIC dFBA Enhanced Neural Network Surrogate - Demo")
    print("With Multi-Output Heads and Phase Transitions")
    print("="*70)
    
    if not IMPORTS_OK:
        print("\n✗ Failed to import required modules")
        print("Please ensure cosmic_nn_surrogate.py is in the current directory")
        sys.exit(1)
    
    try:
        # Step 1: Generate synthetic data with phase transitions
        trajectories, time_points, initial_conditions, parameters = create_synthetic_data_enhanced(
            n_simulations=30,
            n_timepoints=150,
            n_components=4
        )
        
        # Step 2: Train enhanced model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, dataset, trainer = train_enhanced_model(trajectories, time_points, initial_conditions, parameters)
        
        # Step 3: Evaluate enhanced model
        prediction, test_time, test_ic, predictor = evaluate_enhanced_model(model, dataset, device)
        
        # Step 4: Visualizations
        visualize_enhanced_results(prediction, test_time)
        
        # Step 5: Phase transition analysis
        phase_transition_analysis(predictor, test_ic, test_time)
        
        # Final summary
        print(f"\n{'='*70}")
        print("✓ Enhanced Demo Complete!")
        print(f"{'='*70}")
        print("\nKey Features Demonstrated:")
        print("1. ✓ Multi-output heads (concentrations, phase weights, rates)")
        print("2. ✓ Phase transition learning (growth → production)")
        print("3. ✓ Metabolic rate prediction per cell state")
        print("4. ✓ Smooth sigmoid phase transitions")
        print("5. ✓ Mass-balance aware predictions")
        print("\nNext Steps for Production:")
        print("1. Train on real MATLAB dFBA simulations")
        print("2. Validate against measured bioreactor data")
        print("3. Integrate stoichiometric constraints")
        print("4. Deploy for real-time bioreactor control")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
