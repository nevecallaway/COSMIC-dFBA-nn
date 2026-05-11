#!/usr/bin/env python3
"""
Test and Demo Script for COSMIC dFBA Neural Network Surrogate
Runs a complete end-to-end example with synthetic data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Try to import our modules
try:
    from cosmic_nn_surrogate import (
        CosmicNNSurrogate, TrainingManager, PredictionInterface, dFBADataset
    )
    from torch.utils.data import DataLoader, random_split
    IMPORTS_OK = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure cosmic_nn_surrogate.py is in the same directory")
    IMPORTS_OK = False


def create_synthetic_data(n_simulations=30, n_timepoints=150, n_components=4):
    """Create realistic synthetic dFBA data for testing."""
    print(f"\n{'='*70}")
    print("Creating Synthetic Data")
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
        
        # Simulate dynamics
        c = np.zeros((n_timepoints, n_components))
        c[0] = ic
        
        # Parameters vary per simulation
        growth_rate = np.random.uniform(0.15, 0.35)
        switch_time = np.random.uniform(5, 10)
        
        for t_idx in range(1, n_timepoints):
            dt = t[t_idx] - t[t_idx-1]
            
            # Phase transition (growth → production)
            phase = 1.0 / (1.0 + np.exp(-2 * (t[t_idx] - switch_time)))
            
            # Component dynamics
            for comp_idx in range(n_components):
                if comp_idx == 0:  # Biomass
                    # Exponential growth with saturation
                    growth = growth_rate * (1 - c[t_idx-1, 0] / 2.0) * (1 - phase)
                    c[t_idx, comp_idx] = c[t_idx-1, comp_idx] * (1 + dt * growth)
                
                elif comp_idx == 1:  # Substrate
                    # Consumption proportional to biomass
                    consumption = 0.2 * c[t_idx-1, 0] * (1 - phase)
                    c[t_idx, comp_idx] = max(0, c[t_idx-1, comp_idx] - dt * consumption)
                
                else:  # Products
                    # Production during production phase
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
    
    print(f"✓ Generated {n_simulations} synthetic trajectories")
    return trajectories_padded, times_padded, np.array(initial_conditions), parameters


def train_model(trajectories, time_points, initial_conditions, parameters):
    """Train the surrogate model."""
    print(f"\n{'='*70}")
    print("Training Neural Network Surrogate")
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
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = CosmicNNSurrogate(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=32,
        n_heads=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = TrainingManager(model, device=device, learning_rate=1e-3)
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=30, patience=10)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    
    return model, dataset, trainer


def evaluate_model(model, dataset, device):
    """Evaluate model on held-out test data."""
    print(f"\n{'='*70}")
    print("Evaluating Model")
    print(f"{'='*70}")
    
    predictor = PredictionInterface(model, dataset, device=device)
    
    # Test prediction on new initial condition
    test_ic = np.array([0.5, 0.8, 0.2, 0.1])
    test_time = np.linspace(0, 15, 100)
    
    start_time = time.time()
    prediction = predictor.predict(test_ic, test_time)
    elapsed = time.time() - start_time
    
    print(f"Prediction time: {elapsed*1000:.2f}ms")
    print(f"Output shape: {prediction.shape}")
    print(f"Concentration ranges: {prediction.min():.4f} - {prediction.max():.4f}")
    
    # Check for NaNs or Infs
    if np.any(np.isnan(prediction)):
        print("⚠ Warning: NaN values in prediction")
    if np.any(np.isinf(prediction)):
        print("⚠ Warning: Inf values in prediction")
    
    print("✓ Model evaluation successful")
    
    return prediction, test_time, test_ic, predictor


def visualize_results(prediction, test_time, component_names=None):
    """Create visualization of results."""
    print(f"\n{'='*70}")
    print("Creating Visualizations")
    print(f"{'='*70}")
    
    n_components = prediction.shape[1]
    if component_names is None:
        component_names = ['Biomass', 'Substrate', 'Product 1', 'Product 2'][:n_components]
    
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
    if n_components == 1:
        axes = [axes]
    
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        ax.plot(test_time, prediction[:, comp_idx], 'b-', linewidth=2, label='NN Prediction')
        ax.fill_between(test_time, 0, prediction[:, comp_idx], alpha=0.2)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.set_title(component_names[comp_idx])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    output_file = Path('/tmp/cosmic_nn_demo.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")
    
    return fig


def sensitivity_test(predictor, test_ic, test_time):
    """Test sensitivity analysis."""
    print(f"\n{'='*70}")
    print("Sensitivity Analysis")
    print(f"{'='*70}")
    
    # This would normally vary parameters, but we don't have any in this demo
    # So instead, we'll test with different initial conditions
    
    ic_variations = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        ic_var = test_ic * scale
        ic_variations.append(ic_var)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for comp_idx in range(min(4, test_ic.shape[0])):
        ax = axes[comp_idx // 2, comp_idx % 2]
        
        for scale, ic_var in zip([0.5, 0.75, 1.0, 1.25, 1.5], ic_variations):
            pred = predictor.predict(ic_var, test_time)
            ax.plot(test_time, pred[:, comp_idx], label=f'Scale={scale:.2f}', linewidth=2)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.set_title(f'Component {comp_idx} - Sensitivity to Initial Condition')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path('/tmp/cosmic_nn_sensitivity.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Saved sensitivity plot to {output_file}")
    
    return fig


def batch_predictions_test(predictor, test_time, n_batch=10):
    """Test batch predictions."""
    print(f"\n{'='*70}")
    print("Batch Predictions Test")
    print(f"{'='*70}")
    
    # Create random batch of initial conditions
    batch_ics = np.random.uniform(0.1, 1.0, (n_batch, 4))
    
    start_time = time.time()
    batch_preds = predictor.predict(batch_ics, test_time)
    elapsed = time.time() - start_time
    
    print(f"Batch predictions: {n_batch} simulations in {elapsed*1000:.2f}ms")
    print(f"Average per sim: {elapsed/n_batch*1000:.2f}ms")
    print(f"Output shape: {batch_preds.shape}")
    print("✓ Batch prediction successful")


def main():
    """Run complete demo."""
    print("\n" + "="*70)
    print("COSMIC dFBA Neural Network Surrogate - Demo")
    print("="*70)
    
    if not IMPORTS_OK:
        print("\n✗ Failed to import required modules")
        print("Please ensure cosmic_nn_surrogate.py is in the current directory")
        sys.exit(1)
    
    try:
        # Step 1: Generate synthetic data
        trajectories, time_points, initial_conditions, parameters = create_synthetic_data(
            n_simulations=30,
            n_timepoints=150,
            n_components=4
        )
        
        # Step 2: Train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, dataset, trainer = train_model(trajectories, time_points, initial_conditions, parameters)
        
        # Step 3: Evaluate model
        prediction, test_time, test_ic, predictor = evaluate_model(model, dataset, device)
        
        # Step 4: Visualizations
        visualize_results(prediction, test_time)
        
        # Step 5: Sensitivity analysis
        sensitivity_test(predictor, test_ic, test_time)
        
        # Step 6: Batch predictions
        batch_predictions_test(predictor, test_time, n_batch=20)
        
        # Final summary
        print(f"\n{'='*70}")
        print("✓ Demo Complete!")
        print(f"{'='*70}")
        print("\nNext steps:")
        print("1. Export your MATLAB dFBA simulations to .npz format")
        print("2. Load with: MultiSimulationDataset(['sim1.npz', 'sim2.npz', ...])")
        print("3. Train model with the same pipeline")
        print("4. Make predictions for new conditions/parameters")
        print("\nSee integration_guide.py and README.md for full documentation")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
