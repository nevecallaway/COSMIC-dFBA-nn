"""
Integration Guide: Converting MATLAB dFBA to Neural Network
===========================================================

This script demonstrates:
1. Exporting MATLAB simulations for NN training
2. Training the surrogate model
3. Making predictions with new parameters
4. Analyzing model sensitivity
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from cosmic_nn_surrogate import (
    CosmicNNSurrogate, TrainingManager, PredictionInterface, dFBADataset
)
from torch.utils.data import DataLoader, random_split
import json
from pathlib import Path


# ============================================================================
# STEP 1: EXPORT MATLAB SIMULATIONS
# ============================================================================
def export_matlab_simulation_to_npz(time, profiles, flux_growth, flux_prod, 
                                     phase_transition, parameters, output_file):
    """
    Export MATLAB dFBA results to .npz format for NN training.
    
    Call this from MATLAB after running dFBA_results:
    
    MATLAB code to add:
    ```matlab
    % After dFBA simulation completes, export data:
    save([datestr(now,'yyyymmdd_HHMMSS'), '_dFBA_sim.mat'], ...
        'dFBA_results', 'model', 'dFBA_data');
    
    % Then in Python, load and convert
    ```
    """
    np.savez(output_file,
             time=time,
             profiles=profiles,
             flux_growth=flux_growth,
             flux_prod=flux_prod,
             phase_transition=phase_transition,
             **parameters)  # Store all parameters as separate arrays
    
    print(f"Exported to {output_file}")


def import_matlab_mat_file(mat_file):
    """
    Load MATLAB .mat file and extract dFBA results.
    
    Requires: scipy.io.loadmat
    """
    from scipy.io import loadmat
    
    mat = loadmat(mat_file)
    dFBA_results = mat['dFBA_results'][0, 0]
    
    # Extract arrays (MATLAB stores as cell arrays)
    time = dFBA_results['time'].flatten()
    profiles = dFBA_results['profiles']
    flux_growth = dFBA_results['flux_growth']
    flux_prod = dFBA_results['flux_prod']
    phase_transition = dFBA_results['phase_transition'].flatten()
    
    return {
        'time': time,
        'profiles': profiles,
        'flux_growth': flux_growth,
        'flux_prod': flux_prod,
        'phase_transition': phase_transition,
    }


# ============================================================================
# STEP 2: CREATE TRAINING DATA
# ============================================================================
def create_synthetic_training_data(n_simulations=50, n_timepoints=200, n_components=5):
    """
    Create synthetic training data (useful for testing).
    In practice, this would come from multiple MATLAB dFBA simulations.
    """
    trajectories = []
    time_points_list = []
    initial_conditions = []
    
    # Simulate varying kinetic parameters
    kinetic_growth_vm_list = []
    kinetic_prod_vm_list = []
    
    for i in range(n_simulations):
        # Random initial conditions
        ic = np.random.uniform(0.1, 1.0, n_components)
        initial_conditions.append(ic)
        
        # Time grid
        t = np.linspace(0, 10, n_timepoints)
        time_points_list.append(t)
        
        # Random kinetic parameters
        vm_growth = np.random.uniform(0.5, 2.0, n_components)
        vm_prod = np.random.uniform(0.5, 2.0, n_components)
        
        kinetic_growth_vm_list.append(vm_growth)
        kinetic_prod_vm_list.append(vm_prod)
        
        # Simulate dynamics: simple exponential growth + production phase transition
        c = np.zeros((n_timepoints, n_components))
        c[0] = ic
        
        for t_idx in range(1, n_timepoints):
            # Phase transition (switch from growth to production)
            phase = np.tanh((t[t_idx] - 5) / 2)  # Smooth 0->1 transition
            
            # Growth phase: exponential
            growth_rate = 0.2 * (1 - phase)
            
            # Production phase: substrate conversion
            prod_rate = 0.1 * phase
            
            for comp_idx in range(n_components):
                # Simple kinetics
                if comp_idx == 0:  # Biomass
                    c[t_idx, comp_idx] = c[t_idx-1, comp_idx] * np.exp(growth_rate)
                elif comp_idx == 1:  # Substrate
                    decay = vm_growth[comp_idx] * c[t_idx-1, 1] / (1 + c[t_idx-1, 1])
                    c[t_idx, comp_idx] = max(0, c[t_idx-1, comp_idx] - decay)
                else:  # Products
                    production = vm_prod[comp_idx] * c[t_idx-1, 0] * phase / (1 + c[t_idx-1, comp_idx])
                    c[t_idx, comp_idx] = c[t_idx-1, comp_idx] + production
        
        trajectories.append(c)
    
    # Stack and pad to same length
    max_t = max(tp.shape[0] for tp in time_points_list)
    trajectories_padded = np.zeros((n_simulations, max_t, n_components))
    times_padded = np.zeros((n_simulations, max_t))
    
    for i, traj in enumerate(trajectories):
        nt = traj.shape[0]
        trajectories_padded[i, :nt, :] = traj
        times_padded[i, :nt] = time_points_list[i]
    
    parameters = {
        'kinetic_vm_growth': np.array(kinetic_growth_vm_list),
        'kinetic_vm_prod': np.array(kinetic_prod_vm_list),
    }
    
    return trajectories_padded, times_padded, np.array(initial_conditions), parameters


# ============================================================================
# STEP 3: TRAIN THE NEURAL NETWORK
# ============================================================================
def train_surrogate_model(trajectories, time_points, initial_conditions, parameters,
                         output_model_path='cosmic_surrogate_model.pt'):
    """
    Train the neural network surrogate model.
    """
    print("Creating dataset...")
    dataset = dFBADataset(trajectories, time_points, initial_conditions, parameters, normalize=True)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Components: {dataset.n_components}, Time points: {dataset.n_timepoints}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    n_params = sum(v.shape[0] if v.ndim > 1 else 1 for v in parameters.values())
    
    model = CosmicNNSurrogate(
        n_components=dataset.n_components,
        n_params=n_params,
        latent_dim=64,
        n_heads=4
    )
    
    # Train
    trainer = TrainingManager(model, device=device, learning_rate=1e-3)
    best_val_loss = trainer.train(train_loader, val_loader, epochs=50, patience=15)
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    # Save
    trainer.save(output_model_path)
    
    return model, dataset, trainer


# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
def make_predictions(model_path, dataset, initial_conditions, time_points, 
                     parameters=None):
    """
    Load trained model and make predictions for new conditions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine n_params from dataset
    n_params = sum(v.shape[0] if v.ndim > 1 else 1 for v in dataset.parameters.values())
    
    model = CosmicNNSurrogate(
        n_components=dataset.n_components,
        n_params=n_params,
        latent_dim=64,
        n_heads=4
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    predictor = PredictionInterface(model, dataset, device=device)
    
    # Make prediction
    prediction = predictor.predict(initial_conditions, time_points, parameters)
    
    return prediction, predictor


# ============================================================================
# STEP 5: ANALYZE RESULTS
# ============================================================================
def plot_predictions_vs_data(time_data, data_profiles, predicted_profiles, 
                             component_names=None, save_path=None):
    """
    Compare actual dFBA simulations with NN predictions.
    """
    n_components = data_profiles.shape[1]
    if component_names is None:
        component_names = [f'Component {i}' for i in range(n_components)]
    
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
    if n_components == 1:
        axes = [axes]
    
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        
        # Actual data
        ax.plot(time_data, data_profiles[:, comp_idx], 'o-', label='MATLAB dFBA', linewidth=2, markersize=4)
        
        # Prediction
        ax.plot(time_data, predicted_profiles[:, comp_idx], 's--', label='NN Surrogate', linewidth=2, markersize=4)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.set_title(component_names[comp_idx])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def sensitivity_analysis_plot(predictor, initial_conditions, time_points,
                             param_name, param_range, component_idx=0, n_points=5):
    """
    Visualize how output changes with a parameter.
    """
    param_values, predictions = predictor.sensitivity_analysis(
        initial_conditions, time_points, param_name, param_range, n_points=n_points
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    for i, (pval, pred) in enumerate(zip(param_values, predictions)):
        ax.plot(time_points, pred[:, component_idx], 
               label=f'{param_name}={pval:.3f}', color=colors[i], linewidth=2)
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Component Concentration')
    ax.set_title(f'Sensitivity to {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXAMPLE
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("COSMIC dFBA Neural Network Surrogate - Integration Example")
    print("=" * 70)
    
    # Create synthetic training data (replace with real MATLAB data)
    print("\n[Step 1] Creating synthetic training data...")
    trajectories, time_points, initial_conditions, parameters = create_synthetic_training_data(
        n_simulations=50,
        n_timepoints=200,
        n_components=5
    )
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Parameters: {list(parameters.keys())}")
    
    # Train model
    print("\n[Step 2] Training surrogate model...")
    model, dataset, trainer = train_surrogate_model(
        trajectories, time_points, initial_conditions, parameters,
        output_model_path='/tmp/cosmic_surrogate_model.pt'
    )
    
    # Make predictions on held-out data
    print("\n[Step 3] Making predictions on new conditions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = PredictionInterface(model, dataset, device=device)
    
    # Test with new initial conditions
    test_ic = np.array([0.5, 0.8, 0.2, 0.1, 0.0])
    test_time = np.linspace(0, 10, 100)
    test_params = {'kinetic_vm_growth': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                   'kinetic_vm_prod': np.array([0.5, 0.5, 0.5, 0.5, 0.5])}
    
    prediction = predictor.predict(test_ic, test_time, test_params)
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Min/Max values: {prediction.min():.4f} / {prediction.max():.4f}")
    
    # Plot results
    print("\n[Step 4] Generating visualizations...")
    plot_predictions_vs_data(
        test_time, 
        prediction,
        prediction,  # Compare with itself for demo
        component_names=['Biomass', 'Glucose', 'Product 1', 'Product 2', 'Byproduct']
    )
    
    # Sensitivity analysis
    print("\n[Step 5] Performing sensitivity analysis...")
    sensitivity_analysis_plot(
        predictor, test_ic, test_time,
        param_name='kinetic_vm_growth',
        param_range=(np.array([0.5, 0.5, 0.5, 0.5, 0.5]), 
                    np.array([2.0, 2.0, 2.0, 2.0, 2.0])),
        component_idx=0,
        n_points=5
    )
    
    print("\n" + "=" * 70)
    print("Training complete! Model saved to: /tmp/cosmic_surrogate_model.pt")
    print("=" * 70)
