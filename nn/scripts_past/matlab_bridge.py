"""
MATLAB-to-Python Bridge for COSMIC dFBA Neural Network
=======================================================

This provides instructions and utilities for:
1. Exporting MATLAB simulation data
2. Training PyTorch neural networks
3. Making predictions for new conditions
"""

import numpy as np
from scipy.io import loadmat, savemat
import json
from pathlib import Path
import torch


# ============================================================================
# MATLAB EXPORT TEMPLATE
# ============================================================================
MATLAB_EXPORT_CODE = """
% Add this to the end of your COSMIC_dFBA function to export data for NN training

% After dFBA_results are computed, export to Python-compatible format:
export_path = fullfile(pwd, ['dFBA_', datestr(now,'yyyymmdd_HHMMSS'), '.npz']);

% Prepare data structure
data_struct = struct();
data_struct.time = dFBA_results.time;
data_struct.profiles = dFBA_results.profiles;
data_struct.flux_growth = dFBA_results.flux_growth;
data_struct.flux_prod = dFBA_results.flux_prod;
data_struct.phase_transition = dFBA_results.phase_transition;

% Extract parameter information (customize based on your parameters)
data_struct.condition = dFBA_results.condition;

% Save as .mat file (can be loaded in Python)
save(export_path, '-struct', 'data_struct');

% Alternative: Save only trajectories and time for lightweight version
profiles_normalized = dFBA_results.profiles ./ max(dFBA_results.profiles, [], 1);
save([export_path, '_normalized'], 'dFBA_results', 'time', 'profiles_normalized');
"""

# ============================================================================
# UTILITIES FOR LOADING AND CONVERTING DATA
# ============================================================================

class MATLABDataConverter:
    """Convert MATLAB .mat files to PyTorch-compatible formats."""
    
    @staticmethod
    def load_mat(mat_file):
        """Load .mat file exported from MATLAB."""
        try:
            data = loadmat(mat_file)
            return data
        except Exception as e:
            raise IOError(f"Failed to load {mat_file}: {e}")
    
    @staticmethod
    def extract_trajectories(mat_data):
        """
        Extract trajectory information from loaded .mat data.
        
        Returns:
            Dict with keys: 'time', 'profiles', 'flux_growth', 'flux_prod'
        """
        trajectories = {}
        
        # Time
        if 'time' in mat_data:
            trajectories['time'] = mat_data['time'].flatten()
        
        # Concentration profiles
        if 'profiles' in mat_data:
            profiles = mat_data['profiles']
            # Remove MATLAB struct wrapping if needed
            if profiles.dtype == object:
                trajectories['profiles'] = profiles[0, 0]
            else:
                trajectories['profiles'] = profiles
        
        # Fluxes
        if 'flux_growth' in mat_data:
            trajectories['flux_growth'] = mat_data['flux_growth']
        if 'flux_prod' in mat_data:
            trajectories['flux_prod'] = mat_data['flux_prod']
        
        # Phase transition
        if 'phase_transition' in mat_data:
            trajectories['phase_transition'] = mat_data['phase_transition'].flatten()
        
        return trajectories
    
    @staticmethod
    def save_as_npz(mat_file, output_npz_file):
        """
        Convert .mat file to .npz format for efficient PyTorch loading.
        """
        mat_data = MATLABDataConverter.load_mat(mat_file)
        trajectories = MATLABDataConverter.extract_trajectories(mat_data)
        np.savez(output_npz_file, **trajectories)
        print(f"Saved to {output_npz_file}")
    
    @staticmethod
    def batch_convert_to_npz(mat_files_dir, output_dir=None):
        """
        Convert all .mat files in a directory to .npz format.
        """
        mat_files_dir = Path(mat_files_dir)
        if output_dir is None:
            output_dir = mat_files_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        mat_files = list(mat_files_dir.glob('*.mat'))
        print(f"Found {len(mat_files)} .mat files")
        
        for mat_file in mat_files:
            output_file = output_dir / (mat_file.stem + '.npz')
            try:
                MATLABDataConverter.save_as_npz(str(mat_file), str(output_file))
                print(f"  ✓ {mat_file.name} → {output_file.name}")
            except Exception as e:
                print(f"  ✗ {mat_file.name}: {e}")
        
        return output_dir


# ============================================================================
# PARAMETER EXTRACTION AND MANAGEMENT
# ============================================================================

class ParameterManager:
    """Manage varying parameters across MATLAB simulations."""
    
    @staticmethod
    def create_parameter_sweep(base_parameters, sweep_dict):
        """
        Create a sweep of parameter variations.
        
        Args:
            base_parameters: Base parameter dict
            sweep_dict: Dict mapping param names to value arrays
        
        Returns:
            List of parameter dicts for each simulation
        """
        import itertools
        
        param_names = sorted(sweep_dict.keys())
        param_ranges = [sweep_dict[name] for name in param_names]
        
        parameter_dicts = []
        for values in itertools.product(*param_ranges):
            params = base_parameters.copy()
            for name, value in zip(param_names, values):
                params[name] = value
            parameter_dicts.append(params)
        
        return parameter_dicts
    
    @staticmethod
    def save_parameter_config(parameters, output_json):
        """Save parameter configuration for reproducibility."""
        # Convert numpy arrays to lists for JSON serialization
        params_serializable = {}
        for key, val in parameters.items():
            if isinstance(val, np.ndarray):
                params_serializable[key] = val.tolist()
            else:
                params_serializable[key] = val
        
        with open(output_json, 'w') as f:
            json.dump(params_serializable, f, indent=2)
    
    @staticmethod
    def load_parameter_config(json_file):
        """Load parameter configuration."""
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        # Convert lists back to arrays
        for key, val in params.items():
            if isinstance(val, list):
                params[key] = np.array(val)
        
        return params


# ============================================================================
# MULTI-SIMULATION DATASET CREATION
# ============================================================================

class MultiSimulationDataset:
    """Combine multiple MATLAB simulations into a training dataset."""
    
    def __init__(self, npz_files_list):
        """
        Args:
            npz_files_list: List of paths to .npz files
        """
        self.npz_files = npz_files_list
        self.trajectories = []
        self.times = []
        self.initial_conditions = []
        self.metadata = []
    
    def load_all(self):
        """Load all .npz files."""
        for npz_file in self.npz_files:
            data = np.load(npz_file, allow_pickle=True)
            
            profiles = data['profiles']
            time = data['time'].flatten() if len(data['time'].shape) > 1 else data['time']
            
            self.trajectories.append(profiles)
            self.times.append(time)
            self.initial_conditions.append(profiles[0, :])
            
            # Store metadata
            self.metadata.append({
                'file': str(npz_file),
                'n_timepoints': len(time),
                'n_components': profiles.shape[1],
                'max_time': time.max(),
                'min_profile': profiles.min(),
                'max_profile': profiles.max(),
            })
        
        print(f"Loaded {len(self.npz_files)} simulations")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics."""
        n_timepoints = [len(t) for t in self.times]
        n_components = [p.shape[1] for p in self.trajectories]
        
        print(f"\n  Time points: min={min(n_timepoints)}, max={max(n_timepoints)}")
        print(f"  Components: {np.unique(n_components)}")
        print(f"  Profile ranges: [{np.min(self.trajectories):.4f}, {np.max(self.trajectories):.4f}]")
    
    def get_aligned_arrays(self, max_timepoints=None):
        """
        Get trajectories aligned to same time dimension.
        Pads shorter trajectories with last value.
        
        Returns:
            trajectories: (n_simulations, n_timepoints, n_components)
            times: (n_simulations, n_timepoints)
            initial_conditions: (n_simulations, n_components)
        """
        if max_timepoints is None:
            max_timepoints = max(len(t) for t in self.times)
        
        n_simulations = len(self.trajectories)
        n_components = self.trajectories[0].shape[1]
        
        trajectories = np.zeros((n_simulations, max_timepoints, n_components))
        times = np.zeros((n_simulations, max_timepoints))
        
        for i, (traj, t) in enumerate(zip(self.trajectories, self.times)):
            n_t = len(t)
            trajectories[i, :n_t, :] = traj
            times[i, :n_t] = t
            
            # Pad with last value
            if n_t < max_timepoints:
                trajectories[i, n_t:, :] = traj[-1, :]
                times[i, n_t:] = t[-1]
        
        return trajectories, times, np.array(self.initial_conditions)
    
    def split_train_val_test(self, train_frac=0.7, val_frac=0.15):
        """
        Split into train/val/test sets.
        
        Returns:
            (train_indices, val_indices, test_indices)
        """
        n = len(self.trajectories)
        indices = np.random.permutation(n)
        
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx


# ============================================================================
# QUICK START TEMPLATE
# ============================================================================

QUICKSTART_TEMPLATE = """
# Quick Start: MATLAB to NN

## Step 1: Export from MATLAB
# Add to your COSMIC_dFBA script:
{matlab_code}

## Step 2: Convert to PyTorch format (Python)

from matlab_bridge import MATLABDataConverter, MultiSimulationDataset
from cosmic_nn_surrogate import dFBADataset
from torch.utils.data import DataLoader

# Convert .mat files to .npz
converter = MATLABDataConverter()
converter.batch_convert_to_npz('path/to/matlab/outputs', 'path/to/output')

# Load all simulations
dataset = MultiSimulationDataset(['sim1.npz', 'sim2.npz', ...])
dataset.load_all()

# Get aligned arrays
trajectories, times, initial_conditions = dataset.get_aligned_arrays()

## Step 3: Create PyTorch dataset

pytorch_dataset = dFBADataset(
    trajectories=trajectories,
    time_points=times,
    initial_conditions=initial_conditions,
    parameters={{}}  # Add any varying parameters here
)

## Step 4: Train model (see integration_guide.py)

from cosmic_nn_surrogate import CosmicNNSurrogate, TrainingManager
from torch.utils.data import DataLoader, random_split

train_loader = DataLoader(pytorch_dataset, batch_size=8, shuffle=True)
model = CosmicNNSurrogate(n_components=5, n_params=0)
trainer = TrainingManager(model)
trainer.train(train_loader, val_loader, epochs=50)

## Step 5: Make predictions

from cosmic_nn_surrogate import PredictionInterface

predictor = PredictionInterface(model, pytorch_dataset)
pred = predictor.predict(
    initial_conditions=np.array([...]),
    time_points=np.linspace(0, 10, 100)
)
"""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("MATLAB-to-Python Bridge for COSMIC dFBA")
    print("=" * 60)
    
    print("\n📋 MATLAB Export Code Template:")
    print("-" * 60)
    print(MATLAB_EXPORT_CODE)
    
    print("\n" + "=" * 60)
    print("Quick Start Instructions:")
    print("=" * 60)
    print(QUICKSTART_TEMPLATE.format(matlab_code=MATLAB_EXPORT_CODE))
