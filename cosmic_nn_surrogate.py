"""
COSMIC dFBA to Neural Network Surrogate Model
Converts MATLAB dFBA simulations into a neural network for fast predictions
and exploration of new parameters/conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class dFBADataset(Dataset):
    """
    Dataset for dFBA simulation trajectories.
    Each sample is a complete simulation trajectory.
    """
    def __init__(self, trajectories, time_points, initial_conditions, 
                 parameters, normalize=True):
        """
        Args:
            trajectories: Array of shape (n_samples, n_timepoints, n_components)
            time_points: Array of time points for each trajectory
            initial_conditions: Array of shape (n_samples, n_components)
            parameters: Dict of parameter arrays
            normalize: Whether to normalize data
        """
        self.trajectories = trajectories
        self.time_points = time_points
        self.initial_conditions = initial_conditions
        self.parameters = parameters
        
        self.n_samples = trajectories.shape[0]
        self.n_components = trajectories.shape[2]
        self.n_timepoints = trajectories.shape[1]
        
        if normalize:
            self._normalize()
    
    def _normalize(self):
        """Normalize trajectories and initial conditions to [0,1]"""
        self.traj_min = np.percentile(self.trajectories, 1, axis=(0, 1))
        self.traj_max = np.percentile(self.trajectories, 99, axis=(0, 1))
        self.traj_max = np.maximum(self.traj_max, self.traj_min + 1e-6)
        
        self.ic_min = np.percentile(self.initial_conditions, 1, axis=0)
        self.ic_max = np.percentile(self.initial_conditions, 99, axis=0)
        self.ic_max = np.maximum(self.ic_max, self.ic_min + 1e-6)
        
        self.trajectories = (self.trajectories - self.traj_min) / (self.traj_max - self.traj_min)
        self.trajectories = np.clip(self.trajectories, 0, 1)
        
        self.initial_conditions = (self.initial_conditions - self.ic_min) / (self.ic_max - self.ic_min)
        self.initial_conditions = np.clip(self.initial_conditions, 0, 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return (initial_conditions, time, parameters) -> trajectory"""
        traj = torch.FloatTensor(self.trajectories[idx])
        ic = torch.FloatTensor(self.initial_conditions[idx])
        
        # Normalize time to [0,1]
        time = torch.FloatTensor(self.time_points[idx] / np.max(self.time_points[idx]))
        
        # Stack parameters
        param_list = []
        for key, val in self.parameters.items():
            if val is not None:
                param_list.append(torch.FloatTensor([val[idx]]))
        
        params = torch.cat(param_list) if param_list else torch.FloatTensor([])
        
        return {
            'initial_conditions': ic,
            'time': time,
            'parameters': params,
            'trajectory': traj
        }


class DynamicsEncoder(nn.Module):
    """Encodes initial conditions and parameters into a dynamics representation."""
    def __init__(self, n_components, n_params, latent_dim=64):
        super().__init__()
        self.n_components = n_components
        self.n_params = n_params
        
        input_size = n_components + n_params
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
        )
        self.latent_dim = latent_dim
    
    def forward(self, initial_conditions, parameters):
        x = torch.cat([initial_conditions, parameters], dim=-1)
        return self.fc(x)


class TemporalDecoder(nn.Module):
    """
    Decodes dynamics representation and time into trajectory predictions.
    Uses multi-head attention for flexible time sampling.
    """
    def __init__(self, n_components, latent_dim=64, n_heads=4):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Multi-head self-attention over latent + time
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_components),
            nn.Sigmoid()  # Constrain to [0,1] since data is normalized
        )
    
    def forward(self, latent_state, time_points):
        """
        Args:
            latent_state: (batch_size, latent_dim)
            time_points: (batch_size, n_timepoints)
        
        Returns:
            trajectory: (batch_size, n_timepoints, n_components)
        """
        batch_size = latent_state.shape[0]
        n_timepoints = time_points.shape[1]
        
        # Embed time
        time_expanded = time_points.unsqueeze(-1)  # (batch, n_t, 1)
        time_embedded = self.time_embed(time_expanded)  # (batch, n_t, latent_dim)
        
        # Expand latent state
        latent_expanded = latent_state.unsqueeze(1).expand(-1, n_timepoints, -1)
        
        # Apply attention (learn correlations over time)
        attn_out, _ = self.attention(time_embedded, latent_expanded, latent_expanded)
        
        # Concatenate and decode
        combined = torch.cat([latent_expanded, attn_out], dim=-1)
        trajectory = self.decoder(combined)
        
        return trajectory


class CosmicNNSurrogate(nn.Module):
    """
    Complete surrogate model: encodes conditions → learns dynamics → decodes trajectory
    """
    def __init__(self, n_components, n_params, latent_dim=64, n_heads=4):
        super().__init__()
        self.encoder = DynamicsEncoder(n_components, n_params, latent_dim)
        self.decoder = TemporalDecoder(n_components, latent_dim, n_heads)
        self.n_components = n_components
    
    def forward(self, initial_conditions, time_points, parameters):
        """
        Args:
            initial_conditions: (batch_size, n_components)
            time_points: (batch_size, n_timepoints)
            parameters: (batch_size, n_params)
        
        Returns:
            trajectory: (batch_size, n_timepoints, n_components)
        """
        latent = self.encoder(initial_conditions, parameters)
        trajectory = self.decoder(latent, time_points)
        return trajectory


class TrainingManager:
    """Manages training of the surrogate model."""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(ic, time, params)
            
            # Loss: MSE + Smoothness regularization
            mse_loss = nn.functional.mse_loss(pred, target)
            
            # Encourage smooth predictions (adjacent timepoints should be similar)
            smoothness = torch.mean((pred[:, 1:, :] - pred[:, :-1, :]) ** 2)
            
            loss = mse_loss + 0.1 * smoothness
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        self.losses.append(epoch_loss)
        self.scheduler.step()
        
        return epoch_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(self.device)
                time = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)
                
                pred = self.model(ic, time, params)
                loss = nn.functional.mse_loss(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} (BEST)")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_loss
    
    def save(self, path):
        """Save model and training info."""
        torch.save({
            'model_state': self.model.state_dict(),
            'losses': self.losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.losses = checkpoint['losses']
        print(f"Model loaded from {path}")


class PredictionInterface:
    """User-friendly interface for making predictions."""
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataset = dataset
    
    def predict(self, initial_conditions, time_points, parameters=None):
        """
        Make predictions for new conditions.
        
        Args:
            initial_conditions: Array of shape (n_components,) or (batch_size, n_components)
            time_points: Array of time points to predict
            parameters: Dict or array of parameters
        
        Returns:
            predictions: Array of shape (n_timepoints, n_components) or (batch_size, n_timepoints, n_components)
        """
        # Handle single vs batch inputs
        single_input = False
        if initial_conditions.ndim == 1:
            initial_conditions = initial_conditions[np.newaxis, :]
            single_input = True
        
        # Normalize
        ic_norm = (initial_conditions - self.dataset.ic_min) / (self.dataset.ic_max - self.dataset.ic_min)
        ic_norm = np.clip(ic_norm, 0, 1)
        
        time_norm = time_points / np.max(time_points)
        
        # Convert to tensors
        ic_tensor = torch.FloatTensor(ic_norm).to(self.device)
        time_tensor = torch.FloatTensor(time_norm).unsqueeze(0).to(self.device)
        
        # Handle parameters
        if parameters is not None:
            if isinstance(parameters, dict):
                param_list = [torch.FloatTensor([parameters[key]]) for key in sorted(parameters.keys())]
                params_tensor = torch.cat(param_list).unsqueeze(0).to(self.device)
            else:
                params_tensor = torch.FloatTensor(parameters).unsqueeze(0).to(self.device)
        else:
            params_tensor = torch.zeros(1, 1).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_norm = self.model(ic_tensor, time_tensor, params_tensor)
        
        # Denormalize
        pred = pred_norm.cpu().numpy() * (self.dataset.traj_max - self.dataset.traj_min) + self.dataset.traj_min
        pred = np.maximum(pred, 0)  # Ensure non-negative
        
        if single_input:
            return pred[0]
        return pred
    
    def sensitivity_analysis(self, initial_conditions, time_points, param_name, 
                           param_range, n_points=10, parameters=None):
        """
        Analyze sensitivity to a parameter.
        
        Returns:
            param_values: Parameter values tested
            predictions: List of predictions for each parameter value
        """
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        predictions = []
        
        for pval in param_values:
            if parameters is None:
                params = {param_name: pval}
            else:
                params = parameters.copy()
                params[param_name] = pval
            
            pred = self.predict(initial_conditions, time_points, params)
            predictions.append(pred)
        
        return param_values, predictions


# Example usage and helper functions
def load_matlab_simulations(matlab_data_file):
    """
    Load dFBA simulations exported from MATLAB.
    Expected format: .npz file with 'time', 'profiles', 'parameters'
    """
    data = np.load(matlab_data_file)
    return data


def create_dataset_from_simulations(matlab_files_list, batch_size=16):
    """Create a PyTorch dataset from multiple MATLAB simulation files."""
    all_trajectories = []
    all_times = []
    all_ics = []
    all_params = {}
    
    for matlab_file in matlab_files_list:
        data = np.load(matlab_file)
        
        # Extract trajectory
        profiles = data['profiles']  # (n_timepoints, n_components)
        time = data['time']  # (n_timepoints,)
        
        all_trajectories.append(profiles[np.newaxis, :, :])  # Add batch dimension
        all_times.append(time)
        all_ics.append(profiles[0])
        
        # Extract parameters if available
        if 'parameters' in data:
            for key, val in data['parameters'].items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(val)
    
    # Stack into arrays
    trajectories = np.vstack(all_trajectories)
    n_samples = trajectories.shape[0]
    
    # Pad trajectories to same length
    max_timepoints = max(t.shape[0] for t in all_times)
    trajectories_padded = np.zeros((n_samples, max_timepoints, trajectories.shape[2]))
    times_padded = np.zeros((n_samples, max_timepoints))
    
    for i, traj in enumerate(trajectories):
        trajectories_padded[i, :traj.shape[0], :] = traj
        times_padded[i, :all_times[i].shape[0]] = all_times[i]
    
    initial_conditions = np.array(all_ics)
    
    # Convert params to arrays
    for key in all_params:
        all_params[key] = np.array(all_params[key])
    
    # Create dataset
    dataset = dFBADataset(trajectories_padded, times_padded, initial_conditions, all_params)
    
    return dataset


if __name__ == "__main__":
    print("COSMIC dFBA Neural Network Surrogate Model")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    # 1. Load MATLAB dFBA simulations
    dataset = create_dataset_from_simulations(['sim1.npz', 'sim2.npz'])
    
    # 2. Create model
    model = CosmicNNSurrogate(
        n_components=dataset.n_components,
        n_params=10,  # or calculate from dataset
        latent_dim=64
    )
    
    # 3. Train
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer = TrainingManager(model)
    trainer.train(train_loader, val_loader, epochs=100)
    
    # 4. Predict
    predictor = PredictionInterface(model, dataset)
    pred = predictor.predict(
        initial_conditions=np.array([...]),
        time_points=np.linspace(0, 10, 100),
        parameters={'param1': 0.5, 'param2': 1.0}
    )
    """)
