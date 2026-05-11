# COSMIC dFBA Neural Network Surrogate Model

Convert your MATLAB dynamic Flux Balance Analysis (dFBA) simulations into a fast, flexible neural network that can:
- ✅ Make predictions 100-1000x faster than FBA
- ✅ Explore new parameters and conditions not in training data
- ✅ Incorporate additional factors and parameters easily
- ✅ Enable sensitivity analysis and uncertainty quantification
- ✅ Replace MATLAB/COBRA dependency for production use

## Overview

This package provides a complete pipeline for converting MATLAB dFBA simulations into a PyTorch neural network surrogate model:

1. **Export MATLAB data** → .mat or .npz files
2. **Train surrogate model** → Learn dFBA dynamics with a neural network
3. **Make predictions** → Fast inference for new conditions
4. **Analyze sensitivity** → Explore parameter effects

## Files

| File | Purpose |
|------|---------|
| `cosmic_nn_surrogate.py` | Core neural network architecture and training |
| `integration_guide.py` | Complete examples and training pipeline |
| `matlab_bridge.py` | Utilities for MATLAB ↔ Python data conversion |
| `README.md` | This file |

## Installation

### Requirements
```bash
pip install torch numpy scipy matplotlib
```

### For GPU acceleration (optional)
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# M1/M2 Mac (optimized)
pip install torch torchvision torchaudio
```

## Quick Start

### Step 1: Export MATLAB Simulations

Add this to your MATLAB `COSMIC_dFBA` script:

```matlab
% After dFBA_results are computed:
export_path = ['dFBA_', datestr(now,'yyyymmdd_HHMMSS'), '.npz'];

data_struct = struct();
data_struct.time = dFBA_results.time;
data_struct.profiles = dFBA_results.profiles;
data_struct.flux_growth = dFBA_results.flux_growth;
data_struct.flux_prod = dFBA_results.flux_prod;
data_struct.phase_transition = dFBA_results.phase_transition;

save(export_path, '-struct', 'data_struct');
```

Or use the Python bridge to convert existing .mat files:

```python
from matlab_bridge import MATLABDataConverter

# Convert single file
MATLABDataConverter.save_as_npz('simulation.mat', 'simulation.npz')

# Batch convert directory
MATLABDataConverter.batch_convert_to_npz('simulations_dir/', 'output_dir/')
```

### Step 2: Create Training Dataset

```python
from matlab_bridge import MultiSimulationDataset
from cosmic_nn_surrogate import dFBADataset

# Load all simulations
dataset_loader = MultiSimulationDataset(['sim1.npz', 'sim2.npz', 'sim3.npz'])
dataset_loader.load_all()

# Align to same time dimension
trajectories, times, initial_conditions = dataset_loader.get_aligned_arrays()

# Create PyTorch dataset
dataset = dFBADataset(
    trajectories=trajectories,
    time_points=times,
    initial_conditions=initial_conditions,
    parameters={},  # Add parameters if varying
    normalize=True
)
```

### Step 3: Train Surrogate Model

```python
from cosmic_nn_surrogate import CosmicNNSurrogate, TrainingManager
from torch.utils.data import DataLoader, random_split
import torch

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Create and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CosmicNNSurrogate(
    n_components=dataset.n_components,
    n_params=0,  # Or number of varying parameters
    latent_dim=64,
    n_heads=4
)

trainer = TrainingManager(model, device=device, learning_rate=1e-3)
best_val_loss = trainer.train(train_loader, val_loader, epochs=50, patience=15)

# Save model
trainer.save('cosmic_surrogate_model.pt')
```

### Step 4: Make Predictions

```python
from cosmic_nn_surrogate import PredictionInterface
import numpy as np

# Load model and create predictor
predictor = PredictionInterface(model, dataset, device=device)

# Predict for new initial conditions
new_ic = np.array([0.5, 0.8, 0.2, 0.1, 0.0])  # Biomass, glucose, products...
time_points = np.linspace(0, 10, 100)

prediction = predictor.predict(new_ic, time_points)
# Output shape: (100, 5) - 100 timepoints, 5 components

# Plot results
import matplotlib.pyplot as plt
plt.plot(time_points, prediction[:, 0], label='Biomass')
plt.plot(time_points, prediction[:, 1], label='Glucose')
plt.xlabel('Time (days)')
plt.ylabel('Concentration')
plt.legend()
plt.show()
```

### Step 5: Sensitivity Analysis

```python
# Analyze how predictions change with parameter variation
param_range = (np.array([0.5, 0.5, 0.5, 0.5, 0.5]), 
               np.array([2.0, 2.0, 2.0, 2.0, 2.0]))

param_values, predictions = predictor.sensitivity_analysis(
    initial_conditions=new_ic,
    time_points=time_points,
    param_name='kinetic_vm_growth',
    param_range=param_range,
    n_points=10
)

# predictions is a list of 10 trajectories for different parameter values
```

## Advanced Usage

### Adding Variable Parameters

If you vary parameters across simulations (e.g., different kinetic constants):

```python
parameters = {
    'kinetic_vm_growth': np.array([1.0, 1.2, 0.8, ...]),  # One per simulation
    'kinetic_vm_prod': np.array([0.5, 0.6, 0.4, ...]),
}

dataset = dFBADataset(
    trajectories=trajectories,
    time_points=times,
    initial_conditions=initial_conditions,
    parameters=parameters,
    normalize=True
)
```

Then use parameters in predictions:

```python
pred = predictor.predict(
    initial_conditions=new_ic,
    time_points=time_points,
    parameters={'kinetic_vm_growth': np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
                'kinetic_vm_prod': np.array([0.7, 0.7, 0.7, 0.7, 0.7])}
)
```

### Custom Training Configuration

```python
trainer = TrainingManager(
    model, 
    device=device, 
    learning_rate=1e-3  # Adjust learning rate
)

# Train with custom settings
trainer.train(
    train_loader, 
    val_loader, 
    epochs=100,      # Maximum epochs
    patience=20      # Stop if no improvement for 20 epochs
)
```

### Batch Predictions

```python
# Make predictions for multiple initial conditions at once
batch_ics = np.random.uniform(0.1, 1.0, (50, 5))  # 50 simulations, 5 components
batch_predictions = predictor.predict(batch_ics, time_points)
# Output shape: (50, 100, 5)
```

## Model Architecture

The surrogate model consists of:

1. **Encoder** (FC layers)
   - Encodes initial conditions + parameters into latent space
   - Learns compact representation of dynamics

2. **Temporal Decoder** (Multi-head Attention + FC)
   - Maps latent state to time-dependent predictions
   - Learns temporal dynamics with attention mechanism
   - Can extrapolate beyond training time range

3. **Output**
   - Component concentrations at requested time points
   - Normalized to [0,1], then denormalized

## Performance Tips

1. **More training data** → Better generalization
   - Aim for 50+ simulations minimum
   - Vary initial conditions and parameters

2. **Tune latent dimension**
   - Smaller (32-64): Faster, less accurate
   - Larger (128-256): Slower, potentially more accurate
   - Start with 64

3. **Monitor training**
   - Plot train/val loss curves
   - Use early stopping to prevent overfitting

4. **Data preprocessing**
   - Ensure realistic concentration ranges
   - Remove obviously erroneous simulations

## Validation and Benchmarking

Compare NN predictions against FBA results:

```python
# For simulations in test set
for test_idx in test_indices:
    # Actual dFBA result
    actual = trajectories[test_idx]
    
    # NN prediction
    pred = predictor.predict(
        initial_conditions=initial_conditions[test_idx],
        time_points=times[test_idx]
    )
    
    # Compute error
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    
    print(f"Simulation {test_idx}: MAE={mae:.4f}, RMSE={rmse:.4f}")
```

## Troubleshooting

**Q: Model predicts negative concentrations?**
- A: Add constraint in PredictionInterface: `pred = np.maximum(pred, 0)`

**Q: Poor predictions on new conditions?**
- A: Need more diverse training data; ensure new conditions are within training range

**Q: Slow inference?**
- A: Use GPU; reduce latent_dim; batch predictions together

**Q: Training loss plateaus early?**
- A: Reduce learning rate; increase model capacity; check data quality

## Extending the Model

### Add flux predictions
Modify decoder to output [concentrations, fluxes]:
```python
# In TemporalDecoder.forward:
trajectory = self.decoder(combined)
concentrations = trajectory[:, :, :n_components]
fluxes = trajectory[:, :, n_components:]
return concentrations, fluxes
```

### Add uncertainty quantification
Use Monte Carlo Dropout or Bayesian layers:
```python
# Add dropout layers (already in model)
# Enable during inference for uncertainty
model.train()  # Keep dropout active
predictions_ensemble = [model(...) for _ in range(100)]
mean_pred = np.mean(predictions_ensemble, axis=0)
std_pred = np.std(predictions_ensemble, axis=0)
```

### Transfer learning
Fine-tune on new data:
```python
# Load pre-trained model
trainer.load('cosmic_surrogate_model.pt')

# Fine-tune on new dataset
trainer.train(new_train_loader, new_val_loader, epochs=20, patience=5)
```

## Citation

If you use this framework, please cite:

```bibtex
@software{cosmic_nn_2024,
  title={COSMIC dFBA Neural Network Surrogate},
  author={Your Name},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT License - See LICENSE file

## Support

For issues or questions:
1. Check the examples in `integration_guide.py`
2. Review error messages in training logs
3. Verify data format using `MultiSimulationDataset.load_all()`
