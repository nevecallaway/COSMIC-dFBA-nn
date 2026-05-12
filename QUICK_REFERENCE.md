# Quick Reference: Using Enhanced COSMIC-dFBA Model

## 🚀 Quick Start (3 Steps)

### 1. Train Enhanced Model
```bash
cd /Users/nevecallaway/COSMIC-dFBA-nn/COSMIC-dFBA-nn
python nn/test_demo_enhanced.py
```

### 2. Understand Outputs
The enhanced model outputs a dictionary with 4 keys:
```python
result = predictor.predict(initial_conditions, time_points, return_rates=True)

result['concentrations']  # (n_time, n_components) - metabolite levels
result['phase_weights']   # (n_time, 1) - sigmoid(0→1) growth→production
result['growth_rates']    # (n_time, n_components) - rates in growth phase
result['prod_rates']      # (n_time, n_components) - rates in production phase
```

### 3. Analyze Phase Transitions
```python
analysis = predictor.analyze_phase_transition(ic, time, params)
# Plots: phase progression, metabolite dynamics, rate switching
```

---

## 🔄 Choosing Your Model

| Model Type | Use Case | Speed | Interpretability |
|----------|----------|-------|-----------------|
| **Standard** | Fast predictions | 10-100ms | Basic |
| **Enhanced** | Phase analysis + rates | 15-150ms | High |

```python
# Standard (original - still works)
from cosmic_nn_surrogate import CosmicNNSurrogate
model = CosmicNNSurrogate(n_components=5, n_params=0)
pred = predictor.predict(ic, time)  # Returns array directly

# Enhanced (new - paper-compliant)
from cosmic_nn_surrogate import CosmicNNSurrogateEnhanced
model = CosmicNNSurrogateEnhanced(n_components=5, n_params=0)
result = predictor.predict(ic, time, return_rates=True)  # Returns dict
```

---

## 📊 What Each Head Learns

### 1. Concentration Decoder
- **Learns**: Metabolite dynamics (biomass, substrate, products)
- **Output**: Normalized concentrations [0, 1]
- **Biological**: What's in the bioreactor

### 2. State Weighting Layer
- **Learns**: Phase transition timing and smoothness
- **Output**: Weight ∈ [0, 1] indicating cell state
- **Biological**: `0` = all cells growing, `1` = all cells producing

### 3. Rate Prediction Head
- **Learns**: State-specific metabolic rates
- **Outputs**: 
  - `growth_rates`: Consumption/production during growth phase
  - `prod_rates`: Consumption/production during production phase
- **Biological**: Different kinetics per metabolic state

---

## 🏋️ Training Code Template

```python
import torch
import numpy as np
from cosmic_nn_surrogate import (
    CosmicNNSurrogateEnhanced, TrainingManager, dFBADataset
)
from torch.utils.data import DataLoader, random_split

# 1. Load your MATLAB data
trajectories = np.load('dFBA_sim.npz')['profiles']  # (n_sim, n_time, n_comp)
time_points = np.load('dFBA_sim.npz')['time']
initial_conditions = trajectories[:, 0, :]

# 2. Create PyTorch dataset
dataset = dFBADataset(
    trajectories=trajectories,
    time_points=time_points,
    initial_conditions=initial_conditions,
    parameters={},  # Add if varying parameters
    normalize=True
)

# 3. Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# 4. Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CosmicNNSurrogateEnhanced(
    n_components=dataset.n_components,
    n_params=0,
    latent_dim=64,
    n_heads=4
)

# 5. Train (enhanced loss)
trainer = TrainingManager(
    model,
    device=device,
    learning_rate=1e-3,
    model_type='enhanced'
)

best_loss = trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    patience=20
)

# 6. Save
trainer.save('results/cosmic_model_enhanced.pt')
```

---

## 🔍 Inspection & Analysis

### View Phase Transition
```python
# Plot shows growth → production transition
result = predictor.predict(ic, time, return_rates=True)
phases = result['phase_weights'].flatten()

print(f"Start: {phases[0]:.2f}")     # ~0 (growth)
print(f"End: {phases[-1]:.2f}")       # ~1 (production)
print(f"Midpoint: {time[np.argmin(np.abs(phases-0.5))]:.1f} days")
```

### Compare Rates Across Phases
```python
growth_rates = result['growth_rates']
prod_rates = result['prod_rates']

# Substrate (component 1)
print("Growth phase substrate uptake:", growth_rates[:, 1].mean())
print("Production phase substrate uptake:", prod_rates[:, 1].mean())
```

### Sensitivity to Initial Conditions
```python
for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
    ic_scaled = ic * scale
    result = predictor.predict(ic_scaled, time, return_rates=True)
    print(f"Scale {scale}: Transition at {time[np.argmin(np.abs(result['phase_weights']-0.5))]:.1f}d")
```

---

## 🚨 Troubleshooting

**Q: Model predictions are NaNs**
- ✓ Check normalize=True in dFBADataset
- ✓ Check gradient clipping in TrainingManager
- ✓ Reduce learning rate

**Q: Phase weights are stuck at 0.5**
- ✓ Check phase_penalty term in loss (should push toward extremes)
- ✓ Increase training epochs/patience
- ✓ Check synthetic data has clear phase transitions

**Q: Rates are unrealistic**
- ✓ Rates use Tanh activation → range is [-1, 1]
- ✓ Scale appropriately when denormalizing
- ✓ Consider rate_magnitude penalty in loss

**Q: Slow prediction time**
- ✓ CPU slower than GPU (15-150ms vs 10-50ms)
- ✓ Move model to GPU: `model.cuda()`
- ✓ Use smaller batch size

---

## 📈 Production Checklist

- [ ] Trained on 50+ MATLAB simulations
- [ ] Validated on held-out test set (RMSE < 10%)
- [ ] Phase transitions match expected biology
- [ ] Rates make biochemical sense
- [ ] Model saved with metadata
- [ ] Reproducible results (fixed random seed)

```python
# Set seed for reproducibility
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## 📞 Key Functions Reference

```python
from cosmic_nn_surrogate import (
    CosmicNNSurrogateEnhanced,      # Main model
    TrainingManager,                # Training
    PredictionInterface,            # Inference
    MetabolicConstraintEnforcer,    # Constraints
    dFBADataset,                    # Data loading
)

# Training
trainer = TrainingManager(model, model_type='enhanced')
loss = trainer.train(train_loader, val_loader, epochs=100)

# Prediction
predictor = PredictionInterface(model, dataset, model_type='enhanced')
result = predictor.predict(ic, time, return_rates=True)
analysis = predictor.analyze_phase_transition(ic, time)

# Constraints
enforcer = MetabolicConstraintEnforcer()
enforcer.enforce_non_negativity(concentrations)
enforcer.enforce_mass_balance(conc, phases, growth_rates, prod_rates)
```

---

## 🎯 Next Steps

1. **Test on Real Data**: Export MATLAB simulations → train → validate
2. **Tune Hyperparameters**: latent_dim, n_heads, learning_rate
3. **Add Constraints**: Integrate stoichiometry into loss function
4. **Deploy**: Use `train_production.py` on supercomputer
5. **Monitor**: Track phase transitions and rate predictions

---

## 📚 Related Files

- `cosmic_nn_surrogate.py` - Main implementation (600+ lines)
- `test_demo_enhanced.py` - Full working example
- `train_production.py` - Production training script
- `IMPLEMENTATION_SUMMARY.md` - Detailed technical documentation
- `run_cosmic_slurm.sh` - SLURM job submission
