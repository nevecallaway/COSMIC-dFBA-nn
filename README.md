# COSMIC dFBA Neural Network Surrogate Model

A PyTorch surrogate model for predicting bioreactor phase transitions and metabolite trajectories, trained on experimental dFBA data from 10 perfusion reactors.

The primary prediction goal is transition timing for root cause analysis (RCA): predicting when and how a cell line switches from growth phase to production phase, not just final titer. The loss is weighted accordingly, favoring phase transition accuracy over titer. The phase prediction head explicitly outputs two interpretable parameters per reactor: mu (transition midpoint in days) and sigma (transition sharpness in days). Evaluation includes phase AUC, which captures both timing and sharpness of the transition in a single number, alongside transition MAE and standard classification metrics.

**Latest LOO results (simplified model, DoE-only inputs):** f(t) accuracy +/-0.1: 81.5% (paper: 72.3%) | MCC: 0.913 (paper: 0.454) | Transition MAE: 1.38 days

---

## Files

| File | Purpose |
|------|---------|
| `nn/model.py` | Model classes: dataset, encoder, rate heads, phase head, ODE integrator |
| `nn/train.py` | Leave-one-out training on real reactor data |
| `nn/evaluate.py` | Evaluation metrics and comparison plots vs. paper benchmarks |
| `nn/utils.py` | Data loading, normalization, and diagnostic utilities |
| `nn/data/data_1.csv` | DoE coded levels per reactor (O2, AAs, Glc: -1 / 0 / +1) -- model input |
| `nn/data/data_2.csv` | State variable trajectories over time (25 components, 10 reactors) -- model target |
| `nn/data/data_3.csv` | Phase-specific metabolic rates -- not used as model input (requires dFBA) |
| `nn/data/data_4.csv` | FBA objective efficiencies -- not used as model input (requires dFBA) |

---

## Data Layout

**Dimensions used throughout:**

| Symbol | Value | Meaning |
|--------|-------|---------|
| B | 4 (train) / 1 (val) | Batch size |
| T | 13 | Time points per reactor (one per day, day 0-12) |
| C | 25 | Metabolite components |
| n_params | 3 | DoE coded levels: O2, AAs, Glc |
| latent_dim | 32 | Latent state dimension |

**Component layout (C=25):**

| Index | Component |
|-------|-----------|
| 0 | Cell Density |
| 1 | Cell Volume |
| 2 | Glucose |
| 3 | Lactate |
| 4 | Ammonia |
| 5 | Titer (antibody) |
| 6-24 | Amino acids (Glutamine ... Tryptophan) |

**Parameter layout (n_params=3):**

| Index | Content |
|-------|---------|
| 0 | O2 coded level (-1, 0, +1) |
| 1 | AAs coded level (-1, 0, +1) |
| 2 | Glc coded level (-1, 0, +1) |

FBA-derived features (data_3 specific rates, data_4 efficiencies) were tested and dropped. They require running dFBA first, which defeats the purpose of a surrogate model. Ablation: removing them costs ~0.3d on LOO transition MAE (1.28d -> 1.60d).

---

## Model Architecture

The architecture was deliberately kept minimal. With 10 reactors and 3 DoE inputs the problem is low-dimensional, so simpler generalizes better.

Earlier iterations used an LSTM or transformer to predict time-varying rate tensors `(B, T, 25)` and ran the ODE twice (once growth-only to inform the phase head, once blended). Ablations showed this added complexity without improving LOO performance. The current design predicts constant rates per phase -- biologically appropriate since dFBA rates are phase-constant -- and runs a single ODE pass.

```
INPUTS
  initial_conditions  (B, 25)   normalized concentrations at t=0
  time_points         (B, T)    normalized time in [0, 1]  (actual days / 13)
  parameters          (B, 3)    DoE coded levels: O2, AAs, Glc
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DynamicsEncoder                          (model.py)        │
│                                                             │
│  IN:  cat([initial_conditions, parameters])  →  (B, 28)    │
│  FC1: Linear(28, 64) + ReLU                                 │
│  FC2: Linear(64, 32)                                        │
│  OUT: latent_state                           →  (B, 32)     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Rate heads                               (model.py)        │
│                                                             │
│  amp          = Softplus(Linear(32, 25))     →  (B, 25)    │
│  growth_rates = Tanh(Linear(32, 25)) * amp   →  (B, 25)    │
│  prod_rates   = Tanh(Linear(32, 25)) * amp   →  (B, 25)    │
│                                                             │
│  Rates are constant per reactor per phase (no LSTM/time     │
│  embedding). Expanded to (B, T, 25) for the integrator.    │
│  Amplitude separates magnitude from direction.              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  PhaseTransitionHead                      (model.py)        │
│                                                             │
│  IN:  latent_state  (B, 32)                                 │
│  raw   = Linear(32, 2)                       →  (B, 2)     │
│    mu    = sigmoid(raw[:,0])  in (0,1)   transition midpoint│
│    sigma = softplus(raw[:,1]) > 0        transition width   │
│                                                             │
│  f(t) = sigmoid((time - mu) / sigma)        →  (B, T, 1)  │
│  OUT: phase_weights  in [0,1]               →  (B, T, 1)  │
│    0 = pure growth phase, 1 = pure production phase         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DifferentiableIntegrator                 (model.py)        │
│                                                             │
│  blended_rates = (1 - f(t)) * growth_rates                  │
│                +       f(t)  * prod_rates    →  (B, T, 25) │
│                                                             │
│  Implicit Euler ODE per timestep:                           │
│    cells (idx 0-1):  c_next = c_prev + v * c_prev * dt     │
│    metabolites:      c_next = (c_prev + (v*c1 + F*c_in)*dt)│
│                               / (1 + F*dt)                  │
│    F_NORM = 13.0  (perfusion rate in normalised time)       │
│    titer (idx 5):  eta=0, no washout                        │
│    c_in built from DoE coded levels (feed concentrations)   │
│                                                             │
│  OUT: concentrations        (B, T, 25)                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUTS (dict)
  concentrations  (B, T, 25)   predicted state variable trajectories
  phase_weights   (B, T, 1)    f(t): phase fraction at each timepoint
  growth_rates    (B, T, 25)   growth-phase rates (constant, expanded)
  prod_rates      (B, T, 25)   production-phase rates (constant, expanded)
  transition_mu   (B,)         normalised transition midpoint (x13 = days)
  transition_sigma (B,)        normalised transition width (x13 = days)
```

---

## Training Loop

```
TRAINING DATA: 9 reactors per fold, batch_size=4
  ic:      (4, 25)      initial conditions (normalized)
  time:    (4, 13)      normalized time [0, 1]
  params:  (4, 3)       DoE coded levels: O2, AAs, Glc
  target:  (4, 13, 25)  ground truth state variable trajectories
  phases:  (4, 13)      ground truth f(t) phase fraction
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  FORWARD PASS                                               │
│  predictions = model(ic, time, params)                      │
│  OUT: concentrations  (4, 13, 25)                           │
│       phase_weights   (4, 13, 1)                            │
│       growth_rates    (4, 13, 25)                           │
│       prod_rates      (4, 13, 25)                           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  LOSS COMPUTATION  (Trainer.compute_loss)                   │
│                                                             │
│  1. Concentration MSE                                       │
│       cell density 3x weight (primary RCA target)          │
│       titer 2x weight                                       │
│     IN:  concentrations (4,13,25) vs target (4,13,25)       │
│                                                             │
│  2. Endpoint titer loss (weight 0.5)                        │
│     IN:  concentrations[:,−1,5] vs target[:,−1,5]           │
│                                                             │
│  3. Peak-time alignment (weight 1.0)                        │
│     soft-argmax on titer trajectory to align peak day       │
│                                                             │
│  4. Initial condition constraint (weight 0.1)               │
│     IN:  concentrations[:,0,:] vs ic (4,25)                 │
│                                                             │
│  5. Non-flatness penalty (weight 0.2)                       │
│     penalizes low variance trajectories                     │
│                                                             │
│  6. Non-negativity penalty (weight 0.5)                     │
│     penalizes concentrations < 0                            │
│                                                             │
│  7. Phase regression MSE (weight 5.0)                       │
│     IN:  phase_weights (4,13,1) vs phases (4,13)            │
│                                                             │
│  8. Rate + phase smoothness (weights 0.1 / 0.05)            │
│     penalizes large step-to-step changes in rates and f(t)  │
│                                                             │
│  OUT: total_loss (scalar)                                   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKWARD PASS                                              │
│  loss.backward()                                            │
│  Computes gradients for all parameters:                     │
│    encoder FC weights, rate head weights,                   │
│    amplitude head, phase head, ODE is differentiable        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  clip_grad_norm_(max_norm=1.0)   prevents exploding gradients
        │
        ▼
  AdamW step: w = w - lr * (gradient + momentum)
  lr = 1e-4, weight_decay = 1e-5
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  AFTER EACH EPOCH                                           │
│  Validate on held-out reactor (1 reactor, batch_size=1)     │
│  Metrics computed: F1, MCC, transition MAE, titer within 10%│
│  ReduceLROnPlateau: halve lr if val_loss stagnates          │
│  Early stopping if no improvement for 80 epochs             │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Strategy

**Leave-one-out cross-validation**: with 10 reactors, each is held out once as the test set. The model trains from scratch for each fold on the remaining 9 reactors. LOO metrics are averaged across all 10 folds and reported as the main generalization estimate.

**Final model**: after LOO, the model is re-trained on all 10 reactors and saved to `improved_model.pt`.

---

## Running

```bash
# Train
python nn/train.py

# Train permutation baseline (shuffle inputs vs outputs to establish chance performance)
python nn/train.py --shuffle

# Evaluate saved model
python nn/evaluate.py
```
