# COSMIC dFBA Neural Network Surrogate Model

A PyTorch surrogate model for predicting bioreactor phase transitions and metabolite trajectories, trained on experimental dFBA data from 10 perfusion reactors.

The primary prediction goal is transition timing for root cause analysis (RCA): predicting when and how a cell line switches from growth phase to production phase, not just final titer. The loss is weighted accordingly, favoring phase transition accuracy over titer. The phase prediction head explicitly outputs two interpretable parameters per reactor: mu (transition midpoint in days) and sigma (transition sharpness in days). Evaluation includes phase AUC, which captures both timing and sharpness of the transition in a single number, alongside transition MAE and standard classification metrics.

**Latest LOO results:** f(t) accuracy +/-0.1: 90.0% (paper: 72.3%) | MCC: 0.933 (paper: 0.454) | Transition MAE: 1.28 days

---

## Files

| File | Purpose |
|------|---------|
| `nn/model.py` | All neural network classes: dataset, encoder, decoder, ODE integrator, phase head |
| `nn/train.py` | Two-phase training: synthetic pre-training + leave-one-out fine-tuning |
| `nn/evaluate.py` | Evaluation metrics and comparison plots vs. paper benchmarks |
| `nn/utils.py` | Data loading, normalization, and diagnostic utilities |
| `nn/data/data_1.csv` | DoE coded levels per reactor (O2, AAs, Glc: -1 / 0 / +1) |
| `nn/data/data_2.csv` | Metabolite trajectories over time (25 components, 10 reactors) |
| `nn/data/data_3.csv` | Phase-specific metabolic rates (growth and production phase) |
| `nn/data/data_4.csv` | FBA objective efficiencies per reactor |

---

## Data Layout

**Dimensions used throughout:**

| Symbol | Value | Meaning |
|--------|-------|---------|
| B | 4 (train) / 1 (val) | Batch size |
| T | 40 | Time points per reactor |
| C | 25 | Metabolite components |
| n_params | 75 | DoE (3) + specific rates (50) + FBA efficiencies (22) |
| latent_dim | 64 | Latent state dimension |

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

**Parameter layout (n_params=75):**

| Index | Content |
|-------|---------|
| 0-2 | DoE coded levels: O2, AAs, Glc (values: -1, 0, +1) |
| 3-27 | Growth-phase specific rates (25) |
| 28-52 | Production-phase specific rates (25) |
| 53-74 | FBA objective efficiencies (22): how well each of 11 biological objectives was satisfied in growth and production phase, from the paper's NLP optimization |

---

## Model Architecture

```
INPUTS
  initial_conditions  (B, 25)   normalized concentrations at t=0
  time_points         (B, T)    normalized time in [0, 1]  (actual days / 13)
  parameters          (B, 75)   DoE levels + rates + FBA efficiencies
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DynamicsEncoder                          (model.py)        │
│                                                             │
│  IN:  cat([initial_conditions, parameters])  →  (B, 81)    │
│  FC1: Linear(81, 128) + ReLU + Dropout(0.2)                 │
│  FC2: Linear(128, 128) + ReLU + Dropout(0.2)                │
│  FC3: Linear(128, 64)                                       │
│  OUT: latent_state                           →  (B, 64)     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  LSTMTemporalDecoder                      (model.py)        │
│                                                             │
│  Time embedding:                                            │
│    IN:  time_points.unsqueeze(-1)            →  (B, T, 1)  │
│    OUT: time_embedded                        →  (B, T, 64) │
│                                                             │
│  LSTM init from latent:                                     │
│    h0 = Linear(64, 64) × n_layers           →  (2, B, 64) │
│    c0 = Linear(64, 64) × n_layers           →  (2, B, 64) │
│                                                             │
│  LSTM:                                                      │
│    IN:  time_embedded, (h0, c0)              →  (B, T, 64) │
│    OUT: lstm_out                             →  (B, T, 64) │
│                                                             │
│  Skip connection:                                           │
│    combined = cat([lstm_out, latent_expanded]) → (B, T, 128)│
│                                                             │
│  Amplitude scalar (per-reactor, per-component):             │
│    IN:  latent_state                         →  (B, 64)    │
│    OUT: amp = Softplus(Linear)               →  (B, 1, 25) │
│                                                             │
│  Rate heads (Tanh output × amplitude):                      │
│    growth_rates = Tanh(FC(combined)) * amp  →  (B, T, 25) │
│    prod_rates   = Tanh(FC(combined)) * amp  →  (B, T, 25) │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DifferentiableIntegrator — Pass 1        (model.py)        │
│                                                             │
│  IN:  initial_conditions   (B, 25)                          │
│       growth_rates         (B, T, 25)   (f=0, growth only) │
│       time_points          (B, T)                           │
│       doe_params           (B, 3)       for perfusion feed  │
│                                                             │
│  Implicit Euler ODE per timestep:                           │
│    cells (idx 0-1):  c_next = c_prev + v * c_prev * dt     │
│    metabolites:      c_next = (c_prev + (v*c1 + F*c_in)*dt)│
│                               / (1 + F*dt)                  │
│    F_NORM = 13.0  (perfusion rate in normalised time)       │
│    titer (idx 5):  eta=0, no washout                    │
│                                                             │
│  OUT: conc_pass1           (B, T, 25)                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  PhaseTransitionHead                      (model.py)        │
│                                                             │
│  IN:  latent_state         (B, 64)                          │
│       conc_pass1           (B, T, 25)                       │
│       time_points          (B, T)                           │
│                                                             │
│  conc_summary = MeanPool(conc_pass1) over T  →  (B, 25)    │
│  conc_encoded = FC(conc_summary)             →  (B, 16)    │
│  combined     = cat([latent, conc_encoded])  →  (B, 80)    │
│  raw          = FC(combined)                 →  (B, 2)     │
│    mu    = sigmoid(raw[:,0])     in (0,1)  (transition midpoint)│
│    sigma = softplus(raw[:,1])    > 0       (transition width)│
│                                                             │
│  f(t) = sigmoid((time - mu) / sigma)        →  (B, T, 1)  │
│  OUT: phase_weights  in [0,1]               →  (B, T, 1)  │
│    0 = pure growth phase, 1 = pure production phase         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  DifferentiableIntegrator — Pass 2        (model.py)        │
│                                                             │
│  blended_rates = (1 - f(t)) * growth_rates                  │
│                +       f(t)  * prod_rates    →  (B, T, 25) │
│  titer production clamped >= 0 (torch.cat, no in-place)     │
│                                                             │
│  Same ODE as Pass 1, with blended rates                     │
│  OUT: concentrations        (B, T, 25)                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUTS (dict)
  concentrations  (B, T, 25)   predicted metabolite trajectories
  phase_weights   (B, T, 1)    f(t): phase fraction at each timepoint
  growth_rates    (B, T, 25)   growth-phase rates (before blending)
  prod_rates      (B, T, 25)   production-phase rates (before blending)
```

---

## Training Loop

```
TRAINING DATA: 9 reactors per fold, batch_size=4
  ic:      (4, 25)      initial conditions (normalized)
  time:    (4, 40)      normalized time [0, 1]
  params:  (4, 75)      DoE levels + rates + FBA efficiencies
  target:  (4, 40, 25)  ground truth concentration trajectories
  phases:  (4, 40)      ground truth f(t) phase fraction
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  FORWARD PASS                                               │
│  predictions = model(ic, time, params)                      │
│  OUT: concentrations  (4, 40, 25)                           │
│       phase_weights   (4, 40, 1)                            │
│       growth_rates    (4, 40, 25)                           │
│       prod_rates      (4, 40, 25)                           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  LOSS COMPUTATION  (Trainer.compute_loss)                   │
│                                                             │
│  1. Concentration MSE (titer col gets 5x weight)            │
│     IN:  concentrations (4,40,25) vs target (4,40,25)       │
│                                                             │
│  2. Endpoint titer loss (weight 2.0)                        │
│     IN:  concentrations[:,−1,5] vs target[:,−1,5]           │
│                                                             │
│  3. Peak-time alignment (weight 3.0)                        │
│     soft-argmax on titer trajectory to align peak day       │
│                                                             │
│  4. Initial condition constraint (weight 0.1)               │
│     IN:  concentrations[:,0,:] vs ic (4,25)                 │
│                                                             │
│  5. Non-flatness penalty (weight 0.2)                       │
│     penalizes low variance trajectories (avoids trivial flat predictions)│
│                                                             │
│  6. Non-negativity penalty (weight 0.5)                     │
│     penalizes concentrations < 0                            │
│                                                             │
│  7. Phase regression MSE (weight 3.0)                       │
│     IN:  phase_weights (4,40,1) vs phases (4,40)            │
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
│    encoder FC weights, LSTM weights, rate head weights,     │
│    amplitude head, phase head, ODE is differentiable        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  clip_grad_norm_(max_norm=1.0)   prevents exploding gradients
        │
        ▼
  AdamW step: w = w - lr * (gradient + momentum)
  lr = 1e-4 (fine-tuning), weight_decay = 1e-5
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  AFTER EACH EPOCH                                           │
│  Validate on held-out reactor (1 reactor, batch_size=1)     │
│  Metrics computed: F1, MCC, transition MAE, titer within 10%│
│  ReduceLROnPlateau: halve lr if val_loss stagnates (patience=15)│
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

# Train without data_3 specific rates (ablation)
python nn/train.py --no-rates

# Evaluate saved model
python nn/evaluate.py
```
