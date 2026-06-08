# COSMIC dFBA Neural Network Surrogate Model

A PyTorch surrogate model for predicting bioreactor phase transitions and metabolite trajectories, trained on experimental dFBA data from 10 perfusion reactors.

The primary prediction goal is transition timing for root cause analysis (RCA): predicting when and how a cell line switches from growth phase to production phase, not just final titer. The loss is weighted accordingly, favoring phase transition accuracy over titer. Phase is predicted at each time step from the current concentration vector (not from time), so the model responds to the actual metabolic state of the reactor rather than the clock. Transition day is inferred as the first time step where predicted f(t) crosses 0.5. Evaluation includes phase AUC, transition MAE, and standard classification metrics.

**Latest LOO results (FC model, DoE-only inputs, seed=42):** f(t) accuracy +/-0.1: 83.8% (paper: 72.3%) | MCC: 0.906 (paper: 0.454) | Transition MAE: 1.40d | Phase AUC MAE: 0.33d | Shuffled baseline: 2.32d

LSTM variant (same inputs, time-varying rates): LOO MAE 1.46d, f(t) 80.0% -- worse on all metrics. Constant-rate FC is the final model.

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
| `nn/data/data_3.csv` | Phase-specific metabolic rates -- used as v_max bounds on rate heads (not encoder input) |
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

data_3 specific rates are used to set per-component v_max ceilings on the rate heads (cross-reactor mean of absolute rates per phase, normalised so the highest-activity component = 1.0). They are not encoder inputs -- the encoder still takes only IC and DoE. data_4 FBA objective efficiencies are not used.

---

## Model Architecture

The architecture is deliberately minimal. With 10 reactors and 3 DoE inputs the problem is low-dimensional, so simpler generalizes better.

Key design decisions:
- Rates are constant per phase (biologically appropriate -- dFBA rates are phase-constant), not time-varying.
- Rate head magnitude is bounded by data_3 maxima (v_max buffers), not freely learned.
- Phase is predicted from the current concentration vector at each time step, not from time. The metabolic state drives the transition, not the clock.
- The forward pass is sequential: phase and integration alternate step by step.

```mermaid
flowchart TD
    IC["initial_conditions (B, 25)\nnormalized concentrations at t=0"]
    TP["time_points (B, T)\nnormalized time in 0..1"]
    DOE["parameters (B, 3)\nDoE coded levels: O2, AAs, Glc"]
    VM["v_max buffers (25 each)\nfrom data_3 cross-reactor mean\nnormalised: max component = 1.0"]

    IC & DOE --> CAT["cat(IC, params) → (B, 28)"]

    subgraph ENC["DynamicsEncoder"]
        direction TB
        E1["Linear(28→64) + ReLU"]
        E2["Linear(64→32)"]
        E1 --> E2
    end
    CAT --> E1
    E2 --> LAT["latent (B, 32)"]

    subgraph RH["Rate Heads  (constant per phase)"]
        direction TB
        GR["growth_head: Linear(32→25) + Tanh"]
        PR["prod_head:   Linear(32→25) + Tanh"]
    end
    LAT --> GR & PR
    GR & VM --> GRV["growth × v_max_growth → (B, 25)"]
    PR & VM --> PRV["prod × v_max_prod   → (B, 25)"]

    subgraph LOOP["Sequential forward loop  (t = 0 .. T-1)"]
        direction TB
        PH["PhaseTransitionHead\nLinear(25→1) + Sigmoid\nf_t = sigmoid(W · c_t)  →  (B, 1)"]
        BL["blend: v_t = (1−f_t)·growth + f_t·prod  →  (B, 25)"]
        ST["integrator.step(c_t, v_t, t, t+1, DOE)\nexplicit Euler, paper Eq. 2\nc_{t+1} = c_t + dc/dt · dt"]
        PH --> BL --> ST --> PH
    end
    IC --> LOOP
    GRV & PRV --> LOOP

    subgraph ODE["integrator.step  (one step)"]
        direction TB
        IE["dc/dt = v·C1 + F·(c_in − c_t)·η\n\nη = 0: cells (idx 0-1) always\nη = 0 before day 8, 1 after: titer (idx 5)\nη = 1: all other metabolites\nF_NORM = 13.0,  F_TITER = 1.0\nc_in from DoE coded levels"]
    end
    DOE --> ODE

    LOOP --> CONC["concentrations (B, T, 25)"]
    LOOP --> PW["phase_weights (B, T, 1)\ntransition day inferred where f crosses 0.5"]
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
│  Sequential loop: for each t, phase from c_t, then step     │
│  OUT: concentrations  (4, 13, 25)                           │
│       phase_weights   (4, 13, 1)                            │
│       growth_rates    (4, 13, 25)  constant across T        │
│       prod_rates      (4, 13, 25)  constant across T        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  LOSS COMPUTATION  (Trainer.compute_loss)                   │
│                                                             │
│   1. Concentration MSE (weight 1.0)                         │
│        cell density (idx 0): 3x weight -- primary RCA target│
│        titer (idx 5): 2x weight                             │
│      IN:  concentrations (4,13,25) vs target (4,13,25)      │
│                                                             │
│   2. Endpoint titer loss (weight 0.5)                       │
│      IN:  concentrations[:,−1,5] vs target[:,−1,5]          │
│                                                             │
│   3. Peak-time alignment (weight 1.0)                       │
│      soft-argmax on titer trajectory to align peak day      │
│                                                             │
│   4. Initial condition constraint (weight 0.1)              │
│      IN:  concentrations[:,0,:] vs ic (4,25)                │
│                                                             │
│   5. Non-flatness penalty (weight 0.2)                      │
│      penalizes low variance trajectories                    │
│                                                             │
│   6. Non-negativity penalty (weight 0.5)                    │
│      penalizes concentrations < 0                           │
│                                                             │
│   7. Concentration smoothness (weight 0.1)                  │
│      penalizes step-to-step jumps in predicted trajectories │
│                                                             │
│   8. Phase regression MSE (weight 5.0)                      │
│      IN:  phase_weights (4,13,1) vs phases (4,13)           │
│                                                             │
│   9. Rate smoothness (weight 0.1)                           │
│      penalizes step-to-step changes in blended_rates        │
│                                                             │
│  10. Phase smoothness (weight 0.05)                         │
│      penalizes step-to-step changes in f(t)                 │
│                                                             │
│  11. Rate magnitude (weight 0.01)                           │
│      L1 regularization on growth and production rates       │
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
│    phase head (Linear 25→1), ODE is differentiable          │
│    v_max buffers are fixed (not trained)                     │
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
