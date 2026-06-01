# COSMIC dFBA Neural Network Surrogate Model

A PyTorch surrogate model for predicting bioreactor phase transitions and metabolite trajectories, trained on experimental dFBA data.

## Files

| File | Purpose |
|------|---------|
| `nn/model.py` | Neural network architecture (encoder, LSTM decoder, ODE integrator, phase head) |
| `nn/train.py` | Training script: pre-training on synthetic data, LOO cross-validation fine-tuning |
| `nn/evaluate.py` | Evaluation metrics and plots vs. paper benchmarks |
| `nn/utils.py` | Data loading, normalization, and diagnostic utilities |
| `nn/data/` | Experimental data (data_1: DoE levels, data_2: trajectories, data_3: rates, data_4: FBA efficiencies) |

## Model Architecture

**Encoder** — FC layers that compress initial conditions + DoE parameters into a 64-dimensional latent vector.

**LSTM Decoder** — Unrolls the latent state over time to produce growth and production rate predictions at each timestep.

**Amplitude Scalar** — Softplus head on the latent state that scales rate magnitude per reactor, separating shape (Tanh) from scale.

**DifferentiableIntegrator** — Implicit Euler ODE that integrates blended rates into concentration trajectories. Washout term (F_NORM) applies to metabolites only; cells and titer use no washout (epsilon = 0).

**PhaseTransitionHead** — Reads the Pass 1 concentration trajectory and outputs f(t), a sigmoid-shaped phase fraction from 0 (growth) to 1 (production). Used to blend growth and production rates in Pass 2.

## Training Loop

```
                    TRAINING DATA
                (9 reactors per fold, batch_size=4)
                          │
                          ▼
      ┌─────────────────────────────────────────┐
      │           Trainer.train_epoch()         │
      │           (train.py)                    │
      └─────────────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          │          FOR each batch:       │
          │                               │
          ▼                               ▼
    Extract batch                 zero_grad() clears
    ├─ ic:     (4, 25)            gradients from last step
    ├─ time:   (4, 40)
    ├─ params: (4, 56)
    └─ target: (4, 40, 25)
          │
          ▼
    FORWARD PASS
    predictions = model(ic, time, params)
    │
    └─ CosmicNNSurrogateLSTM (model.py)
       ├─ encoder(ic, params)  →  latent (4, 64)
       └─ decoder(latent, time, ic, doe_params)
          ├─ LSTM processes time steps
          ├─ RatePredictionHead outputs growth + production rates
          ├─ ODE Pass 1 (f=0, growth rates only)
          ├─ PhaseTransitionHead reads Pass 1 concentrations → f(t)
          ├─ ODE Pass 2 (rates blended by f(t))
          └─ Returns: concentrations, phase_weights, growth_rates, prod_rates
          │
          ▼
    LOSS COMPUTATION
    loss = compute_loss(predictions, target, ic, phases)
    │
    └─ Fused PINN loss (8 terms)
       ├─ Concentration MSE (titer 5x weight)
       ├─ Endpoint titer loss
       ├─ Peak-time alignment loss
       ├─ Initial condition constraint
       ├─ Non-negativity penalty
       ├─ Concentration smoothness
       ├─ Phase regression MSE (weight 3.0)
       └─ Rate + phase smoothness
          │
          ▼
    BACKWARD PASS
    loss.backward()
    └─ Computes gradients for all weights:
       encoder, LSTM, rate heads, phase head, ODE interactions
          │
          ▼
    GRADIENT CLIPPING
    clip_grad_norm_(max_norm=1.0)
          │
          ▼
    OPTIMIZER STEP
    AdamW: w ← w - lr × (gradient + momentum)
    └─ lr = 1e-4 (fine-tuning), weight_decay = 1e-5

    ┌──────────────────────────────────────────┐
    │  After each epoch:                       │
    │  - Validate on held-out reactor (LOO)    │
    │  - Check early stopping (patience=80)    │
    │  - ReduceLROnPlateau if no improvement   │
    └──────────────────────────────────────────┘
```

## Training Strategy

**Phase 1 — Synthetic pre-training** (if `synthetic_training.npz` exists): trains on simulated data normalized to match the real dataset scale. Runs for 500 epochs with no early stopping.

**Phase 2 — Leave-one-out fine-tuning**: with only 10 reactors, each reactor is held out once as the test set. The model is reset to pre-trained weights for each fold and fine-tuned on the remaining 9. LOO metrics are averaged across all 10 folds.

**Final model**: after LOO, the model is re-trained on all 10 reactors and saved to `improved_model.pt`.

## Running

```bash
# Train (with synthetic pre-training if available)
python nn/train.py

# Train without synthetic pre-training
python nn/train.py --no-synthetic

# Train permutation baseline (shuffled inputs vs outputs)
python nn/train.py --shuffle

# Evaluate saved model
python nn/evaluate.py
```
