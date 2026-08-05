# en Primeur

A fast neural surrogate of a mechanistic CHO bioreactor model.

> **Status:** research prototype (end of internship phase). Trained on synthetic
> data only, treat as a surrogate of the mechanistic simulator, not a
> wet-lab-validated model. Start with [`nn/RESULTS.md`](nn/RESULTS.md) for
> results and handoff notes.

*En primeur* is the practice of judging a wine before it is bottled, tasting the
young vintage to forecast the finished product. This project does the same for a
bioreactor run: predict the full trajectory of a cell culture (cell density,
metabolites, antibody titer) from its early days and its operating conditions,
without paying the cost of solving the underlying mechanistic model each time.

## What it does

The reference is a mechanistic CHO model (COSMIC dynamic flux-balance analysis):
a system of ODEs that, given fluxes and feed/perfusion conditions, produces the
daily concentration trajectory of 25 components. It is accurate but slow to solve.

en Primeur replaces the solver with a small neural network that predicts the
trajectory directly. Trained on synthetic data from the mechanistic model, it
reproduces the simulation in absolute units to within a few percent while running
~3,000x faster (see `nn/RESULTS.md`).

Three model families, in order of how much physics they embed:

- **Pure NN** (`model.py`, `nn_baseline.py`): predicts next-day concentrations
  directly. No ODE at all. The control that shows the ODE is not what makes the
  model accurate.
- **Hybrid neural-ODE** (`model_primeur.py`, `model_stripped.py`): the network
  predicts per-cell fluxes, a fixed closed-form ODE step integrates them to the
  next day. Mass balance holds by construction. The ODE is in the forward pass.
- **Physics-informed (PINN)** (`pinn_baseline.py`): predicts concentrations
  directly, with the ODE used only as a soft training penalty (a lambda-weighted
  residual). The solver is absent at inference, so it keeps the speedup while
  staying mechanistically informed during training.

A `closed_form_step` (exact analytic one-day solution of the linear per-day ODE,
matching the numerical solver to ~1e-9) makes the mechanistic step cheap and
differentiable, and underlies both the hybrid forward pass and the PINN penalty.

## Key results (see `nn/RESULTS.md` for detail)

- **Accuracy:** a physics-free NN reproduces all 8 tracked variables in absolute
  units to within a few percent (peak ratios 0.99-1.05). Low correlations are
  flat metabolites, not errors.
- **Speed:** ~3,000x faster than the numerical ODE solver (CPU, same machine).
  The exact closed-form step is a separate ~70x exact speedup.
- **Physics-informed:** a small physics weight (lambda ~= 0.1) measurably steadies
  the cell-density rollout (correlation 0.79 -> 0.89), with solver-free inference.

**Scope:** trained on synthetic data, so the model is a surrogate of the
simulator, not yet validated against wet-lab reactors.

## Structure

The `nn/` directory is condensed to the synthetic-only pure-NN pipeline. Older
explorations (fed-batch, real-reactor training/LORO, deprecated experiments) are
archived under `nn/scripts_past/`. The essential files, grouped by function:

### Data generation — build the synthetic data the NN learns from

- **`generate_synthetic_ode.py`** — the mechanistic simulator. Integrates the
  per-day ODE (feed + washout + cell uptake/secretion) to produce daily
  trajectories, and writes `synthetic_ode.npz`. This is the ground truth the NN is
  trained to reproduce.
- **`compute_aa_scales.py`** — sizes each amino acid's perfusion feed to the minimum
  that keeps it non-negative (no clamp). Writes `data/aa_scales.npy`, which the
  generator reads. Run once, before generating.

### Models — the network definitions

- **`model.py`** — the pure NN (`NextDayPredictor`): 1D CNN -> attention -> head,
  predicts next-day concentrations directly. The primary/headline model.
- **`model_primeur.py`** — the hybrid neural-ODE decoder (`FluxDecoder`) plus
  `closed_form_step` / `ode_step` (the exact one-day ODE step). Supplies the ODE
  machinery reused by the hybrid, the PINN penalty, and the speed benchmark.

### Training & evaluation — the main experiment

- **`nn_baseline.py`** — trains the pure NN on the synthetic data, forecasts each
  held-out run autoregressively, scores R2 / rho / MAE, and saves the figures. The
  main deliverable script.
- **`evaluate.py`** — shared evaluation helpers, most importantly the autoregressive
  `rollout` that `nn_baseline` uses to forecast a run from its 6-day seed.
- **`pinn_baseline.py`** — the physics-informed variant: same prediction task, but
  the ODE is added as a soft training penalty (solver-free at inference).

### Benchmarking

- **`speed_benchmark.py`** — times NN inference against solving the ODE (numerical,
  closed-form, torchdiffeq) on the same device, plus the hybrid variants.

### Utilities & jobs

- **`device_utils.py`** — `pick_device()`: use GPU if available, else CPU. Imported
  by everything that runs a model.
- **`run_nnbaseline.sbatch`** — SLURM job for `nn_baseline.py` (regenerates data if missing).
- **`run_speed_gpu.sbatch`** — SLURM job for `speed_benchmark.py` on a GPU.

### Data & docs

- **`data/`** — `data_1..4` (DoE, trajectories, rates, FBA efficiencies) + `aa_scales.npy`.
- **`RESULTS.md`** — results + handoff notes (read this first).
- **`scripts_past/`** — archived fed-batch, real-data, and deprecated scripts.
- **`og_code/`** (repo root) — original mechanistic (MATLAB) reference.

### How they fit together

The pipeline runs top to bottom; arrows are "produces / feeds into":

```
compute_aa_scales.py  ─►  data/aa_scales.npy
                                 │
generate_synthetic_ode.py  ◄─────┘   (reads the feed scales)
        │
        ▼
   synthetic_ode.npz
        │
        ▼
   nn_baseline.py  ─►  figures + R2/rho/MAE  (summarized in RESULTS.md)
        ├─ uses model.py         (the network)
        ├─ uses evaluate.py      (autoregressive rollout)
        └─ uses device_utils.py  (GPU/CPU)

pinn_baseline.py  and  speed_benchmark.py
        └─ use model.py, model_primeur.py, generate_synthetic_ode.py, device_utils.py
```

Import graph (who imports whom; the three at the bottom have no local imports):

- `nn_baseline.py`   -> `model`, `evaluate`, `generate_synthetic_ode`, `device_utils`
- `evaluate.py`      -> `model`, `model_primeur`, `device_utils`
- `pinn_baseline.py` -> `model`, `model_primeur`, `generate_synthetic_ode`, `device_utils`
- `speed_benchmark.py` -> `model`, `model_primeur`, `generate_synthetic_ode`, `device_utils`
- `compute_aa_scales.py` -> `generate_synthetic_ode`
- `model.py`, `model_primeur.py`, `device_utils.py` -> (leaf modules, no local imports)

Data note: `data_2.csv` is the measured trajectories (normalized per reactor);
`data_1.csv` the DoE conditions; `data_3.csv` phase-specific rates. Real data is
**not** used for training or evaluation, the model is trained and scored on the
synthetic ODE runs alone (see `RESULTS.md`).

## Running

```bash
# generate synthetic data
python nn/generate_synthetic_ode.py --n-extra 3000 --fast --output nn/synthetic_ode.npz

# pure-NN accuracy baseline (all 8 features)
python nn/nn_baseline.py --data nn/synthetic_ode.npz --all-features

# physics-informed surrogate, sweep the physics weight
python nn/pinn_baseline.py --data nn/synthetic_ode.npz --sweep 0 0.03 0.1 0.3 1.0

# speed benchmark (add --n 20000 in a GPU allocation)
python nn/speed_benchmark.py --n 5000
```

Compute-heavy runs go through SLURM (`run_*.sbatch`). Compute nodes have no
internet, so download data and `pip install` on a login node first.

## Next steps

- **Real data.** The model is a simulator surrogate until validated against
  wet-lab reactors. `data_2` is normalized per reactor, so cross-reactor
  magnitude is unrecoverable without the original scale factors; a public
  perfusion dataset with absolute concentrations (or the scale factors) would
  unlock this.
- **Continuous-time PINN.** The current physics penalty is integrated form (exact
  one-day step). A continuous-time model would allow the differential form
  (`dx/dt = f` via autograd) and finer physics constraints.
- **Adaptive surrogate.** Detect out-of-domain inputs, simulate only there,
  fine-tune incrementally; the physics penalty carries over once real data
  arrives.

Done this phase (see `nn/RESULTS.md`): pure-NN accuracy baseline, hybrid and
physics-informed (PINN) surrogates, and a hardware-isolated speed benchmark that
times the ODE on the same device as the NN (torch closed-form + torchdiffeq).
