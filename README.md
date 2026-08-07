# EN PRIMEUR

A fast neural surrogate of a mechanistic CHO bioreactor model.

> **Status:** research prototype (end of internship phase). Trained on synthetic
> data only, treat as a surrogate of the mechanistic simulator, not a
> wet-lab-validated model. Start with [`nn/RESULTS.md`](nn/RESULTS.md) for
> results and handoff notes, and [`history.md`](history.md) for how the project
> evolved.

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

Two models, differing in whether the ODE is embedded in the network:

- **Pure NN** (`model.py`, `nn_baseline.py`) — the primary/headline model.
  Predicts next-day concentrations directly, with no ODE anywhere. It is the
  control that shows the ODE is not what makes the prediction accurate, and it is
  the fastest, since inference is a single network pass.
- **Hybrid neural-ODE** (`model_primeur.py`) — the network predicts per-cell
  fluxes and a fixed closed-form ODE step integrates them to the next day. Mass
  balance holds by construction, and the per-day fluxes are an explicit output.
  The ODE runs in the forward pass, so it is slower than the pure NN (about 2x at
  inference) but physically consistent.

|                     | pure NN                 | hybrid neural-ODE            |
| ------------------- | ----------------------- | ---------------------------- |
| network predicts    | concentrations          | per-cell fluxes              |
| ODE in forward pass | no                      | yes (`closed_form_step`)     |
| mass balance        | not enforced            | guaranteed by construction   |
| fluxes available    | only by back-solving    | direct output                |
| inference cost      | one NN pass (fastest)   | NN pass + ODE step (~2x)     |

A `closed_form_step` (exact analytic one-day solution of the linear per-day ODE,
matching the numerical solver to ~1e-9) makes the mechanistic step cheap and
differentiable, and underlies the hybrid forward pass.

A physics-informed (PINN) variant — concentrations predicted directly with the
ODE as a soft training penalty, solver-free at inference — was also explored and
is archived under [`nn/scripts_past/pinn_baseline.py`](nn/scripts_past/pinn_baseline.py).

### How a forecast is made

The model is a one-day-ahead predictor applied **autoregressively**, and it is
trained differently from how a whole run is forecast:

- **Training is teacher-forced.** Every training window is real (synthetic-truth)
  data and the target is the real next day (`nn_baseline.py`). The model only ever
  predicts one step ahead from ground truth.
- **Inference is a free-running rollout.** The model is seeded with only the first
  `SEQ_LEN` days (6 by default) of a run, then forecasts day by day, feeding each
  prediction back in as the input for the next step (`evaluate.rollout`).

The key point: **past the seed, the model gets no real data at inference.** Days
6-12 are predicted entirely from the model's own previous outputs plus the fixed
operating conditions, with no ground-truth correction along the way. The reported
R2 is measured on this autoregressive rollout (the hard test), not on one-step
prediction from ground truth.

A free-running *training* variant, which also rolls out on its own predictions
during training (to test whether that narrows the train/inference gap), lives in
[`nn/nn_freerun.py`](nn/nn_freerun.py).

## Key results (see `nn/RESULTS.md` for detail)

- **Accuracy:** a physics-free NN reproduces all 8 tracked variables in absolute
  units to within a few percent (peak ratios 0.99-1.05). Low correlations are
  flat metabolites, not errors.
- **Speed:** ~3,000x faster than the numerical ODE solver (CPU, same machine).
  The exact closed-form step is a separate ~70x exact speedup.

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
  machinery reused by the hybrid and the speed benchmark.

### Training & evaluation — the main experiment

- **`nn_baseline.py`** — trains the pure NN on the synthetic data, forecasts each
  held-out run autoregressively, scores R2 / rho / MAE, and saves the figures. The
  main deliverable script.
- **`evaluate.py`** — shared evaluation helpers, most importantly the autoregressive
  `rollout` that both training scripts use at inference to forecast a run from its
  6-day seed (no real data past the seed).
- **`nn_freerun.py`** — the free-running training variant: same architecture and
  inference as `nn_baseline`, but trained by rolling out on its own predictions
  (not teacher-forced), to test exposure bias. Prints the same R2 table for a
  head-to-head comparison.

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
- **`scripts_past/`** — archived fed-batch, real-data, physics-informed (PINN),
  and deprecated scripts.
- **`og_code/`** (repo root) — original mechanistic (MATLAB) reference.
- **`archive/`** (repo root) — earlier-phase dev logs, real-data notebooks, and
  stale job scripts, kept for reference, not part of the current pipeline.
- **`figures/past/`** — superseded figures from earlier runs; the current figures
  live directly in `figures/`.

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

speed_benchmark.py
        └─ uses model.py, model_primeur.py, generate_synthetic_ode.py, device_utils.py
```

Import graph (who imports whom; the three at the bottom have no local imports):

- `nn_baseline.py`   -> `model`, `evaluate`, `generate_synthetic_ode`, `device_utils`
- `evaluate.py`      -> `model`, `model_primeur`, `device_utils`
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
- **Continuous-time PINN.** An archived physics-informed baseline
  (`scripts_past/pinn_baseline.py`) uses the integrated-form penalty (exact
  one-day step). A continuous-time model would allow the differential form
  (`dx/dt = f` via autograd) and finer physics constraints.
- **Adaptive surrogate.** Detect out-of-domain inputs, simulate only there,
  fine-tune incrementally; the physics penalty carries over once real data
  arrives.

Done this phase (see `nn/RESULTS.md`): pure-NN accuracy baseline, a hybrid
neural-ODE variant, and a hardware-isolated speed benchmark that times the ODE on
the same device as the NN (torch closed-form + torchdiffeq). A physics-informed
(PINN) baseline was explored and archived under `scripts_past/`.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
