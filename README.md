# en Primeur

A fast, physics-informed neural surrogate for CHO bioreactor dynamics.

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

```
nn/
  RESULTS.md            end-of-phase results + handoff notes (read this first)

  model.py              pure NN (NextDayPredictor)
  model_primeur.py      hybrid neural-ODE decoder; closed_form_step / ode_step
  model_stripped.py     low-capacity hybrid (1 conv layer)
  model_fedbatch.py     fed-batch variant of the hybrid ODE step

  nn_baseline.py        pure-NN accuracy experiment (no ODE)
  pinn_baseline.py      physics-informed surrogate (ODE as training penalty only)
  speed_benchmark.py    NN vs mechanistic solver timing (CPU + GPU, torchdiffeq)

  generate_synthetic_ode.py     mechanistic ODE data generator (closed-form / RK45)
  train_real.py / loro_real.py  real-reactor training + leakage-free LORO
  real_data.py          denormalize data_2 onto the ODE scale
  device_utils.py       CPU/GPU selection

  data/                 data_1..4 (DoE, trajectories, rates, FBA efficiencies)

run_*.sbatch / run_*_slurm.sh    SLURM job scripts
og_code/                original mechanistic (MATLAB) reference
```

Data: `data_2.csv` is the measured trajectories (normalized per reactor);
`data_1.csv` the DoE conditions; `data_3.csv` phase-specific rates. The
25-component layout is documented in earlier git history.

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
- **GPU-native solver.** `speed_benchmark.py` now times the ODE on the same
  device as the NN (torch closed-form; torchdiffeq if installed) for a fair,
  hardware-isolated comparison.
- **Adaptive surrogate.** Detect out-of-domain inputs, simulate only there,
  fine-tune incrementally; the physics penalty carries over once real data
  arrives.
