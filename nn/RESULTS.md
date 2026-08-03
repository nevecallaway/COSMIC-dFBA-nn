# COSMIC-dFBA-nn: results summary

End-of-internship results and handoff notes. Two questions the group asked:
does a plain NN need the ODE, and is the NN actually faster than solving the
mechanistic equations?

Scripts: `nn_baseline.py` (accuracy), `speed_benchmark.py` (speed),
`run_speed_gpu.sbatch` / `run_nnbaseline_seqlen.sbatch` (SLURM).

---

## 1. Accuracy: the NN reproduces the physics without the ODE

Setup: pure `NextDayPredictor` (window -> next-day concentrations directly, NO
ODE), trained on **synthetic** data, **window-level holdout** (a few windows per
reactor, reactors NOT held out; in-distribution). Scored in absolute units on all
8 variables against the ODE simulation. seq_len = 6.

R2 = coefficient of determination (1 - SS_res/SS_tot, mean as baseline), pooled
over all forecast points per feature. Unitless regression metric (per Kimberly).
It shares rho's flat-signal caveat: over a near-constant window SS_tot ~ 0, so R2
is not meaningful and is marked n/a. Numbers below are AFTER the amino-acid
data-gen fix (see note beneath the table).

| feature | R2 | peak ratio | norm MAE | rho | fcast range |
|---|---|---|---|---|---|
| Cell Density | 0.99 | 1.01 | 0.036 | 0.77 | 0.31 |
| Cell Size | 0.98 | 0.96 | 0.024 | 1.00 | 0.51 |
| Titer | 0.97 | 0.95 | 0.061 | 0.98 | 0.72 |
| Glucose | 1.00 | 1.04 | 0.040 | 0.93 | 0.23 |
| Glycine | 0.97 | 1.05 | 0.062 | 0.59 | 0.18 |
| Glutamine | n/a | 1.00 | 0.025 | 0.77 | 0.06 |
| Asparagine | n/a | 0.99 | 0.028 | 0.90 | 0.08 |
| Serine | n/a | 1.01 | 0.023 | 0.51 | 0.09 |

Five variables score real R2 0.97-1.00. The remaining three amino acids read n/a
only because their FORECAST-window range is just under 0.1, not because they are
flat: their rho jumped to 0.77-0.90 (from ~0.2 noise before the fix), so the model
IS tracking their forecast-window trend.

**Amino-acid data-gen fix (important):** earlier runs showed glutamine/serine/
asparagine as perfectly flat lines with rho ~0.2. That was a generator artifact:
the AA initial pool was tied to a 210x-inflated perfusion feed, pinning each AA at
its feed level. Decoupling the initial pool (realistic DMEM) from the feed (enriched
only as much as each AA needs to stay non-negative, no clamp) restored the real
rise-then-deplete dynamics seen in data_2. The dynamics are now FRONT-LOADED (most
of the swing is in days 0-6, the model's input window), so the forecast window
(days 6-12) is milder -> real but small range. A shorter seed window would move
more AA dynamics into the forecast window if we want those R2 scored.

**Takeaway:** R2 0.97-1.00 on every scoreable variable, peak ratios 0.95-1.05, tiny
MAE everywhere. A physics-free NN reproduces the dynamic variables to within a few
percent and explains 97-100% of their variance, and the amino acids are now genuine
(non-flat) targets it tracks. **The ODE is not what makes the model accurate.** Two
honest notes: (1) R2 is pooled across reactors, so it includes cross-reactor level
spread (getting each reactor's level right already explains much of the variance);
per-reactor trajectory-shape R2 is a harder test we can add. (2) Single seed;
digits move run to run, so report mean +/- std before quoting a hard number.

Figures: `nn_baseline_r2_<Feature>.png` (per-feature 2x5 grids), `nn_baseline_parity.png`
(predicted-vs-true diagonal, all variables). Hero slides = Titer, Glucose, Cell Density.

**Scope caveat:** synthetic, in-distribution. This shows the NN can REPRODUCE the
ODE, not that it transfers to real reactors.

---

## 2. Speed: ~3,000x faster than solving the ODE

Setup: 5,000 reactors x 13-day trajectories, matched network size so differences
isolate the ODE step. CPU (the mechanistic solver is scipy, CPU-only).

| method | ms/reactor | speedup vs numerical solver |
|---|---|---|
| numerical ODE (solve_ivp) | 37.9 | 1x (baseline) |
| closed-form ODE | 0.51 | ~70x |
| pure NN (no ODE) | 0.013 | ~3,000x |
| hybrid NN (closed-form ODE in forward) | 0.014 | ~2,600x |
| hybrid NN (50-substep Euler in forward) | 0.044 | ~870x |

**Takeaways:**
- NN surrogate is ~3,000x faster than the numerical solver (the "why an NN" number).
- Closed-form vs numerical is a separate ~70x EXACT speedup (matches to ~1e-9),
  same equations solved analytically. Consistent across runs (64x/71x/74x). But it
  only works because our per-day ODE is linear; the original full COSMIC-dFBA
  (flux-balance LP each step) has no closed form -> the real reason to learn a surrogate.
- The closed-form ODE in the forward pass is nearly free (~1.1x); only the
  50-substep Euler integrator pays a real penalty (~3.5x).

**GPU (A100) footnote:** the NN is effectively free, faster than the timer can
resolve, so the speedup is 4+ orders of magnitude. Quote the CPU ~3,000x as the
conservative, measurable figure; the GPU widens the lead and cannot speed up the
CPU-only solver.

### 2b. Same-device comparison (the honest isolation)

The ~3,000x above compares scipy (CPU, one reactor at a time) against a batched
NN, which conflates the solver with the hardware. `speed_benchmark.py` now times
every method the RIGHT way: warm up once, then loop inference for a fixed 45s
wall-clock budget and amortize (per-reactor = total / (reps * N)). A single GPU
pass is tens of microseconds, dominated by kernel-launch overhead, not compute, so
timing one pass makes two fast methods report the SAME number (the timer floor).
Looping thousands of reps pushes the overhead below the real compute, and it also
runs the ODE on the SAME device as the NN. A100, n=20,000, 45s/method (9,000+ reps
per fast method):

| method | ms/reactor | vs solve_ivp (CPU) | vs torchdiffeq (GPU) |
|---|---|---|---|
| numerical ODE (solve_ivp, CPU) | 25.06 | 1x | - |
| closed-form ODE (numpy, CPU) | 0.35 | 72x | 0.02x |
| closed-form ODE (torch, cuda) | 0.0005 | 47,000x | 23x |
| torchdiffeq odeint (cuda) | 0.012 | 2,000x | 1x (baseline) |
| pure NN (cuda) | 0.0002 | 103,000x | 50x |
| hybrid NN (closed-form, cuda) | 0.0006 | 42,000x | 20x |
| hybrid NN (50-substep Euler, cuda) | 0.0049 | 5,100x | 2.5x |

Two baselines on purpose: "vs solve_ivp" is the incumbent (scipy, CPU-only, no GPU
version exists); "vs torchdiffeq" is a general differentiable solver on the NN's
device, which isolates the algorithm from the hardware. Read the torchdiffeq
column as the honest surrogate-vs-solver number.

**The honest findings:**
- Properly amortized, the pure NN (0.0002 ms) and the exact closed-form torch ODE
  (0.0005 ms) NO LONGER tie: the NN is ~2.4x faster. The earlier "identical" reading
  was purely the single-pass timer floor, exactly the artifact to distrust.
- Against a FAIR GPU solver (torchdiffeq), the NN is ~50x faster, not 100,000x. The
  100,000x vs scipy mostly reflects CPU-vs-GPU hardware. torchdiffeq sits at ~2,000x
  vs scipy and ~23x slower than the closed form (it still takes adaptive steps).
- On THIS linear per-day ODE the exact closed form is within ~2.4x of the NN, so the
  surrogate barely wins. The durable reason to learn a surrogate is mechanistic
  models with NO closed form (the full flux-balance COSMIC-dFBA), where the relevant
  penalty is torchdiffeq's ~2,000x, not the closed form's.

Do NOT combine speed and accuracy in one table (different units/comparisons). The
bridge sentence: the NN matches the ODE's accuracy (few percent) and, on a GPU, its
speed too; on this simplified model it neither gains nor loses much, its value is
for models that cannot be solved in closed form.

---

## 3. Window length: opposite trends on real vs synthetic

Swept input window (seq_len 1-5 vs 6) on the pure-NN synthetic baseline.
- **Synthetic (in-distribution):** magnitude error falls monotonically as the
  window grows; 6 days is best on every variable. Very short windows (1-2) are
  unstable. So longer is better here.
- **Real reactors (earlier LORO):** shorter windows HELPED (rho 0.44 -> 0.78 going
  6 -> 2 days).
- **Interpretation:** the optimal window depends on the data regime, not the window
  itself. Shorter regularizes when generalizing from few reactors (real, n=10);
  longer adds useful context when interpolating from many (synthetic).
  (Single-seed; the magnitude trend is robust, the rho trend is noisy.)

---

## Handoff / open items for the next person

1. **Real-data magnitude is unrecoverable as-is.** `data_2` is normalized per
   reactor (each / its own max), so cross-reactor productivity is mathematically
   gone; our denormalization uses the ODE peak as scale, so real-data magnitude
   metrics validate against the ODE, not ground truth. FIX: get the per-reactor
   scale factors (or absolute final titers) from Sarat, then swap them into
   `real_data.denormalize_data2`.
2. **Normalization should be done after combining all data**, not per reactor, so
   scaling is an invertible post-processing step (group's suggestion).
3. **Residual/ODE-relaxation knob** (`--residual-weight`, `--residual-l2` in
   train_real): letting the model bend ~30% off the ODE lifted real-data rho
   0.78 -> 0.90 by fixing the day-8 peak timing; L2 penalty added to curb the
   resulting overshoot, not fully swept.
4. **Perfusion validation data:** no clean public perfusion dataset with absolute
   concentrations + conditions. Best target = pseudo-perfusion modeling paper
   PMC12687770 (data on request). Fed-batch has the public Golzarijalal dataset
   (Figshare 10.26188/28943096).
5. **Aspiration:** experimental validation on real perfusion reactors.
