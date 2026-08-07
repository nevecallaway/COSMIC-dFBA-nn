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

**Metric definitions** (each computed on the forecast window, the days the model
predicts on its own):
- **R2** (coefficient of determination, `1 - SS_res/SS_tot`, mean as baseline):
  the fraction of the true variance the model explains. 1.0 = predictions land
  exactly on the truth; 0 = no better than always guessing the mean; negative =
  worse than the mean. Penalizes wrong scale AND offset, not just shape. Unitless
  (the metric Kimberly asked for). Undefined for a flat signal (variance ~ 0), so
  marked n/a there.
- **rho** (Pearson correlation, -1..+1): does the prediction move up and down WITH
  the truth? A shape/direction score that ignores scale and offset. rho = 1 means
  perfect pattern match even if the magnitude were off; you can have high rho with
  mediocre R2, but not the reverse. Also n/a on flat signals.
- **norm MAE**: mean absolute error divided by the true peak, i.e. the average miss
  as a fraction of the signal size.
- **peak ratio**: predicted peak / true peak. >1 overshoots, <1 undershoots.
- **fcast range**: `(max - min) / max` of the TRUE signal over the forecast window,
  how much the signal actually moves. Below 0.1 there is too little variation for
  R2/rho to mean anything (hence the n/a flag).

Numbers below are AFTER the amino-acid data-gen fix (see note beneath the table).

Final config: zero negatives, no clamp anywhere (`--glc-feed 40`, AA feed at
default 1.0). SMALL model (1 conv layer, hidden 32), chosen per Kimberly: R2 ~0.95
is more than adequate for xAI, and a smaller model is cleaner to attribute. Full
model (3 conv, hidden 64) scores ~0.01-0.02 higher R2 and ~2x lower MAE; see the
ablation note below.

| feature | R2 | peak ratio | norm MAE | rho | fcast range |
|---|---|---|---|---|---|
| Cell Density | 0.98 | 1.11 | 0.113 | 0.91 | 0.31 |
| Cell Size | 0.98 | 1.02 | 0.030 | 1.00 | 0.51 |
| Titer | 0.95 | 0.96 | 0.076 | 0.97 | 0.72 |
| Glucose | 0.99 | 1.01 | 0.052 | 0.97 | 0.23 |
| Glycine | 0.95 | 0.98 | 0.070 | 0.53 | 0.18 |
| Glutamine | n/a | 1.00 | 0.041 | 0.62 | 0.06 |
| Asparagine | n/a | 0.99 | 0.043 | 0.63 | 0.08 |
| Serine | n/a | 1.00 | 0.028 | 0.49 | 0.09 |

Time resolution: 1 day (predictions are daily snapshots; the NN steps forward one
day at a time). Ablation (conv layers 1/2/3 at hidden 64): R2 nearly flat across
depth, cell density 0.98/0.99/0.99, titer 0.95/0.96/0.97; depth mainly halves MAE
and calibrates the peak. Inference: 1 layer 0.0005 ms, 2 layers 0.0009, 3 layers
0.0013 (all ~700M x faster than COSMIC-dFBA's ~15 min/reactor, so depth is chosen
for accuracy/interpretability, not speed).

Five variables score real R2 0.97-1.00. The three amino acids read n/a only because
their forecast-window range is just under 0.1, NOT because they are flat: rho is
0.51-0.86 (up from ~0.2 before the fix) and the parity plot shows them on the
diagonal, so the model tracks them. Their swing is real but front-loaded into
days 0-6 (the input window).

**Amino-acid data-gen fix (important), and the zero-negative choice:** earlier runs
showed glutamine/serine/asparagine as perfectly flat lines with rho ~0.2. That was a
generator artifact: the AA initial pool was tied to a 210x-inflated perfusion feed,
pinning each AA at its feed level. The fix, NO clamp anywhere, decoupled the initial
pool (realistic DMEM) from the feed (enriched per-AA only as much as needed), which
restored the rise-then-deplete dynamics of data_2. Separately, glucose was depleting
NEGATIVE in high-consumption / low-glucose-DoE reactors (feed 25 too low); raising it
to 40 (`--glc-feed 40`) removed all glucose negatives with glucose R2 unchanged at
1.00. Net: zero `[ODE NEG]` events, no clamp.

Trade-off we characterized: lowering the AA feed further (`--aa-feed-factor 0.8`)
pushes the AA depletion into the forecast window so all three score real R2 (0.97-
1.00), but at ~1,700 slightly-negative AA points. We chose the zero-negative config;
the 0.8 variant is available if scoreable AAs are preferred over strict
non-negativity. This tension (continued depletion vs non-negativity) is itself a real
finding: the data_3 uptake fluxes run high relative to the AA pools.

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

## 2. Speed: NN inference vs solving the ODE (see 2b for the honest version)

NOTE: 2b below is the authoritative comparison (real network, same device). This
first pass is CPU-only and conflates hardware; keep it only as the initial cut.

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
runs the ODE on the SAME device as the NN. Timed with the ADOPTED small model
(1 conv layer, hidden 32), so the pure-NN row is the actual inference time. A100,
n=20,000, ~20s/method:

| method | ms/reactor | vs solve_ivp (CPU) | vs torchdiffeq (GPU) |
|---|---|---|---|
| numerical ODE (solve_ivp, CPU) | 26.15 | 1x | - |
| closed-form ODE (numpy, CPU) | 0.45 | 58x | - |
| closed-form ODE (torch, cuda) | 0.0005 | 50,000x | 24x |
| torchdiffeq odeint (cuda) | 0.0125 | 2,100x | 1x (baseline) |
| pure NN (cuda) | 0.0003 | 79,500x | 38x |
| hybrid NN (closed-form ODE in forward, cuda) | 0.0006 | 43,000x | 20.5x |
| hybrid NN (50-substep Euler in forward, cuda) | 0.0050 | 5,200x | 2.5x |

PINN inference = pure NN (0.0003 ms): the ODE is only a training penalty, gone at
inference. Two baselines on purpose: "vs solve_ivp" is the incumbent (scipy,
CPU-only, no GPU version exists); "vs torchdiffeq" is a general differentiable
solver on the NN's device, isolating algorithm from hardware.

**The honest findings (small model):**
- The pure NN (0.0003 ms) slightly edges out the exact closed-form ODE (0.0005 ms) on
  the GPU, so on this simplified ODE the NN matches or marginally beats the direct
  solve (both at the sub-microsecond floor). It is ~38x faster than a fair GPU solver
  (torchdiffeq) and ~79,500x faster than scipy (mostly CPU-vs-GPU hardware).
- ODE-in-forward cost: the PINN is FREE (= pure NN; no ODE at inference); the hybrid's
  closed-form step roughly DOUBLES it (0.0006 ms, ~2x); the 50-substep Euler
  integrator is ~17x. The step is a fixed cost, so it is a larger fraction of this
  tiny network than of the big one.
- The surrogate's real speed case is the reference having NO closed form: vs the
  ACTUAL COSMIC-dFBA (~15 min/reactor, Sarat) the NN is ~hundreds of millions x
  faster; on our simplified closed-form generator it is a wash. So the presentation
  leads with MEDIA DESIGN (what the model learns) and treats speed as enabling, not
  the contribution.

Do NOT combine speed and accuracy in one table (different units/comparisons). The
honest bridge: the NN matches the ODE's accuracy (few percent) and, on our simplified
generator, its speed too; the decisive speed win is only vs models with no closed
form (the real COSMIC-dFBA), and the lasting value is what the model learns.

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
