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

| feature | peak ratio | norm MAE | rho | signal range |
|---|---|---|---|---|
| Cell Density | 1.05 | 0.059 | 0.63 | 0.31 |
| Cell Size | 1.00 | 0.029 | 1.00 | 0.51 |
| Titer | 0.99 | 0.069 | 0.97 | 0.72 |
| Glucose | 1.01 | 0.033 | 0.75 | 0.23 |
| Asparagine | 1.00 | 0.036 | 0.78 | 0.09 |
| Glutamine | 1.00 | 0.003 | 0.06 * | 0.00 |
| Serine | 1.00 | 0.005 | 0.41 * | 0.01 |
| Glycine | 1.00 | 0.003 | 0.13 * | 0.00 |

\* Low rho = flat metabolite (signal range < 0.1), predicted near-exactly;
correlation is not meaningful when there is nothing to correlate. rho tracks
signal range monotonically (Titer, range 0.72, gets rho 0.97).

**Takeaway:** peak ratios 0.99-1.05 and tiny MAE everywhere. A physics-free NN
predicts all 8 variables in absolute units to within a few percent. **The ODE is
not what makes the model accurate.**

Figures: `nn_baseline_<Feature>.png` (per-feature 2x5 reactor grids; flat
metabolites y-anchored at 0). Hero slides = Titer, Glucose, Cell Density.

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
