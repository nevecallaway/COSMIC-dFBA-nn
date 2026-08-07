# Project History

A consolidated development journal for the COSMIC-dFBA neural surrogate (BYU /
Lewis Lab internship, May 2026). It records how the project moved from an early
real-data model that failed, to the current synthetic-only forecasting surrogate
in this repo.

> **Note on scope.** The current repo is the condensed synthetic-only, pure-NN
> pipeline (see [`README.md`](README.md)). Much of the work below is the earlier
> real-data / leave-one-reactor-out (LORO) phase, now archived under
> `nn/scripts_past/` and `archive/`. This file keeps that history so the
> reasoning behind the current design is not lost.

## Timeline

**May 4-8, 2026: Setup.** Read the COSMIC-dFBA paper, finalized BYU documents,
got Dr. Hill's approval to use BYU research computing.

**May 11: Lit review** finished.

**May 12: First model, first failure.** Reached synthetic-data training and
pulled real data from the paper's supplemental tables (10 reactors). A sigmoid
phase-regression model predicted flat lines (~0.5) for everything and ignored the
metabolite dynamics. Diagnosed causes: synthetic dynamics were smooth while real
switches were hard; too little data (only 7-10 reactors); initial conditions were
ignored; the loss did not penalize flatness. Tried IC / non-flatness /
bistability / phase-smoothness penalties (did not work), then binary phase
classification with cross-entropy, then a **phase-aware architecture** that blends
separate growth and production decoders. Key insight: phase must directly modulate
the output, or the model ignores it.

**May 13-14: Comparisons and the first breakthrough.**
- Head-to-head of three architectures: **the simple attention baseline won** (MSE
  0.0958); explicit phase handling actually hurt.
- **Synthetic augmentation made it worse** (MSE 0.210 vs 0.0958): with only ~10
  reactors, real data beat synthetic.
- Diagnosis: Glucose was the easiest component (R2 0.446), Titer the hardest (R2
  0.016); the early phase was the worst-predicted window.
- **Uncertainty (heteroscedastic) head:** predict a value plus a confidence, so the
  model reports high uncertainty where data are messy (early phase). Result: about
  +75% overall, Titer about +67%.

**May 15: Kimberly meeting (direction set).** Make dFBA differentiable as the
ultimate goal (manifold intuition: backprop nudges weights downhill toward the
solution). Predict the *velocity* of consumption, not just concentrations. Use
better metrics (F1, MCC, R2). Motivating use case: root-cause analysis, catch the
titer drop-off before the cells die. Pointed toward a Neural-ODE / PINN design:
predict rates, integrate them to concentrations, add a mass-balance penalty. Early
claim: about 1000x faster than the MATLAB reference.

**May 15-18: Rebuild on 25 metabolites + bug fixes.** Replaced the 4-component
generator with a 25-metabolite one; sampled ICs by perturbing real day-0 values;
built a pre-train-on-synthetic then fine-tune-on-real pipeline. Fixed several
serious bugs:
- **Normalization mismatch:** synthetic and real were normalized with different
  stats (one component hit R2 = -2.5 million). Fixed by applying the real data's
  normalization to synthetic before training.
- **Pre-training stopped at epoch 1** (early stopping fired immediately); disabled
  early stopping during pre-training.
- **Synthetic Titer shape wrong:** real Titer peaks then declines (fed-batch
  dilution) but the generator was monotonic and too low; added peak-then-decline
  and raised production/growth rates.
- **Scale-up:** synthetic 1k -> 20k, and replaced the random 70/30 split with
  leave-one-out cross-validation for stable metrics on 10 reactors.

**May 19: The turnaround.** Two changes carried it:
- **Gaussian augmentation of real trajectories** (each synthetic sample is a noisy
  copy of a real reactor) instead of hand-coded biology, so synthetic stays in the
  real domain automatically.
- **DoE process parameters (O2, AAs, Glc) as explicit inputs** (the single biggest
  improvement): previously the model had to infer why reactors differed from day-0
  alone. Also fixed a scheduler that had decayed the learning rate to ~1e-6.

**May 20: Physics-informed encoder-decoder.** Attention-based encoder-decoder,
continuous phase regression, DoE + 50 phase-specific rates as inputs, transfer
learning on 50k augmented trajectories, LOO metrics, and EDA + evaluation tooling
(+ SLURM / Apptainer for the cluster).

**May 21: Titer shape, not monotonicity.** SLURM runs looked strong but were
overfitting (F1 = 1.0). Removed the monotonicity penalty (it fought the real
peak-then-decline) and added a **peak_time_loss** (a differentiable soft-argmax on
the titer series that matches when the peak occurs). An external candidate dataset
did not align well enough to integrate.

**May 22: Model spec + planning.** Finalized inputs (78 features: 25 ICs + 3 DoE +
50 rates) predicting a 25 x 13 trajectory, with an auxiliary phase output. Sarat /
Kimberly guidance: establish a scrambled-input random baseline as the number to
beat; consider LSTM vs transformer; the real prediction target is when the cell
line dies (root-cause analysis).

**Later (current phase): en Primeur.** The project narrowed to the synthetic-only,
pure-NN forecasting surrogate documented in `README.md`: forecast the rest of a run
from its first ~6 days using a small 1D-CNN + attention model, autoregressive
rollout, and global (invertible) normalization. This is the version presented and
kept in the main pipeline.

## Key turnaround (real-data LORO metrics)

| Metric                   | Start (05/19) | After fixes |
| ------------------------ | ------------- | ----------- |
| Mean LOO R2              | -1.68         | +0.09       |
| Folds with positive R2   | 0 / 10        | 7 / 10      |
| Best single-fold R2      | (none)        | +0.46       |
| Best Titer R2            | -0.86         | +0.90       |
| Mean F1 (phase)          | 0.88          | 0.95        |

The model went from worse-than-the-mean to positive on most folds, and Titer (the
hardest target) from strongly negative to +0.90 on its best fold.

## Findings from the figures

- **Phase is a strong, learnable signal (`figures/5_pca_phase.png`,
  `6_umap_phase.png`).** Real timepoints separate cleanly into growth vs production
  along PC1, with transition points in between. Phase state is real structure, not
  noise, which is why coupling phase to the output helped.
- **The metabolite space is effectively low-dimensional
  (`figures/2_correlation_heatmap.png`).** The essential amino acids (Proline,
  Threonine, Histidine, Lysine, Valine, Methionine, Arginine, Tyrosine,
  Isoleucine, Leucine, Phenylalanine) form a tightly co-varying block (~0.9),
  i.e. they are largely redundant. Titer tracks NH4 and Cell Density. Many of the
  25 metabolites carry overlapping information.
- **The early model collapsed to the mean
  (`figures/past/comprehensive_analysis.png`).** Predictions piled around ~0.45
  regardless of the true value (mean absolute error ~0.25), the flat-line
  pathology. Glucose was easiest, Titer hardest; some reactors (R0003, R0011) were
  easy and others (R0004, R0008) hard. This motivated the uncertainty head and the
  DoE inputs.
- **The current surrogate is accurate across the board
  (`figures/nn_baseline_parity_final.png`).** Every variable hugs the diagonal:
  Glucose R2 = 1.00, Cell Density 0.99, Cell Size 0.98, Titer 0.97, Glycine 0.95.
  The three near-flat amino acids (Glutamine, Asparagine, Serine) score n/a for R2
  but still track the small movement.

## Lessons learned

- **Phase must modulate the output**, otherwise the model treats it as optional and
  averages everything to the middle.
- **Real beats hand-coded synthetic**; Gaussian augmentation of real trajectories
  captured the true dynamics without brittle ODE tuning.
- **DoE process parameters as explicit inputs were the single biggest accuracy
  gain**; day-0 measurements alone cannot explain why reactors differ.
- **Normalize synthetic and real with the same statistics**; a mismatch is
  catastrophic.
- **Watch the optimizer plumbing**: an over-eager LR scheduler and premature early
  stopping each silently crippled training.
- **Titer is the hard target** (peak-then-decline product formation); it needs
  shape-aware losses (peak_time_loss), not a monotonicity constraint.
- **Always establish a scrambled-input random baseline** as the number to beat.
