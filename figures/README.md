# Figures Guide

## model_comparison.png

Three-panel comparison of the FC model against the shuffled baseline and the paper benchmark.

---

### Left panel: Classification Metrics (LOO)

Four classification metrics shown as grouped bars for three groups: Shuffled (chance baseline),
FC (our model, DoE-only inputs), and the Paper benchmark. All metrics are on a 0-1 scale;
higher is always better. Our FC model (LOO) is compared directly to the paper's in-sample
metrics -- a fair comparison is noted in the caption below.

| Metric | What it measures |
|--------|-----------------|
| MCC | Matthews Correlation Coefficient. Classifies each time point as growth or production phase. Range -1 to +1; 0 = random chance, 1 = perfect. More reliable than accuracy or F1 for unbalanced classes. |
| F1 | Harmonic mean of precision and recall for phase classification. |
| Specificity | True negative rate: fraction of growth-phase time points correctly identified as growth. |
| Sensitivity | True positive rate: fraction of production-phase time points correctly identified as production. |

---

### Middle panel: Phase Fraction Accuracy

f(t) accuracy at two tolerance levels, same three-group layout.

| Metric | What it measures |
|--------|-----------------|
| f(t) ±0.1 | At each of the 13 daily time points, does predicted f(t) land within 0.1 of ground truth? Percentage across all time points and reactors. |
| f(t) ±0.2 | Same at a looser tolerance. |

f(t) is the phase transition function: a sigmoid from 0 (pure growth phase) to 1 (pure
production phase). The paper benchmark for f(t) ±0.1 is 72.3%; our FC model reaches 83.8%.

---

### Right panel: Per-Reactor Transition Error (FC)

Each bar is one reactor. Height = absolute error in predicted transition day vs actual (days).

- Dashed grey line: shuffled model LOO MAE (2.32d) -- chance-level reference
- Dotted blue line: FC model mean error (0.66d)

Hard reactors (consistently above 1d): R0002, R0004, R0008. These have transition dynamics
the 3 DoE inputs cannot fully explain.

Easy reactors (under 0.5d): R0003, R0005, R0006, R0011, R0012.

The paper has no per-reactor breakdown, so it does not appear in this panel.

---

## Terminology

| Term | Meaning |
|------|---------|
| DoE | Design of Experiment. The 3 coded input variables set before the run: O2, amino acids (AAs), glucose (Glc), each at level -1, 0, or +1. |
| f(t) | Phase transition function: a sigmoid from 0 (pure growth phase) to 1 (pure production phase). Parameterised by mu (transition midpoint in days) and sigma (transition width in days). |
| LOO | Leave-One-Out cross-validation. Train on 9 reactors, test on the 10th, repeat for each reactor. The honest generalization estimate when only 10 samples are available. |
| Shuffled | Permutation baseline: inputs and outputs are randomly mismatched before training (seed=0). Any model learning genuine signal should beat this. LOO MAE = 2.32d. |
| FC (DoE only) | Final model. Uses only DoE coded levels and initial conditions as inputs. Rate heads are fully connected layers predicting one constant rate vector per phase per reactor. Seed=42. LOO MAE = 1.40d. |
| Paper benchmark | COSMIC dFBA mechanistic model (Gopalakrishnan et al.). Evaluated in-sample on all 10 reactors -- not LOO. Our LOO metrics are a stricter test. |
| Phase AUC | Integral of f(t) dt over the full run. Equals the number of days spent in production phase. Summarises the entire transition curve as a single interpretable number. |
| MCC | Matthews Correlation Coefficient. Phase classification score ranging from -1 (perfectly wrong) to +1 (perfect). 0 = random chance. |
