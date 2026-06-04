# Figures Guide

## model_comparison.png

Three-panel comparison of model variants on the phase transition prediction task.

---

### Left panel: LOO Summary Metrics

Four metrics shown side by side for each model. Because they have different units and scales,
each is normalized to [0, 1] so they fit on one axis -- higher bar always means better.

**Why does a bar show e.g. "2.20d" but only reach 0.12 on the y-axis?**
The annotated label is the actual value. The bar height is the normalized score.
For LOO Trans MAE, lower days is better, so it is inverted: `score = 1 - MAE / 2.5`.
For the shuffled model: `1 - 2.20 / 2.5 = 0.12`. The annotation tells you the real number;
the bar height lets you compare visually.

**The four metrics:**

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| LOO Trans MAE | Average error in predicting the transition day, evaluated by holding each reactor out of training in turn (leave-one-out). Primary metric. Units: days. | Lower is better (bar inverted) |
| MCC | Matthews Correlation Coefficient. Classifies each time point as growth phase or production phase and scores correctness. Range -1 to +1; 0 = random chance, 1 = perfect. More reliable than accuracy or F1 when the two classes are unequal in size. | Higher is better |
| f(t) +/-0.1 accuracy | At each of the 13 daily time points, does the predicted phase fraction f(t) land within 0.1 of ground truth? Reported as a percentage across all time points and reactors. | Higher is better |
| Phase AUC MAE | The integral of f(t) over the full run equals the number of days the reactor spent in production phase. Error between predicted and actual integral, in days. Captures both timing and sharpness of the transition in one number. | Lower is better (bar inverted) |

---

### Middle panel: Per-Reactor Transition Error

Each pair of bars is one reactor. Bar height = absolute error in predicted transition day vs actual.
The dashed grey line is the shuffled model's LOO MAE -- a chance-level reference.

Hard reactors (both models consistently above ~1d): R0002, R0004, R0008. These have transition
dynamics that the 3 DoE inputs cannot fully explain.

Easy reactors (both models under 0.5d): R0003, R0005, R0006, R0011, R0012.

---

### Right panel: Accuracy vs Generalization

Each model is a point. The x-axis is LOO Trans MAE (inverted, so right = lower error = better).
The y-axis is f(t) accuracy (higher = better). Upper-right is best in both dimensions.

The paper benchmark has no LOO MAE (different evaluation method), so it appears as a horizontal
dashed line at its f(t) accuracy (72.3%) -- read off where our model sits relative to it on
the y-axis.

---

## Terminology

| Term | Meaning |
|------|---------|
| DoE | Design of Experiment. The 3 coded input variables set before the run: O2, amino acids (AAs), glucose (Glc), each at level -1, 0, or +1. |
| f(t) | Phase transition function: a sigmoid from 0 (pure growth phase) to 1 (pure production phase). Parameterised by mu (transition midpoint in days) and sigma (transition width in days). |
| LOO | Leave-One-Out cross-validation. Train on 9 reactors, test on the 10th, repeat for each reactor. The honest generalization estimate when only 10 samples are available. |
| Shuffled | Permutation baseline: inputs and outputs are randomly mismatched before training. Any model learning genuine signal should beat this. LOO MAE ~2.20d. |
| FC (DoE only) | Final model. Uses only DoE coded levels and initial conditions as inputs. Rate heads are fully connected layers predicting one constant rate vector per phase per reactor. |
| LSTM (DoE only) | Comparison variant. Same inputs as FC but rate heads use an LSTM to produce time-varying rates. Worse on all metrics -- constant rates are the right inductive bias for phase-constant dFBA dynamics. |
| Complex (all inputs) | Earlier model variant that also used FBA-derived specific rates and efficiencies as inputs. Dropped because they require running dFBA first, defeating the surrogate purpose. |
| Phase AUC | Integral of f(t) dt over the full run. Equals the number of days spent in production phase. Summarises the entire transition curve as a single interpretable number. |
| MCC | Matthews Correlation Coefficient. Phase classification score ranging from -1 (perfectly wrong) to +1 (perfect). 0 = random chance. Preferred over accuracy and F1 for unbalanced classes. |
