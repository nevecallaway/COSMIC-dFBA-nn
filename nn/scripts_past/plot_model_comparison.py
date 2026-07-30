#!/usr/bin/env python3
"""
Standalone comparison figure: FC model vs shuffled baseline vs paper benchmarks.
Run from any directory -- outputs to nn/figures/model_comparison.png.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'figures' / 'model_comparison.png'

REACTORS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
            'R0006', 'R0008', 'R0010', 'R0011', 'R0012']

# Per-reactor absolute transition errors -- FC model only (paper has no per-reactor data)
fc_errors = [0.8, 1.1, 0.1, 1.4, 0.3, 0.4, 1.7, 0.5, 0.1, 0.2]

# Summary metrics for paper-comparable metrics
# Paper only reports MCC, F1, Spec, Sens, f(t) -- no LOO Trans MAE or Phase AUC MAE
#                    Shuffled  FC      Paper
mcc_vals  = [0.816,  0.906,  0.454]
f1_vals   = [0.867,  0.954,  0.731]
spec_vals = [0.907,  0.975,  0.780]
sens_vals = [0.931,  0.952,  0.681]
ft01_vals = [57.7,   83.8,   72.3]   # f(t) ±0.1 accuracy (%)
ft02_vals = [78.5,   94.6,   90.8]   # f(t) ±0.2 accuracy (%)

SHUFFLED_LOO_MAE = 2.32   # shown as reference line in per-reactor panel
FC_LOO_MAE       = 1.40

COLORS = {
    'shuffled': '#bdbdbd',
    'fc':       '#4393c3',
    'paper':    '#d6604d',
}

models = ['Shuffled\n(chance)', 'FC\n(DoE only)', 'Paper\nbenchmark']
x3     = np.arange(3)
w      = 0.22

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Phase Transition Prediction: FC Model vs Shuffled Baseline vs Paper',
             fontsize=13, y=1.01)

# ── Panel 1: Grouped bar chart of paper-comparable metrics ───────────────────
ax = axes[0]
cols = [COLORS['shuffled'], COLORS['fc'], COLORS['paper']]

metrics       = [mcc_vals, f1_vals, spec_vals, sens_vals]
metric_labels = ['MCC', 'F1', 'Specificity', 'Sensitivity']
offsets       = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

for i, (vals, label) in enumerate(zip(metrics, metric_labels)):
    bars = ax.bar(x3 + offsets[i], vals, w, label=label, alpha=0.85)

ax.set_xticks(x3)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Score (higher = better)')
ax.set_ylim(0, 1.15)
ax.set_title('Classification Metrics (LOO)')
ax.legend(fontsize=7, loc='lower right')
ax.axhline(0, color='black', linewidth=0.5)

for i, vals in enumerate(metrics):
    for xi, val in enumerate(vals):
        ax.text(xi + offsets[i], val + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6, color='#333333')

# ── Panel 2: f(t) accuracy grouped bars ──────────────────────────────────────
ax = axes[1]
w2 = 0.3
ft_metrics = [ft01_vals, ft02_vals]
ft_labels  = ['f(t) ±0.1', 'f(t) ±0.2']
offsets2   = [-0.5*w2, 0.5*w2]

for i, (vals, label) in enumerate(zip(ft_metrics, ft_labels)):
    ax.bar(x3 + offsets2[i], vals, w2, label=label, alpha=0.85)

ax.set_xticks(x3)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Accuracy % (higher = better)')
ax.set_ylim(0, 110)
ax.set_title('Phase Fraction Accuracy')
ax.legend(fontsize=8)
ax.axhline(0, color='black', linewidth=0.5)

for i, vals in enumerate(ft_metrics):
    for xi, val in enumerate(vals):
        ax.text(xi + offsets2[i], val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=7, color='#333333')

# ── Panel 3: Per-reactor transition error (FC only; shuffled as reference) ───
ax = axes[2]
xr = np.arange(len(REACTORS))

ax.bar(xr, fc_errors, color=COLORS['fc'], alpha=0.85,
       label=f'FC (DoE only, LOO MAE={FC_LOO_MAE}d)')
ax.axhline(SHUFFLED_LOO_MAE, color=COLORS['shuffled'], linewidth=1.5,
           linestyle='--', label=f'Shuffled LOO MAE ({SHUFFLED_LOO_MAE}d)')
ax.axhline(np.mean(fc_errors), color=COLORS['fc'],
           linewidth=1, linestyle=':', alpha=0.8,
           label=f'FC mean ({np.mean(fc_errors):.2f}d)')

ax.set_xticks(xr)
ax.set_xticklabels(REACTORS, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Absolute transition error (days)')
ax.set_title('Per-Reactor Transition Error (FC)')
ax.legend(fontsize=8)
ax.set_ylim(0, 3.0)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT}')
