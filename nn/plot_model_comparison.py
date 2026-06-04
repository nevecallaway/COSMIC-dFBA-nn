#!/usr/bin/env python3
"""
Standalone comparison figure: shuffled vs FC (DoE-only) vs LSTM (DoE-only).
Run from any directory -- outputs to nn/figures/model_comparison.png.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent / 'figures' / 'model_comparison.png'

REACTORS = ['R0001', 'R0002', 'R0003', 'R0004', 'R0005',
            'R0006', 'R0008', 'R0010', 'R0011', 'R0012']

# Per-reactor absolute transition errors (days)
# FC model, DoE only, LOO MAE = 1.37d
fc_errors   = [0.9, 1.1, 0.1, 1.7, 0.4, 0.1, 1.4, 0.5, 0.2, 0.0]

# LSTM model, DoE only, LOO MAE = 1.46d
lstm_errors = [0.7, 1.3, 0.1, 1.6, 0.2, 0.0, 2.0, 0.5, 0.1, 0.2]

# Shuffled baseline -- LOO MAE = 2.20d, shown as a horizontal line only

# Summary metrics
models     = ['Shuffled\n(chance)', 'FC\n(DoE only)', 'LSTM\n(DoE only)']
loo_mae    = [2.20,  1.37,  1.46]
loo_mcc    = [0.845, 0.878, 0.896]
ft_acc     = [60.0,  85.4,  80.0]
auc_mae    = [1.71,  0.35,  0.44]

COLORS = {
    'shuffled': '#bdbdbd',
    'simple':   '#4393c3',
    'complex':  '#2166ac',
    'paper':    '#d6604d',
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Model Comparison: Shuffled vs FC vs LSTM (DoE-only inputs)', fontsize=13, y=1.01)

# ── Panel 1: Summary bar chart ────────────────────────────────────────────────
ax = axes[0]
x   = np.arange(3)
w   = 0.22
cols = [COLORS['shuffled'], COLORS['simple'], COLORS['complex']]

# Normalise each metric to [0, 1] so all four fit on one axis
# LOO MAE: lower is better -- invert (best=0 → 1, worst=2.5 → 0)
mae_norm  = 1 - np.array(loo_mae)  / 2.5
mcc_norm  = np.array(loo_mcc)
ft_norm   = np.array(ft_acc) / 100
auc_norm  = 1 - np.array(auc_mae) / 2.0

metrics      = [mae_norm, mcc_norm, ft_norm, auc_norm]
metric_labels = ['LOO Trans MAE\n(inverted, higher=better)',
                 'LOO MCC',
                 'f(t) ±0.1 accuracy',
                 'Phase AUC MAE\n(inverted, higher=better)']
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

for i, (met, label) in enumerate(zip(metrics, metric_labels)):
    ax.bar(x + offsets[i], met, w, label=label, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Normalised score (higher = better)')
ax.set_ylim(0, 1.12)
ax.set_title('LOO Summary Metrics')
ax.legend(fontsize=7, loc='upper left')
ax.axhline(0, color='black', linewidth=0.5)

# annotate actual values
for mi, met in enumerate(metrics):
    for xi, (val, color) in enumerate(zip(met, cols)):
        ax.text(xi + offsets[mi], val + 0.01,
                [f'{loo_mae[xi]:.2f}d', f'{loo_mcc[xi]:.3f}',
                 f'{ft_acc[xi]:.1f}%', f'{auc_mae[xi]:.2f}d'][mi],
                ha='center', va='bottom', fontsize=6, color='#333333')

# ── Panel 2: Per-reactor transition error ─────────────────────────────────────
ax = axes[1]
x  = np.arange(len(REACTORS))
w  = 0.3

ax.bar(x - w/2, fc_errors,   w, label='FC (DoE only, 1.37d)',
       color=COLORS['complex'], alpha=0.85)
ax.bar(x + w/2, lstm_errors, w, label='LSTM (DoE only, 1.46d)',
       color=COLORS['simple'],  alpha=0.85)
ax.axhline(2.20, color=COLORS['shuffled'], linewidth=1.5,
           linestyle='--', label='Shuffled LOO MAE (2.20d)')
ax.axhline(np.mean(fc_errors),   color=COLORS['complex'],
           linewidth=1, linestyle=':', alpha=0.7)
ax.axhline(np.mean(lstm_errors), color=COLORS['simple'],
           linewidth=1, linestyle=':', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(REACTORS, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Absolute transition error (days)')
ax.set_title('Per-Reactor Transition Error')
ax.legend(fontsize=8)
ax.set_ylim(0, 3.0)

# ── Panel 3: LOO MAE + f(t) accuracy scatter ──────────────────────────────────
ax = axes[2]

scatter_data = [
    ('Shuffled',         2.20, 60.0,  COLORS['shuffled'], 's', 80),
    ('FC (DoE only)',    1.37, 85.4,  COLORS['complex'],  'o', 100),
    ('LSTM (DoE only)',  1.46, 80.0,  COLORS['simple'],   'o', 100),
    ('Paper benchmark',  None, 72.3,  COLORS['paper'],    '^', 80),
]

for label, mae, ft, color, marker, size in scatter_data:
    if mae is not None:
        ax.scatter(mae, ft, c=color, marker=marker, s=size, zorder=3,
                   label=label, edgecolors='white', linewidths=0.5)
        ax.annotate(label, (mae, ft), textcoords='offset points',
                    xytext=(6, 4), fontsize=8, color=color)
    else:
        ax.axhline(ft, color=color, linestyle='--', linewidth=1.2,
                   label=f'{label} ({ft}%)', alpha=0.8)

ax.set_xlabel('LOO Transition MAE (days, lower is better)')
ax.set_ylabel('f(t) ±0.1 accuracy % (higher is better)')
ax.set_title('Accuracy vs Generalization')
ax.legend(fontsize=8)
ax.set_xlim(1.2, 2.5)
ax.set_ylim(50, 95)
ax.invert_xaxis()  # lower MAE = better = right side

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT}')
