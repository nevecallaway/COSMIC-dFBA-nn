#!/usr/bin/env python3
"""
Model diagnosis script for COSMIC-dFBA.

Answers:
  1. Does the model beat random / mean baselines?
  2. What's different about the reactors it gets wrong?
  3. Where in time / which components / which phase does it fail?

Usage:
    python nn/diagnose.py
    python nn/diagnose.py --model nn/improved_model.pt

Outputs ~12 figures to nn/figures/diag_*.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from model import CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
from utils import load_experimental_data
from torch.utils.data import DataLoader

COMPONENT_NAMES = [
    'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid', 'L-Serine',
    'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine', 'L-Histidine',
    'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine', 'L-Tyrosine',
    'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine', 'L-Tryptophan',
]
IDX_TITER = 5
DOE_NAMES = ['O₂', 'AAs', 'Glc']


# ── Data / model loading ──────────────────────────────────────────────────────

def load_everything(model_path, here):
    doe_file   = str(here / 'data' / 'data_1.csv')
    rates_file = str(here / 'data' / 'data_3.csv')
    trajs, times, ics, meta = load_experimental_data(
        str(here / 'data' / 'data_2.csv'),
        doe_file=doe_file, rates_file=rates_file)

    reactors = list(meta['reactors'])
    phases   = meta['phases']
    doe_arr  = meta.get('doe_params')
    rate_arr = meta.get('specific_rates')

    parameters = {}
    if doe_arr  is not None:
        parameters.update({'O2': doe_arr[:, 0], 'AAs': doe_arr[:, 1], 'Glc': doe_arr[:, 2]})
    if rate_arr is not None:
        for k in range(rate_arr.shape[1]):
            parameters[f'rate_{k}'] = rate_arr[:, k]

    dataset = dFBADataset(trajs, times, ics, parameters=parameters,
                          normalize=True, phases=phases)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(model_path, map_location=device)
    hp     = ckpt['hyperparams']
    model  = CosmicNNSurrogateEnhanced(
        n_components=hp['n_components'], n_params=hp['n_params'],
        latent_dim=hp['latent_dim'],    n_heads=hp['n_heads'])
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        collate_fn=dfba_collate_fn)
    batch  = next(iter(loader))
    with torch.no_grad():
        out = model(batch['initial_conditions'].to(device),
                    batch['time'].to(device),
                    batch['parameters'].to(device))

    y_true      = batch['trajectory'].numpy()          # (N, T, C)
    y_pred      = out['concentrations'].cpu().numpy()  # (N, T, C)
    phase_true  = batch['phases'].numpy()              # (N, T)
    phase_pred  = out['phase_weights'].cpu().numpy()   # (N, T, 1)

    return (y_true, y_pred, phase_true, phase_pred,
            reactors, doe_arr, dataset, ckpt)


# ── Baseline predictors ───────────────────────────────────────────────────────

def mean_predictor(y_true):
    """Predict the global mean trajectory for every reactor."""
    mean_traj = y_true.mean(axis=0, keepdims=True)          # (1, T, C)
    return np.repeat(mean_traj, y_true.shape[0], axis=0)   # (N, T, C)

def last_value_predictor(y_true):
    """Predict the last observed value held constant (naive persistence)."""
    last = y_true[:, -1:, :]                                # (N, 1, C)
    return np.repeat(last, y_true.shape[1], axis=1)

def per_reactor_mean_predictor(y_true):
    """Each reactor predicted by its own temporal mean (LOO-unfair but useful floor)."""
    reactor_mean = y_true.mean(axis=1, keepdims=True)
    return np.repeat(reactor_mean, y_true.shape[1], axis=1)

def spearman_per_reactor(y_true, y_pred):
    """(N, C) Spearman rho: within-reactor, across timepoints."""
    N, T, C = y_true.shape
    rho = np.full((N, C), np.nan)
    for r in range(N):
        for c in range(C):
            t, p = y_true[r, :, c], y_pred[r, :, c]
            if t.std() > 1e-8 and p.std() > 1e-8:
                rho[r, c], _ = spearmanr(t, p)
    return rho


# ── 1. Baseline comparison ────────────────────────────────────────────────────

def plot_baseline_comparison(y_true, y_pred, reactors, out_dir):
    baselines = {
        'Model':            y_pred,
        'Mean trajectory':  mean_predictor(y_true),
        'Last value':       last_value_predictor(y_true),
    }

    # Compute metrics for each baseline
    metrics = {}
    for name, pred in baselines.items():
        rho_mat = spearman_per_reactor(y_true, pred)
        flat_t  = y_true.flatten()
        flat_p  = pred.flatten()
        r2      = r2_score(flat_t, flat_p)
        metrics[name] = {
            'Global R²':       r2,
            'Titer R²':        r2_score(y_true[:, :, IDX_TITER].flatten(),
                                        pred[:, :, IDX_TITER].flatten()),
            'Mean Spearman':   float(np.nanmean(rho_mat)),
            'Titer Spearman':  float(np.nanmean(rho_mat[:, IDX_TITER])),
        }

    metric_names = list(list(metrics.values())[0].keys())
    baseline_names = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1565C0', '#E53935', '#43A047']
    for i, (bname, color) in enumerate(zip(baseline_names, colors)):
        vals = [metrics[bname][m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=bname, color=color, alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model vs Baselines', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_1_baseline_comparison.png', dpi=150)
    plt.close(fig)

    print('  saved diag_1_baseline_comparison.png')
    print('\n  Baseline comparison:')
    for bname, m in metrics.items():
        print(f'    {bname:20s}  R²={m["Global R²"]:+.3f}  '
              f'TiterR²={m["Titer R²"]:+.3f}  '
              f'Spearman={m["Mean Spearman"]:+.3f}  '
              f'TiterSpearman={m["Titer Spearman"]:+.3f}')


# ── 2. Per-reactor Spearman ranking ──────────────────────────────────────────

def plot_reactor_ranking(y_true, y_pred, reactors, out_dir):
    rho_mat = spearman_per_reactor(y_true, y_pred)   # (N, C)
    titer_rho = rho_mat[:, IDX_TITER]
    order = np.argsort(titer_rho)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: titer Spearman by reactor
    ax = axes[0]
    colors = ['#E53935' if r < 0 else '#1565C0' for r in titer_rho[order]]
    ax.barh([reactors[i] for i in order], titer_rho[order], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Titer Spearman ρ')
    ax.set_title('Titer Spearman by Reactor')

    # Right: mean Spearman per component
    ax = axes[1]
    comp_rho = np.nanmean(rho_mat, axis=0)
    comp_order = np.argsort(comp_rho)
    colors_c = ['#E53935' if r < 0 else '#1565C0' for r in comp_rho[comp_order]]
    names = [COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f'comp_{i}'
             for i in comp_order]
    ax.barh(names, comp_rho[comp_order], color=colors_c)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean Spearman ρ (across reactors)')
    ax.set_title('Mean Spearman by Component')
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / 'diag_2_spearman_rankings.png', dpi=150)
    plt.close(fig)
    print('  saved diag_2_spearman_rankings.png')


# ── 3. Residual heatmaps ──────────────────────────────────────────────────────

def plot_residual_heatmaps(y_true, y_pred, reactors, out_dir):
    residuals = np.abs(y_true - y_pred)   # (N, T, C)
    T = y_true.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap 1: titer residual (reactor × time)
    ax = axes[0]
    im = ax.imshow(residuals[:, :, IDX_TITER], aspect='auto', cmap='Reds')
    plt.colorbar(im, ax=ax, label='|residual|')
    ax.set_yticks(range(len(reactors)))
    ax.set_yticklabels(reactors, fontsize=8)
    ax.set_xlabel('Timepoint')
    ax.set_title('Titer Absolute Residual (reactor × time)')

    # Heatmap 2: mean residual (reactor × component)
    ax = axes[1]
    mean_res = residuals.mean(axis=1)   # (N, C)
    im = ax.imshow(mean_res, aspect='auto', cmap='Reds')
    plt.colorbar(im, ax=ax, label='mean |residual|')
    ax.set_yticks(range(len(reactors)))
    ax.set_yticklabels(reactors, fontsize=8)
    ax.set_xticks(range(len(COMPONENT_NAMES)))
    ax.set_xticklabels([n[:6] for n in COMPONENT_NAMES], rotation=90, fontsize=6)
    ax.set_title('Mean Absolute Residual (reactor × component)')

    fig.tight_layout()
    fig.savefig(out_dir / 'diag_3_residual_heatmaps.png', dpi=150)
    plt.close(fig)
    print('  saved diag_3_residual_heatmaps.png')


# ── 4. Error vs time ─────────────────────────────────────────────────────────

def plot_error_vs_time(y_true, y_pred, phase_true, out_dir):
    residuals = np.abs(y_true - y_pred)   # (N, T, C)
    T = y_true.shape[1]
    t = np.linspace(0, 1, T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean error over time for key component groups
    groups = {
        'Biomass (CD, CV)':    [0, 1],
        'Substrates (Glc, Gln, NH4)': [2, 6, 4],
        'Products (Titer, Lac)':      [IDX_TITER, 3],
        'Amino acids':         list(range(7, 25)),
    }
    ax = axes[0]
    for gname, idxs in groups.items():
        mean_err = residuals[:, :, idxs].mean(axis=(0, 2))
        ax.plot(t, mean_err, marker='o', markersize=3, label=gname)
    ax.set_xlabel('Normalised time')
    ax.set_ylabel('Mean |residual|')
    ax.set_title('Error over Time by Component Group')
    ax.legend(fontsize=7)

    # Titer error over time per reactor
    ax = axes[1]
    for r in range(y_true.shape[0]):
        ax.plot(t, residuals[r, :, IDX_TITER], alpha=0.6, marker='o', markersize=3)
    ax.set_xlabel('Normalised time')
    ax.set_ylabel('|residual|')
    ax.set_title('Titer Error over Time (each line = one reactor)')

    # Error by phase state
    ax = axes[2]
    flat_phase = phase_true.flatten()
    flat_titer_res = residuals[:, :, IDX_TITER].flatten()
    growth_err = flat_titer_res[flat_phase < 0.2]
    trans_err  = flat_titer_res[(flat_phase >= 0.2) & (flat_phase <= 0.8)]
    prod_err   = flat_titer_res[flat_phase > 0.8]
    data = [d for d in [growth_err, trans_err, prod_err] if len(d) > 0]
    labels = [l for l, d in zip(['Growth\n(f<0.2)', 'Transition\n(0.2-0.8)', 'Production\n(f>0.8)'],
                                  [growth_err, trans_err, prod_err]) if len(d) > 0]
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Titer |residual|')
    ax.set_title('Titer Error by Phase State')

    fig.tight_layout()
    fig.savefig(out_dir / 'diag_4_error_vs_time_phase.png', dpi=150)
    plt.close(fig)
    print('  saved diag_4_error_vs_time_phase.png')


# ── 5. DoE space colored by performance ──────────────────────────────────────

def plot_doe_performance(y_true, y_pred, doe_arr, reactors, out_dir):
    if doe_arr is None:
        print('  (skipping DoE performance plot — no DoE data)')
        return

    rho_mat   = spearman_per_reactor(y_true, y_pred)
    titer_rho = rho_mat[:, IDX_TITER]

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (i, j) in zip(axes, pairs):
        sc = ax.scatter(doe_arr[:, i], doe_arr[:, j],
                        c=titer_rho, cmap='RdYlGn', vmin=-1, vmax=1,
                        s=120, edgecolors='black', linewidths=0.5, zorder=3)
        for r_idx, name in enumerate(reactors):
            ax.annotate(name, (doe_arr[r_idx, i], doe_arr[r_idx, j]),
                        fontsize=6, textcoords='offset points', xytext=(4, 4))
        plt.colorbar(sc, ax=ax, label='Titer Spearman ρ')
        ax.set_xlabel(DOE_NAMES[i])
        ax.set_ylabel(DOE_NAMES[j])
        ax.set_title(f'{DOE_NAMES[i]} vs {DOE_NAMES[j]}')
        # Add jitter lines for discrete -1/0/+1 values
        for v in [-1, 0, 1]:
            ax.axvline(v, color='gray', linewidth=0.4, linestyle=':')
            ax.axhline(v, color='gray', linewidth=0.4, linestyle=':')

    fig.suptitle('DoE Space Colored by Titer Spearman\n'
                 '(green = model predicts well, red = model predicts poorly)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_5_doe_performance.png', dpi=150)
    plt.close(fig)
    print('  saved diag_5_doe_performance.png')


# ── 6. Wrong vs right reactors: IC comparison ────────────────────────────────

def plot_wrong_vs_right(y_true, y_pred, reactors, out_dir):
    rho_mat   = spearman_per_reactor(y_true, y_pred)
    titer_rho = rho_mat[:, IDX_TITER]

    good = np.where(titer_rho >= 0.7)[0]
    bad  = np.where(titer_rho <  0.0)[0]

    if len(good) == 0 or len(bad) == 0:
        print('  (skipping wrong vs right — insufficient contrast in performance)')
        return

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    T = y_true.shape[1]
    t = np.linspace(0, 1, T)

    key_comps = [0, 2, IDX_TITER, 6, 3]
    key_names = [COMPONENT_NAMES[c] for c in key_comps]

    for row, (group, label, color) in enumerate([
        (good, 'Well predicted (Titer ρ ≥ 0.7)', '#1565C0'),
        (bad,  'Poorly predicted (Titer ρ < 0)',  '#E53935'),
    ]):
        for col, (cidx, cname) in enumerate(zip(key_comps, key_names)):
            ax = axes[row, col]
            for r in group:
                ax.plot(t, y_true[r, :, cidx], '-', color=color, alpha=0.5, linewidth=1.5)
                ax.plot(t, y_pred[r, :, cidx], '--', color='gray', alpha=0.5, linewidth=1)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_title(cname, fontsize=8)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                from matplotlib.lines import Line2D
                ax.legend(handles=[
                    Line2D([0], [0], color=color, label='Actual'),
                    Line2D([0], [0], color='gray', linestyle='--', label='Predicted'),
                ], fontsize=6)

    fig.suptitle('Well-predicted vs Poorly-predicted Reactors\n'
                 '(solid = actual, dashed = predicted)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_6_wrong_vs_right.png', dpi=150)
    plt.close(fig)
    print('  saved diag_6_wrong_vs_right.png')


# ── 7. All reactors titer overlay ────────────────────────────────────────────

def plot_titer_overlay(y_true, y_pred, reactors, out_dir):
    T = y_true.shape[1]
    t = np.linspace(0, 1, T)
    rho_mat   = spearman_per_reactor(y_true, y_pred)
    titer_rho = rho_mat[:, IDX_TITER]

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, arr) in zip(axes, [('Actual', y_true), ('Predicted', y_pred)]):
        for r in range(y_true.shape[0]):
            color = cmap(norm(titer_rho[r]))
            ax.plot(t, arr[r, :, IDX_TITER], marker='o', markersize=3,
                    color=color, linewidth=1.5, label=reactors[r])
        ax.set_xlabel('Normalised time')
        ax.set_ylabel('Titer (normalised)')
        ax.set_title(f'Titer Trajectories — {label}')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Titer Spearman ρ')

    fig.tight_layout()
    fig.savefig(out_dir / 'diag_7_titer_overlay.png', dpi=150)
    plt.close(fig)
    print('  saved diag_7_titer_overlay.png')


# ── 8. Predicted vs actual scatter per key component ─────────────────────────

def plot_scatter_grid(y_true, y_pred, out_dir):
    key = [0, 2, IDX_TITER, 3, 6, 19]   # CD, Glc, Titer, Lac, Gln, L-Arg
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, cidx in zip(axes, key):
        name = COMPONENT_NAMES[cidx] if cidx < len(COMPONENT_NAMES) else f'comp_{cidx}'
        t = y_true[:, :, cidx].flatten()
        p = y_pred[:, :, cidx].flatten()
        ax.scatter(t, p, s=8, alpha=0.4, color='#1565C0')
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        rho, _ = spearmanr(t, p)
        r2 = r2_score(t, p)
        ax.set_title(f'{name}\nSpearman={rho:.3f}  R²={r2:.3f}', fontsize=8)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.tick_params(labelsize=7)

    fig.suptitle('Predicted vs Actual Scatter (all reactors × timepoints)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_8_scatter_grid.png', dpi=150)
    plt.close(fig)
    print('  saved diag_8_scatter_grid.png')


# ── 9. Phase prediction quality ───────────────────────────────────────────────

def plot_phase_quality(phase_true, phase_pred, reactors, out_dir):
    phase_pred_1d = phase_pred[:, :, 0]   # (N, T)
    T = phase_true.shape[1]
    t = np.linspace(0, 1, T)

    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey=True)
    axes = axes.flatten()

    for r, ax in enumerate(axes):
        ax.plot(t, phase_true[r], 'o-', color='#1565C0', markersize=4, label='Actual')
        ax.plot(t, phase_pred_1d[r], 's--', color='#E53935', markersize=3, label='Predicted')
        ax.axhline(0.2, color='gray', linewidth=0.6, linestyle=':')
        ax.axhline(0.8, color='gray', linewidth=0.6, linestyle=':')
        ax.set_ylim(-0.05, 1.05)
        name = reactors[r] if r < len(reactors) else f'R{r}'
        res = np.abs(phase_true[r] - phase_pred_1d[r]).mean()
        ax.set_title(f'{name}  MAE={res:.3f}', fontsize=8)
        ax.tick_params(labelsize=6)
        if r == 0:
            ax.legend(fontsize=6)

    fig.suptitle('Phase Prediction: Actual vs Predicted f(t)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_9_phase_quality.png', dpi=150)
    plt.close(fig)
    print('  saved diag_9_phase_quality.png')


# ── 10. IC values for good vs bad reactors ───────────────────────────────────

def plot_ic_comparison(y_true, y_pred, reactors, out_dir):
    rho_mat   = spearman_per_reactor(y_true, y_pred)
    titer_rho = rho_mat[:, IDX_TITER]
    ics = y_true[:, 0, :]   # (N, C) — initial conditions in normalised space

    fig, axes = plt.subplots(5, 5, figsize=(18, 14))
    axes = axes.flatten()
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    for c, ax in enumerate(axes):
        if c >= ics.shape[1]:
            ax.set_visible(False)
            continue
        name = COMPONENT_NAMES[c] if c < len(COMPONENT_NAMES) else f'comp_{c}'
        colors = [cmap(norm(rho)) for rho in titer_rho]
        ax.bar(range(len(reactors)), ics[:, c], color=colors)
        ax.set_title(name, fontsize=7)
        ax.set_xticks(range(len(reactors)))
        ax.set_xticklabels([r[-4:] for r in reactors], rotation=90, fontsize=5)
        ax.tick_params(labelsize=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[-1], label='Titer Spearman ρ')
    fig.suptitle('Initial Conditions per Reactor\n'
                 '(bar color = titer Spearman, green=good, red=bad)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_10_ic_comparison.png', dpi=150)
    plt.close(fig)
    print('  saved diag_10_ic_comparison.png')


# ── 11. Residual distribution by component ───────────────────────────────────

def plot_residual_distributions(y_true, y_pred, out_dir):
    residuals = (y_pred - y_true)   # signed, (N, T, C)
    fig, axes = plt.subplots(5, 5, figsize=(18, 14))
    axes = axes.flatten()

    for c, ax in enumerate(axes):
        if c >= residuals.shape[2]:
            ax.set_visible(False)
            continue
        name = COMPONENT_NAMES[c] if c < len(COMPONENT_NAMES) else f'comp_{c}'
        data = residuals[:, :, c].flatten()
        ax.hist(data, bins=20, color='#1565C0', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='red', linewidth=1)
        ax.axvline(data.mean(), color='orange', linewidth=1, linestyle='--')
        ax.set_title(f'{name}\nbias={data.mean():.3f}', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle('Signed Residual Distributions (predicted − actual)\n'
                 'red=zero, orange=mean bias', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'diag_11_residual_distributions.png', dpi=150)
    plt.close(fig)
    print('  saved diag_11_residual_distributions.png')


# ── 12. Summary table ────────────────────────────────────────────────────────

def print_summary_table(y_true, y_pred, reactors):
    rho_mat   = spearman_per_reactor(y_true, y_pred)
    titer_rho = rho_mat[:, IDX_TITER]
    mean_rho  = np.nanmean(rho_mat, axis=1)

    print(f"\n{'='*65}")
    print(f"{'Reactor':<10} {'TiterSpearman':>15} {'MeanSpearman':>14} {'Assessment':>12}")
    print(f"{'='*65}")
    for r, name in enumerate(reactors):
        t_rho = titer_rho[r]
        m_rho = mean_rho[r]
        assessment = ('GOOD' if t_rho >= 0.7 else
                      'OK'   if t_rho >= 0.3 else
                      'POOR' if t_rho >= 0.0 else 'FAIL')
        print(f"{name:<10} {t_rho:>15.4f} {m_rho:>14.4f} {assessment:>12}")
    print(f"{'='*65}")
    print(f"{'Mean':<10} {np.nanmean(titer_rho):>15.4f} {np.nanmean(mean_rho):>14.4f}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(here / 'improved_model.pt'))
    args = parser.parse_args()

    out_dir = here / 'figures'
    out_dir.mkdir(exist_ok=True)

    print('Loading model and data...')
    (y_true, y_pred, phase_true, phase_pred,
     reactors, doe_arr, dataset, ckpt) = load_everything(args.model, here)

    print(f'  {len(reactors)} reactors, {y_true.shape[1]} timepoints, '
          f'{y_true.shape[2]} components\n')

    print_summary_table(y_true, y_pred, reactors)

    print('\nGenerating diagnostic figures...')
    plot_baseline_comparison(y_true, y_pred, reactors, out_dir)
    plot_reactor_ranking(y_true, y_pred, reactors, out_dir)
    plot_residual_heatmaps(y_true, y_pred, reactors, out_dir)
    plot_error_vs_time(y_true, y_pred, phase_true, out_dir)
    plot_doe_performance(y_true, y_pred, doe_arr, reactors, out_dir)
    plot_wrong_vs_right(y_true, y_pred, reactors, out_dir)
    plot_titer_overlay(y_true, y_pred, reactors, out_dir)
    plot_scatter_grid(y_true, y_pred, out_dir)
    plot_phase_quality(phase_true, phase_pred, reactors, out_dir)
    plot_ic_comparison(y_true, y_pred, reactors, out_dir)
    plot_residual_distributions(y_true, y_pred, out_dir)

    print(f'\nDone. All figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
