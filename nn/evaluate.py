#!/usr/bin/env python3
"""
Local evaluation script for a trained COSMIC-dFBA surrogate model.

Usage:
    # 1. Copy the model from the cluster:
    #    scp user@cluster:~/cosmic-dfba/improved_model.pt nn/
    #
    # 2. Run evaluation against local real data:
    #    python nn/evaluate.py
    #    python nn/evaluate.py --model nn/improved_model.pt   # explicit path

Outputs:
  - Metrics printed to stdout (R², Spearman, phase F1 per reactor)
  - Trajectory plots saved to nn/figures/eval_*.png
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import CosmicNNSurrogateEnhanced, CosmicNNSurrogateLSTM, dFBADataset, dfba_collate_fn
from utils import load_experimental_data, ModelDiagnostics
from torch.utils.data import DataLoader

COMPONENT_NAMES = [
    'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid', 'L-Serine',
    'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine', 'L-Histidine',
    'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine', 'L-Tyrosine',
    'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine', 'L-Tryptophan',
]

IDX_TITER = 5


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hyperparams']
    if hp.get('arch', 'transformer') == 'lstm':
        model = CosmicNNSurrogateLSTM(
            n_components=hp['n_components'],
            n_params=hp['n_params'],
            latent_dim=hp['latent_dim'],
            n_layers=hp.get('n_layers', 2),
        )
    else:
        model = CosmicNNSurrogateEnhanced(
            n_components=hp['n_components'],
            n_params=hp['n_params'],
            latent_dim=hp['latent_dim'],
            n_heads=hp['n_heads'],
        )
    result = model.load_state_dict(ckpt['model_state'], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  Warning: architecture mismatch (checkpoint may be from an older version)")
        print(f"    Missing : {result.missing_keys}")
        print(f"    Unexpected: {result.unexpected_keys}")
    model.to(device).eval()
    def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else "n/a"
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Saved LOO Trans MAE : {_fmt(ckpt.get('loo_mean_trans_mae'))}")
    print(f"  Saved LOO MCC       : {_fmt(ckpt.get('loo_mean_mcc'))}")
    print(f"  Saved LOO F1        : {_fmt(ckpt.get('loo_mean_f1'))}")
    return model, ckpt


def run_inference(model, dataset, device):
    """Run model on all reactors, return (n_reactors, T, C) arrays."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        collate_fn=dfba_collate_fn)
    batch = next(iter(loader))
    ic     = batch['initial_conditions'].to(device)
    time   = batch['time'].to(device)
    params = batch['parameters'].to(device)
    target = batch['trajectory'].to(device)

    # Truncate params if the model was trained with fewer features than the
    # current dataset provides (e.g. shuffled model trained before FBA features).
    if params.shape[1] != model.n_params:
        params = params[:, :model.n_params]

    with torch.no_grad():
        out = model(ic, time, params)

    y_pred  = out['concentrations'].cpu().numpy()   # (N, T, C)
    y_true  = target.cpu().numpy()
    phases_pred = out['phase_weights'].cpu().numpy()  # (N, T, 1)
    phases_true = batch['phases'].cpu().numpy() if 'phases' in batch else None

    return y_true, y_pred, phases_true, phases_pred


def print_metrics(y_true, y_pred, phases_true, phases_pred, reactors):
    print(f"\n{'='*65}")
    print("Regression Metrics (normalised space)")
    print(f"{'='*65}")
    reg = ModelDiagnostics.calculate_regression_metrics(y_true, y_pred)
    print(f"  Global R²   : {reg['global_r2']:.4f}")
    print(f"  Titer R²    : {reg['component_r2'].get('comp_5', float('nan')):.4f}")

    print(f"\n{'='*65}")
    print("Within-Reactor Spearman Correlation")
    print(f"{'='*65}")
    spear = ModelDiagnostics.calculate_spearman_metrics(y_true, y_pred)
    print(f"  Mean Spearman       : {spear['mean_spearman']:.4f}")
    print(f"  Titer Spearman      : {spear['titer_spearman']:.4f}")
    print(f"\n  Per-reactor titer Spearman:")
    for r_key, rho in spear['per_reactor_titer_spearman'].items():
        idx = int(r_key.split('_')[1])
        name = reactors[idx] if idx < len(reactors) else r_key
        print(f"    {name}: {rho:.4f}")

    print(f"\n  Bottom-5 components by Spearman:")
    comp_spear = spear['component_spearman']
    sorted_comps = sorted(comp_spear.items(), key=lambda x: x[1])
    for k, v in sorted_comps[:5]:
        idx = int(k.split('_')[1])
        name = COMPONENT_NAMES[idx] if idx < len(COMPONENT_NAMES) else k
        print(f"    {name}: {v:.4f}")

    if phases_true is not None:
        print(f"\n{'='*65}")
        print("Phase Metrics  (paper benchmarks: F1=0.731, MCC=0.454, Spec=0.78, Sens=0.681)")
        print(f"{'='*65}")
        pm = ModelDiagnostics.calculate_phase_metrics(phases_true, phases_pred)
        print(f"  F1          : {pm['phase_f1']:.4f}   (paper: 0.731)")
        print(f"  MCC         : {pm['mcc']:.4f}   (paper: 0.454)")
        print(f"  Specificity : {pm['specificity']:.4f}   (paper: 0.780)")
        print(f"  Sensitivity : {pm['sensitivity']:.4f}   (paper: 0.681)")
        print(f"  Confusion matrix:\n{pm['confusion_matrix']}")


def print_transition_metrics(phases_true, phases_pred, time_points, reactors):
    """Print COSMIC-paper-aligned transition timing metrics."""
    if phases_true is None:
        return
    tm = ModelDiagnostics.calculate_transition_metrics(phases_true, phases_pred, time_points)

    print(f"\n{'='*65}")
    print("Transition-Time Metrics  (primary metric per supervisor)")
    print(f"{'='*65}")
    print(f"  Mean transition MAE    : {tm['transition_mae_days']:.2f} ± {tm['transition_std_days']:.2f} days")
    print(f"  f(t) accuracy  ±0.1   : {tm['f_accuracy_01']*100:.1f}%  (paper benchmark: 72%)")
    print(f"  f(t) accuracy  ±0.2   : {tm['f_accuracy_02']*100:.1f}%  (paper benchmark: 91%)")
    auc = calculate_phase_auc(phases_true, phases_pred, time_points)
    print(f"  Phase AUC MAE          : {auc['auc_mae']:.2f} ± {auc['auc_std']:.2f} days")
    print(f"    (AUC = integral of f(t) dt — captures both timing and sharpness of transition)")

    print(f"\n  Per-reactor transition day (actual → predicted) and AUC:")
    for r, name in enumerate(reactors):
        t_true   = tm['true_transition_days'][r]
        t_pred   = tm['pred_transition_days'][r]
        err      = tm['transition_errors_per_reactor'][r]
        a_true   = auc['auc_true'][r]
        a_pred   = auc['auc_pred'][r]
        a_err    = auc['auc_errors'][r]
        print(f"    {name}: day {t_true:.1f}→{t_pred:.1f} (err={err:.1f}d) | "
              f"AUC {a_true:.2f}→{a_pred:.2f} (err={a_err:.2f}d)")


def plot_trajectories(y_true, y_pred, phases_true, phases_pred, reactors, out_dir):
    """One figure per reactor: predicted vs actual for key metabolites."""
    key_comps = [0, 2, 3, IDX_TITER, 6]  # CD, Glc, Lac, Titer, Gln
    key_names = [COMPONENT_NAMES[i] for i in key_comps]
    n_reactors = y_true.shape[0]
    T = y_true.shape[1]
    t = np.linspace(0, 1, T)

    for r in range(n_reactors):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        name = reactors[r] if r < len(reactors) else f"Reactor {r}"

        for i, (cidx, cname) in enumerate(zip(key_comps, key_names)):
            ax = axes[i]
            ax.plot(t, y_true[r, :, cidx], 'o-', color='#1565C0', label='Actual', markersize=4)
            ax.plot(t, y_pred[r, :, cidx], 's--', color='#E53935', label='Predicted', markersize=4)
            ax.set_title(cname, fontsize=9)
            ax.set_xlabel('Normalised time')
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)

        # Phase subplot
        ax = axes[5]
        if phases_true is not None:
            ax.plot(t, phases_true[r], 'o-', color='#1565C0', label='Actual f', markersize=4)
        ax.plot(t, phases_pred[r, :, 0], 's--', color='#E53935', label='Predicted f', markersize=4)
        ax.axhline(0.2, color='gray', linewidth=0.7, linestyle=':')
        ax.axhline(0.8, color='gray', linewidth=0.7, linestyle=':')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Phase f(t)')
        ax.set_xlabel('Normalised time')
        ax.legend(fontsize=7)

        fig.suptitle(f'{name} — Predicted vs Actual', fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f'eval_{name}.png', dpi=150)
        plt.close(fig)

    print(f"\nTrajectory plots saved to {out_dir}/")


def plot_transition_times(phases_true, phases_pred, time_points, reactors, out_dir):
    """Bar chart: actual vs predicted transition day per reactor."""
    if phases_true is None:
        return
    tm = ModelDiagnostics.calculate_transition_metrics(phases_true, phases_pred, time_points)
    true_t = tm['true_transition_days']
    pred_t = tm['pred_transition_days']
    N = len(reactors)
    x = np.arange(N)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart of actual vs predicted transition day
    ax = axes[0]
    ax.bar(x - w/2, true_t, w, label='Actual',    color='#1565C0', alpha=0.8)
    ax.bar(x + w/2, pred_t, w, label='Predicted', color='#E53935', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(reactors, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Transition day (f crosses 0.5)')
    ax.set_title(f'Transition Time: Actual vs Predicted\n'
                 f'MAE = {tm["transition_mae_days"]:.2f} days')
    ax.legend()
    ax.axhline(true_t.mean(), color='#1565C0', linewidth=0.8, linestyle='--', alpha=0.5)

    # Right: scatter predicted vs actual (identity line = perfect)
    ax = axes[1]
    ax.scatter(true_t, pred_t, s=80, zorder=3, color='#E53935', edgecolors='black', linewidths=0.6)
    for i, name in enumerate(reactors):
        ax.annotate(name, (true_t[i], pred_t[i]),
                    fontsize=7, textcoords='offset points', xytext=(5, 5))
    lo = min(true_t.min(), pred_t.min()) - 0.5
    hi = max(true_t.max(), pred_t.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x  (perfect)')
    ax.set_xlabel('Actual transition day')
    ax.set_ylabel('Predicted transition day')
    ax.set_title(f'Predicted vs Actual Transition Day\n'
                 f'f(t) acc ±0.1: {tm["f_accuracy_01"]*100:.1f}%  '
                 f'±0.2: {tm["f_accuracy_02"]*100:.1f}%')
    ax.legend(fontsize=8)

    fig.suptitle('State Transition Timing  (f(t) crosses 0.5)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'eval_transition_times.png', dpi=150)
    plt.close(fig)
    print('  saved eval_transition_times.png')


def calculate_phase_auc(phases_true, phases_pred, times_days):
    """
    Compute AUC of f(t) per reactor using the trapezoidal rule.

    AUC = integral of f(t) dt over the full run (units: days).
    It captures both transition timing and sharpness in a single number:
    a reactor that transitions earlier or more gradually has a larger AUC.
    Two reactors with identical transition MAE can have very different AUCs
    if one switches sharply and the other drifts slowly.

    phases_true : (N, T)
    phases_pred : (N, T, 1) or (N, T)
    times_days  : (N, T)  actual day values per reactor
    """
    if phases_pred.ndim == 3:
        phases_pred = phases_pred[:, :, 0]
    N = phases_true.shape[0]
    auc_true   = np.array([np.trapz(phases_true[r], times_days[r]) for r in range(N)])
    auc_pred   = np.array([np.trapz(phases_pred[r], times_days[r]) for r in range(N)])
    auc_errors = np.abs(auc_pred - auc_true)
    return {
        'auc_true':   auc_true,
        'auc_pred':   auc_pred,
        'auc_errors': auc_errors,
        'auc_mae':    float(auc_errors.mean()),
        'auc_std':    float(auc_errors.std()),
    }


def collect_metrics(model, dataset, device, times_days):
    """
    Run inference and return a flat dict of all summary-table metrics.

    times_days: (n_reactors, T) actual day values — used for transition MAE.
    """
    y_true, y_pred, phases_true, phases_pred = run_inference(model, dataset, device)

    m = {}
    if phases_true is not None:
        pm = ModelDiagnostics.calculate_phase_metrics(phases_true, phases_pred)
        m['mcc']         = pm['mcc']
        m['f1']          = pm['phase_f1']
        m['specificity'] = pm['specificity']
        m['sensitivity'] = pm['sensitivity']
        tm = ModelDiagnostics.calculate_transition_metrics(phases_true, phases_pred, times_days)
        m['trans_mae'] = tm['transition_mae_days']
        m['f_acc_01']  = tm['f_accuracy_01']
        m['f_acc_02']  = tm['f_accuracy_02']
    if phases_true is not None:
        auc = calculate_phase_auc(phases_true, phases_pred, times_days)
        m['auc_mae'] = auc['auc_mae']
        m['auc_std'] = auc['auc_std']
        m['_auc']    = auc   # full per-reactor data for printing
    actual_final    = y_true[:, -1, IDX_TITER]
    predicted_final = y_pred[:, -1, IDX_TITER]
    frac_err = np.abs(predicted_final - actual_final) / (np.abs(actual_final) + 1e-8)
    m['titer_mean_err'] = float(frac_err.mean())
    m['within10']       = int((frac_err <= 0.10).sum())
    m['n_reactors']     = len(actual_final)
    return m, y_true, y_pred, phases_true, phases_pred


def print_summary_table(real_m, shuffled_m=None, real_loo=None):
    """
    Print the paper-aligned summary table.
    Columns: metric | our model | shuffled baseline | paper benchmark
    Paper benchmarks from Gopalakrishnan et al. (COSMIC dFBA).
    """
    N = real_m['n_reactors']

    def fmt_mae(v):    return f"{v:.2f}d"     if v is not None else "—"
    def fmt_f(v):      return f"{v:.3f}"      if v is not None else "—"
    def fmt_pct(v):    return f"{v*100:.1f}%" if v is not None else "—"
    def fmt_n10(v, n): return f"{v}/{n}"      if v is not None else "—"
    def shuf(key):     return real_m.get(key) if shuffled_m is None else shuffled_m.get(key)

    loo_trans = real_loo.get('trans_mae') if real_loo else None

    rows = [
        # (label, ours, shuffled, paper)
        ("Transition MAE (LOO)",  fmt_mae(loo_trans),                    "—",                           "—"),
        ("Transition MAE (full)", fmt_mae(real_m.get('trans_mae')),       fmt_mae(shuf('trans_mae')),     "—"),
        ("MCC",                   fmt_f(real_m.get('mcc')),               fmt_f(shuf('mcc')),             "0.454"),
        ("F1",                    fmt_f(real_m.get('f1')),                fmt_f(shuf('f1')),              "0.731"),
        ("Specificity",           fmt_f(real_m.get('specificity')),       fmt_f(shuf('specificity')),     "0.780"),
        ("Sensitivity",           fmt_f(real_m.get('sensitivity')),       fmt_f(shuf('sensitivity')),     "0.681"),
        ("f(t) ±0.1 accuracy",    fmt_pct(real_m.get('f_acc_01')),        fmt_pct(shuf('f_acc_01')),      "72.3%"),
        ("f(t) ±0.2 accuracy",    fmt_pct(real_m.get('f_acc_02')),        fmt_pct(shuf('f_acc_02')),      "90.8%"),
        ("Phase AUC MAE (days)",   fmt_mae(real_m.get('auc_mae')),         fmt_mae(shuf('auc_mae')),       "—"),
        ("Final titer mean error",fmt_pct(real_m.get('titer_mean_err')),  fmt_pct(shuf('titer_mean_err')),"—"),
        ("Within 10% titer",      fmt_n10(real_m.get('within10'), N),     fmt_n10(shuf('within10'), N) if shuffled_m else "—", "—"),
    ]

    c0 = max(len(r[0]) for r in rows)
    c1 = max(len(r[1]) for r in rows) + 2
    c2 = max(len(r[2]) for r in rows) + 2
    c3 = max(len(r[3]) for r in rows) + 2
    hdr = f"{'Metric':<{c0}}  {'Our model':<{c1}}  {'Shuffled':<{c2}}  {'Paper':<{c3}}"
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{'-'*len(hdr)}")
    for r in rows:
        print(f"{r[0]:<{c0}}  {r[1]:<{c1}}  {r[2]:<{c2}}  {r[3]:<{c3}}")
    print(sep)


def print_final_titer_metrics(y_true, y_pred, reactors):
    """
    Paper-aligned titer metric: predicted vs actual final (day-13) titer.
    The COSMIC paper reports 'within 10% of measured data' as its titer benchmark.
    All values are in normalised space; fractional errors are still meaningful.
    """
    actual_final    = y_true[:, -1, IDX_TITER]
    predicted_final = y_pred[:, -1, IDX_TITER]
    frac_errors     = np.abs(predicted_final - actual_final) / (np.abs(actual_final) + 1e-8)
    within_10pct    = (frac_errors <= 0.10).sum()
    within_20pct    = (frac_errors <= 0.20).sum()
    N               = len(reactors)

    print(f"\n{'='*65}")
    print("Final Titer Metrics  (paper benchmark: within 10% of measured)")
    print(f"{'='*65}")
    print(f"  Within 10% : {within_10pct}/{N}  ({within_10pct/N*100:.0f}%)   (paper: 10/10 → 100%)")
    print(f"  Within 20% : {within_20pct}/{N}  ({within_20pct/N*100:.0f}%)")
    print(f"  Mean fractional error : {frac_errors.mean()*100:.1f}%")
    print(f"\n  Per-reactor (actual → predicted, error%):")
    for i, name in enumerate(reactors):
        print(f"    {name}: {actual_final[i]:.3f} → {predicted_final[i]:.3f}  "
              f"({frac_errors[i]*100:.1f}%)")


def plot_paper_comparison(real_m, real_loo, out_dir):
    """
    Bar chart comparing our model's metrics against the paper's COSMIC dFBA values.

    Binary classification (MCC, F1, Spec, Sens) uses LOO values from the
    checkpoint where available — that is the fair comparison since the paper
    evaluates its mechanistic model on all 10 reactors without fitting.
    Continuous f(t) accuracy is compared on the full dataset (same basis as paper).
    """
    PAPER = {
        'MCC':            0.454,
        'F1':             0.731,
        'Specificity':    0.780,
        'Sensitivity':    0.681,
        'f(t) ±0.1':      0.723,
        'f(t) ±0.2':      0.908,
    }

    def _loo_or_full(loo_key, full_key):
        v = real_loo.get(loo_key) if real_loo else None
        return v if v is not None else real_m.get(full_key)

    def _is_loo(loo_key):
        return real_loo is not None and real_loo.get(loo_key) is not None

    # Prefer LOO for binary metrics; fall back to full-dataset
    ours = {
        'MCC':         _loo_or_full('mcc',  'mcc'),
        'F1':          _loo_or_full('f1',   'f1'),
        'Specificity': _loo_or_full('spec', 'specificity'),
        'Sensitivity': _loo_or_full('sens', 'sensitivity'),
        'f(t) ±0.1':   real_m.get('f_acc_01'),
        'f(t) ±0.2':   real_m.get('f_acc_02'),
    }

    # LOO flag per metric — drives annotation
    is_loo = {
        'MCC':         _is_loo('mcc'),
        'F1':          _is_loo('f1'),
        'Specificity': _is_loo('spec'),
        'Sensitivity': _is_loo('sens'),
        'f(t) ±0.1':   False,
        'f(t) ±0.2':   False,
    }

    labels = list(PAPER.keys())
    paper_vals = [PAPER[k] for k in labels]
    our_vals   = [ours[k]  for k in labels]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_paper = ax.bar(x - w/2, paper_vals, w, label='COSMIC dFBA (paper)',
                        color='#1565C0', alpha=0.85)
    bars_ours  = ax.bar(x + w/2, our_vals,   w, label='Our NN surrogate',
                        color='#E53935', alpha=0.85)

    # Value labels on bars
    for bar in bars_paper:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8,
                color='#1565C0')
    for bar, key in zip(bars_ours, labels):
        suffix = ' (LOO)' if is_loo[key] else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}{suffix}', ha='center', va='bottom',
                fontsize=8, color='#E53935')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('NN Surrogate vs COSMIC dFBA (Gopalakrishnan et al.)\n'
                 'Binary metrics: LOO where available; f(t) accuracy: full dataset',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / 'eval_paper_comparison.png', dpi=150)
    plt.close(fig)
    print('  saved eval_paper_comparison.png')


def plot_titer_summary(y_true, y_pred, reactors, out_dir):
    """Scatter of predicted vs actual final titer across all reactors."""
    actual_final    = y_true[:, -1, IDX_TITER]
    predicted_final = y_pred[:, -1, IDX_TITER]
    frac_errors     = np.abs(predicted_final - actual_final) / (np.abs(actual_final) + 1e-8)
    within_10 = (frac_errors <= 0.10).sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual_final, predicted_final, s=60, zorder=3)
    for i, name in enumerate(reactors):
        ax.annotate(name, (actual_final[i], predicted_final[i]),
                    fontsize=7, textcoords='offset points', xytext=(4, 4))
    lo = min(actual_final.min(), predicted_final.min()) - 0.05
    hi = max(actual_final.max(), predicted_final.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x  (perfect)')
    # ±10% bands
    ax.fill_between([lo, hi], [lo*0.9, hi*0.9], [lo*1.1, hi*1.1],
                    alpha=0.1, color='green', label='±10% band')
    ax.set_xlabel('Actual final titer (normalised)')
    ax.set_ylabel('Predicted final titer (normalised)')
    ax.set_title(f'Final Titer: Predicted vs Actual\n'
                 f'{within_10}/{len(reactors)} within 10%  (paper benchmark: 10/10)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'eval_titer_scatter.png', dpi=150)
    plt.close(fig)
    print("  saved eval_titer_scatter.png")


# ── Conformal prediction ──────────────────────────────────────────────────────
def build_conformal_intervals(conformal_cal, y_pred, alpha=0.1):
    """
    Split conformal and adjusted (normalised) conformal prediction intervals.

    conformal_cal: list of dicts with 'y_true', 'y_pred' from LOO folds
    y_pred: (N, T, C) — predictions to wrap with intervals
    alpha:  miscoverage rate (0.1 → 90% coverage)

    Split conformal: fixed-width intervals from raw LOO residual quantile.
    Adjusted conformal: locally adaptive intervals — sigma derived from the
    per-(timepoint, component) std of LOO residuals, so the interval width
    reflects how consistently wrong the model is at each point in time.

    Returns dict with 'split' and 'adjusted' keys, each with 'lower'/'upper'
    arrays of shape (N, T, C).
    """
    if not conformal_cal:
        return None

    cal_true = np.concatenate([c['y_true'] for c in conformal_cal], axis=0)
    cal_pred = np.concatenate([c['y_pred'] for c in conformal_cal], axis=0)
    raw_residuals = np.abs(cal_true - cal_pred)   # (n_cal, T, C)

    n_cal = raw_residuals.shape[0]
    level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)

    # Split conformal: one quantile per (T, C) position
    q_split = np.quantile(raw_residuals, level, axis=0)   # (T, C)
    split = {
        'lower': y_pred - q_split,
        'upper': y_pred + q_split,
        'q': q_split,
    }

    # Adjusted conformal: sigma = std of LOO residuals per (T, C)
    # Normalise calibration scores by this sigma, then scale new intervals
    sigma_cal = np.std(raw_residuals, axis=0) + 1e-8       # (T, C)
    norm_residuals = raw_residuals / sigma_cal              # (n_cal, T, C)
    q_adj = np.quantile(norm_residuals, level, axis=0)     # (T, C)
    adjusted = {
        'lower': y_pred - q_adj * sigma_cal,
        'upper': y_pred + q_adj * sigma_cal,
        'q': q_adj,
        'sigma': sigma_cal,
    }

    return {'split': split, 'adjusted': adjusted}


def plot_conformal_titer(y_true, y_pred, intervals, reactors, out_dir):
    """Per-reactor titer trajectory with split and adjusted conformal bands."""
    n = y_true.shape[0]
    T = y_true.shape[1]
    t = np.linspace(0, 1, T)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes = axes.flatten()

    for r in range(n):
        ax = axes[r]
        name = reactors[r] if r < len(reactors) else f"R{r}"

        ax.plot(t, y_true[r, :, IDX_TITER], 'o-', color='#1565C0',
                label='Actual', markersize=5, zorder=4)
        ax.plot(t, y_pred[r, :, IDX_TITER], 's--', color='#E53935',
                label='Predicted', markersize=4, zorder=3)

        if intervals and intervals['split']:
            lo = intervals['split']['lower'][r, :, IDX_TITER]
            hi = intervals['split']['upper'][r, :, IDX_TITER]
            ax.fill_between(t, lo, hi, alpha=0.15, color='#E53935', label='Split 90% CI')

        if intervals and intervals['adjusted']:
            lo = intervals['adjusted']['lower'][r, :, IDX_TITER]
            hi = intervals['adjusted']['upper'][r, :, IDX_TITER]
            ax.fill_between(t, lo, hi, alpha=0.25, color='#FF6F00', label='Adjusted 90% CI')


        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Normalised time', fontsize=7)
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.legend(fontsize=6)

    fig.suptitle('Titer Trajectories with Conformal Prediction Intervals', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'eval_conformal_titer.png', dpi=150)
    plt.close(fig)
    print('  saved eval_conformal_titer.png')


def plot_problem_reactors(y_true, y_pred, phases_true, phases_pred,
                          reactors, out_dir):
    """
    Deep-dive plots for reactors with negative titer Spearman.
    Shows all 25 components side by side for the problem cases.
    """
    from scipy.stats import spearmanr

    # Identify problem reactors (titer Spearman < 0)
    problem = []
    for r in range(y_true.shape[0]):
        t = y_true[r, :, IDX_TITER]
        p = y_pred[r, :, IDX_TITER]
        if t.std() > 1e-8 and p.std() > 1e-8:
            rho, _ = spearmanr(t, p)
            if rho < 0:
                problem.append((r, rho))

    if not problem:
        print('  No problem reactors found (all titer Spearman ≥ 0)')
        return

    print(f'  Problem reactors (negative titer Spearman): '
          f'{[reactors[r] for r, _ in problem]}')

    T = y_true.shape[1]
    t = np.linspace(0, 1, T)
    ncols = 5
    nrows = 5  # 25 components in a 5×5 grid

    for r_idx, rho in problem:
        name = reactors[r_idx] if r_idx < len(reactors) else f"R{r_idx}"
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
        axes = axes.flatten()

        for c in range(25):
            ax = axes[c]
            ax.plot(t, y_true[r_idx, :, c], 'o-', color='#1565C0',
                    markersize=3, linewidth=1, label='Actual')
            ax.plot(t, y_pred[r_idx, :, c], 's--', color='#E53935',
                    markersize=3, linewidth=1, label='Predicted')
            cname = COMPONENT_NAMES[c] if c < len(COMPONENT_NAMES) else f'comp_{c}'
            # Highlight titer
            color = '#B71C1C' if c == IDX_TITER else 'black'
            ax.set_title(cname, fontsize=7, color=color)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.legend(fontsize=6)

        fig.suptitle(f'{name} — All Components  |  Titer Spearman = {rho:.3f}',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f'eval_problem_{name}.png', dpi=150)
        plt.close(fig)
        print(f'  saved eval_problem_{name}.png')

        # Also plot the phase trajectory for this reactor
        if phases_true is not None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, phases_true[r_idx], 'o-', color='#1565C0', label='Actual f')
            ax.plot(t, phases_pred[r_idx, :, 0], 's--', color='#E53935', label='Predicted f')
            ax.axhline(0.2, color='gray', linestyle=':', linewidth=0.8)
            ax.axhline(0.8, color='gray', linestyle=':', linewidth=0.8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Normalised time')
            ax.set_ylabel('f (phase fraction)')
            ax.set_title(f'{name} — Phase Trajectory')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f'eval_problem_{name}_phase.png', dpi=150)
            plt.close(fig)
            print(f'  saved eval_problem_{name}_phase.png')


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          default=str(here / 'improved_model.pt'))
    parser.add_argument('--shuffled-model', default=str(here / 'shuffled_model.pt'),
                        help='Path to permutation-baseline model for comparison table')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data — include all feature files so n_params matches the trained model
    doe_file   = str(here / 'data' / 'data_1.csv')
    rates_file = str(here / 'data' / 'data_3.csv')
    fba_file   = str(here / 'data' / 'data_4.csv')
    trajs, times, ics, meta = load_experimental_data(
        str(here / 'data' / 'data_2.csv'),
        doe_file=doe_file, rates_file=rates_file, fba_file=fba_file)
    reactors = list(meta['reactors'])
    phases   = meta['phases']
    doe_arr  = meta.get('doe_params')
    rate_arr = meta.get('specific_rates')
    fba_arr  = meta.get('fba_efficiencies')

    parameters = {}
    if doe_arr  is not None:
        parameters.update({'O2': doe_arr[:, 0], 'AAs': doe_arr[:, 1], 'Glc': doe_arr[:, 2]})
    if rate_arr is not None:
        for k in range(rate_arr.shape[1]):
            parameters[f'rate_{k}'] = rate_arr[:, k]
    if fba_arr  is not None:
        for k in range(fba_arr.shape[1]):
            parameters[f'fba_{k}'] = fba_arr[:, k]

    dataset = dFBADataset(trajs, times, ics, parameters=parameters,
                          normalize=True, phases=phases)

    # Load real model
    model, ckpt = load_model(args.model, device)
    conformal_cal = ckpt.get('conformal_cal', [])
    real_loo = {
        'trans_mae': ckpt.get('loo_mean_trans_mae'),
        'mcc':       ckpt.get('loo_mean_mcc'),
        'f1':        ckpt.get('loo_mean_f1'),
        'spec':      ckpt.get('loo_mean_spec'),
        'sens':      ckpt.get('loo_mean_sens'),
    }

    # Collect real-model metrics and inference outputs
    real_m, y_true, y_pred, phases_true, phases_pred = collect_metrics(
        model, dataset, device, times)

    # Shuffled baseline (optional)
    shuffled_m = None
    shuffled_path = Path(args.shuffled_model)
    if shuffled_path.exists():
        print(f"\nLoading shuffled baseline from {shuffled_path}...")
        shuffled_model, _ = load_model(str(shuffled_path), device)
        shuffled_m, *_ = collect_metrics(shuffled_model, dataset, device, times)

    # ── Summary table (primary output) ───────────────────────────────────────
    print_summary_table(real_m, shuffled_m, real_loo)

    # ── Detailed per-metric breakdowns ───────────────────────────────────────
    print_metrics(y_true, y_pred, phases_true, phases_pred, reactors)
    print_final_titer_metrics(y_true, y_pred, reactors)
    print_transition_metrics(phases_true, phases_pred, times, reactors)

    # Conformal intervals from LOO calibration residuals
    intervals = build_conformal_intervals(conformal_cal, y_pred, alpha=0.1)
    if intervals:
        print(f"\nConformal prediction (90% coverage):")
        q = intervals['split']['q'][:, IDX_TITER]
        print(f"  Split interval half-width (titer, mean over time)   : ±{q.mean():.4f}")
        q_adj = intervals['adjusted']['q'][:, IDX_TITER]
        print(f"  Adjusted interval scale   (titer, mean over time)   : ×{q_adj.mean():.4f} × σ_cal")

    # Plots
    out_dir = here / 'figures'
    out_dir.mkdir(exist_ok=True)
    plot_paper_comparison(real_m, real_loo, out_dir)
    plot_trajectories(y_true, y_pred, phases_true, phases_pred, reactors, out_dir)
    plot_transition_times(phases_true, phases_pred, times, reactors, out_dir)
    plot_titer_summary(y_true, y_pred, reactors, out_dir)
    plot_conformal_titer(y_true, y_pred, intervals, reactors, out_dir)
    plot_problem_reactors(y_true, y_pred, phases_true, phases_pred, reactors, out_dir)


if __name__ == '__main__':
    main()
