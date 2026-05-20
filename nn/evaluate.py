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
from model import CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
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
    ckpt = torch.load(checkpoint_path, map_location=device)
    hp = ckpt['hyperparams']
    model = CosmicNNSurrogateEnhanced(
        n_components=hp['n_components'],
        n_params=hp['n_params'],
        latent_dim=hp['latent_dim'],
        n_heads=hp['n_heads'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Saved LOO R²             : {ckpt.get('loo_mean_r2', 'n/a'):.4f}")
    print(f"  Saved LOO Titer R²       : {ckpt.get('loo_mean_titer_r2', 'n/a'):.4f}")
    print(f"  Saved LOO Spearman       : {ckpt.get('loo_mean_spearman', 'n/a'):.4f}")
    print(f"  Saved LOO Titer Spearman : {ckpt.get('loo_mean_titer_spearman', 'n/a'):.4f}")
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
        print("Phase Metrics")
        print(f"{'='*65}")
        pm = ModelDiagnostics.calculate_phase_metrics(phases_true, phases_pred)
        print(f"  Phase F1 : {pm['phase_f1']:.4f}")
        print(f"  Confusion matrix:\n{pm['confusion_matrix']}")


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


def plot_titer_summary(y_true, y_pred, reactors, out_dir):
    """Scatter of predicted vs actual final titer across all reactors."""
    actual_final    = y_true[:, -1, IDX_TITER]
    predicted_final = y_pred[:, -1, IDX_TITER]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual_final, predicted_final, s=60, zorder=3)
    for i, name in enumerate(reactors):
        ax.annotate(name, (actual_final[i], predicted_final[i]),
                    fontsize=7, textcoords='offset points', xytext=(4, 4))
    lo = min(actual_final.min(), predicted_final.min()) - 0.05
    hi = max(actual_final.max(), predicted_final.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel('Actual final titer (normalised)')
    ax.set_ylabel('Predicted final titer (normalised)')
    ax.set_title('Final Titer: Predicted vs Actual')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'eval_titer_scatter.png', dpi=150)
    plt.close(fig)
    print("  saved eval_titer_scatter.png")


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(here / 'improved_model.pt'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
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

    # Load model
    model, _ = load_model(args.model, device)

    # Inference
    y_true, y_pred, phases_true, phases_pred = run_inference(model, dataset, device)

    # Metrics
    print_metrics(y_true, y_pred, phases_true, phases_pred, reactors)

    # Plots
    out_dir = here / 'figures'
    out_dir.mkdir(exist_ok=True)
    plot_trajectories(y_true, y_pred, phases_true, phases_pred, reactors, out_dir)
    plot_titer_summary(y_true, y_pred, reactors, out_dir)


if __name__ == '__main__':
    main()
