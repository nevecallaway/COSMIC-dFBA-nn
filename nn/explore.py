#!/usr/bin/env python3
"""
Exploration figures for COSMIC-dFBA surrogate model.

New questions answered here (beyond diagnose.py):
  1. Does the model beat a nearest-neighbor baseline?
  2. Are problem reactors isolated in latent space?
  3. Is model error correlated with distance from the DoE centroid?
  4. Do problem reactors cluster separately by trajectory shape?
  5. What do all titer trajectories look like overlaid?
  6. Are prediction errors random over time, or structured?
  7. Are problem reactors outliers in IC space?

Usage:
    python nn/explore.py
    python nn/explore.py --model nn/improved_model.pt

Outputs 7 figures to nn/figures/explore_*.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from model import CosmicNNSurrogateEnhanced, CosmicNNSurrogateLSTM, dFBADataset, dfba_collate_fn
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
    doe_arr  = meta.get('doe_params')
    rate_arr = meta.get('specific_rates')

    parameters = {}
    if doe_arr is not None:
        parameters.update({'O2': doe_arr[:, 0], 'AAs': doe_arr[:, 1], 'Glc': doe_arr[:, 2]})
    if rate_arr is not None:
        for k in range(rate_arr.shape[1]):
            parameters[f'rate_{k}'] = rate_arr[:, k]

    dataset = dFBADataset(trajs, times, ics, parameters=parameters,
                          normalize=True, phases=meta['phases'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    hp     = ckpt['hyperparams']
    if hp.get('arch', 'transformer') == 'lstm':
        model = CosmicNNSurrogateLSTM(
            n_components=hp['n_components'], n_params=hp['n_params'],
            latent_dim=hp['latent_dim'],    n_layers=hp.get('n_layers', 2))
    else:
        model = CosmicNNSurrogateEnhanced(
            n_components=hp['n_components'], n_params=hp['n_params'],
            latent_dim=hp['latent_dim'],    n_heads=hp['n_heads'])
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        collate_fn=dfba_collate_fn)
    batch = next(iter(loader))

    with torch.no_grad():
        out    = model(batch['initial_conditions'].to(device),
                       batch['time'].to(device),
                       batch['parameters'].to(device))
        latent = model.encoder(batch['initial_conditions'].to(device),
                               batch['parameters'].to(device))

    y_true    = batch['trajectory'].numpy()           # (N, T, C)
    y_pred    = out['concentrations'].cpu().numpy()   # (N, T, C)
    latent_np = latent.cpu().numpy()                  # (N, latent_dim)

    return y_true, y_pred, latent_np, reactors, doe_arr


# ── Shared helpers ────────────────────────────────────────────────────────────

def spearman_per_reactor(y_true, y_pred):
    """Returns (N, C) Spearman rho matrix (within-reactor, across timepoints)."""
    N, T, C = y_true.shape
    rho = np.full((N, C), np.nan)
    for r in range(N):
        for c in range(C):
            t, p = y_true[r, :, c], y_pred[r, :, c]
            if t.std() > 1e-8 and p.std() > 1e-8:
                rho[r, c], _ = spearmanr(t, p)
    return rho


def titer_spearman(y_true, y_pred):
    """Returns (N,) titer Spearman rho per reactor."""
    return spearman_per_reactor(y_true, y_pred)[:, IDX_TITER]


def nearest_neighbor_predictor(y_true):
    """
    For each reactor, predict using the most similar other reactor's trajectory.
    Similarity = Pearson correlation of the full flattened trajectory.
    Returns (pred array, list of nn indices).
    """
    N = y_true.shape[0]
    flat = y_true.reshape(N, -1)
    pred = np.zeros_like(y_true)
    nn_idx = []
    for r in range(N):
        others = [i for i in range(N) if i != r]
        sims = [np.corrcoef(flat[r], flat[o])[0, 1] for o in others]
        best = others[int(np.argmax(sims))]
        pred[r] = y_true[best]
        nn_idx.append(best)
    return pred, nn_idx


# ── 1. Nearest-neighbor baseline ─────────────────────────────────────────────

def plot_nn_baseline(y_true, y_pred, reactors, out_dir):
    nn_pred, nn_idx = nearest_neighbor_predictor(y_true)
    mean_pred = np.repeat(y_true.mean(axis=0, keepdims=True), y_true.shape[0], axis=0)

    methods = {
        'Model':            y_pred,
        'Nearest Neighbor': nn_pred,
        'Mean Trajectory':  mean_pred,
    }

    N = y_true.shape[0]
    x = np.arange(N)
    width = 0.25
    colors = ['#1565C0', '#E53935', '#43A047']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-reactor titer Spearman
    ax = axes[0]
    for i, (name, pred) in enumerate(methods.items()):
        rhos = titer_spearman(y_true, pred)
        ax.bar(x + i * width, rhos, width, label=name, color=colors[i], alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x + width)
    ax.set_xticklabels(reactors, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Titer Spearman ρ')
    ax.set_title('Per-reactor Titer Spearman: Model vs Baselines')
    ax.legend(fontsize=8)

    # Right: mean ± std summary
    ax = axes[1]
    means = [np.nanmean(titer_spearman(y_true, p)) for p in methods.values()]
    stds  = [np.nanstd(titer_spearman(y_true, p))  for p in methods.values()]
    ax.bar(list(methods.keys()), means, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Mean Titer Spearman ρ ± std')
    ax.set_title('Summary: Model vs Baselines')

    fig.tight_layout()
    fig.savefig(out_dir / 'explore_1_nn_baseline.png', dpi=150)
    plt.close(fig)
    print('  saved explore_1_nn_baseline.png')

    print('\n  Nearest-neighbor assignments:')
    for r, nn in enumerate(nn_idx):
        rho_model = titer_spearman(y_true, y_pred)[r]
        rho_nn    = titer_spearman(y_true, nn_pred)[r]
        print(f'    {reactors[r]} → {reactors[nn]}  '
              f'(model ρ={rho_model:+.3f}, NN ρ={rho_nn:+.3f})')


# ── 2. Latent space PCA ───────────────────────────────────────────────────────

def plot_latent_pca(latent_np, y_true, y_pred, reactors, out_dir):
    titer_rho = titer_spearman(y_true, y_pred)

    pca = PCA(n_components=2)
    z   = pca.fit_transform(latent_np)

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[:, 0], z[:, 1],
               c=titer_rho, cmap=cmap, norm=norm,
               s=160, edgecolors='black', linewidths=0.8, zorder=3)
    for r, name in enumerate(reactors):
        ax.annotate(name, (z[r, 0], z[r, 1]),
                    fontsize=8, textcoords='offset points', xytext=(5, 5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Titer Spearman ρ')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Latent Space PCA\n(color = titer Spearman, red=poor, green=good)')
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_2_latent_pca.png', dpi=150)
    plt.close(fig)

    var = pca.explained_variance_ratio_
    print(f'  saved explore_2_latent_pca.png  '
          f'(PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%)')


# ── 3. DoE distance from centroid vs Spearman ────────────────────────────────

def plot_doe_distance(y_true, y_pred, doe_arr, reactors, out_dir):
    if doe_arr is None:
        print('  (skipping DoE distance — no DoE data)')
        return

    titer_rho = titer_spearman(y_true, y_pred)
    centroid  = doe_arr.mean(axis=0)
    dists     = np.linalg.norm(doe_arr - centroid, axis=1)

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dists, titer_rho, c=titer_rho, cmap=cmap, norm=norm,
               s=120, edgecolors='black', linewidths=0.5, zorder=3)
    for r, name in enumerate(reactors):
        ax.annotate(name, (dists[r], titer_rho[r]),
                    fontsize=7, textcoords='offset points', xytext=(4, 4))
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    if len(dists) > 2:
        m, b = np.polyfit(dists, titer_rho, 1)
        xs = np.linspace(dists.min(), dists.max(), 50)
        ax.plot(xs, m * xs + b, 'k--', linewidth=1, alpha=0.5,
                label=f'linear trend (slope={m:.2f})')
        ax.legend(fontsize=8)

    ax.set_xlabel('L2 distance from DoE centroid')
    ax.set_ylabel('Titer Spearman ρ')
    ax.set_title('DoE Distance vs Model Performance\n'
                 '(negative slope = harder to predict far from centre)')
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_3_doe_distance.png', dpi=150)
    plt.close(fig)
    print('  saved explore_3_doe_distance.png')


# ── 4. Titer trajectory clustering ───────────────────────────────────────────

def plot_trajectory_clustering(y_true, y_pred, reactors, out_dir):
    N = y_true.shape[0]
    titer = y_true[:, :, IDX_TITER]   # (N, T)
    titer_rho = titer_spearman(y_true, y_pred)

    # Correlation-based dissimilarity
    corr = np.corrcoef(titer)                    # (N, N)
    corr = (corr + corr.T) / 2                   # enforce symmetry (fp rounding)
    dist = np.clip(1 - corr, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist), method='average')

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dendrogram
    ax = axes[0]
    dend = dendrogram(Z, labels=reactors, ax=ax,
                      leaf_rotation=45, leaf_font_size=9,
                      color_threshold=0)
    for tick, leaf_idx in zip(ax.get_xticklabels(), dend['leaves']):
        tick.set_color(cmap(norm(titer_rho[leaf_idx])))
    ax.set_title('Titer Trajectory Clustering\n(leaf color = titer Spearman)')
    ax.set_ylabel('Dissimilarity (1 − Pearson r)')

    # Correlation heatmap ordered by dendrogram
    ax = axes[1]
    order = dend['leaves']
    corr_ord = corr[np.ix_(order, order)]
    im = ax.imshow(corr_ord, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    names_ord = [reactors[i] for i in order]
    ax.set_xticks(range(N)); ax.set_xticklabels(names_ord, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(names_ord, fontsize=8)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('Titer Correlation Matrix (clustered order)')

    fig.tight_layout()
    fig.savefig(out_dir / 'explore_4_trajectory_clustering.png', dpi=150)
    plt.close(fig)
    print('  saved explore_4_trajectory_clustering.png')


# ── 5. All titer trajectories overlaid ───────────────────────────────────────

def plot_all_titer(y_true, y_pred, reactors, out_dir):
    N, T, _ = y_true.shape
    t = np.linspace(0, 1, T)
    titer_rho = titer_spearman(y_true, y_pred)

    cmap = plt.cm.tab10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, arr) in zip(axes, [('Actual', y_true), ('Predicted', y_pred)]):
        for r in range(N):
            ax.plot(t, arr[r, :, IDX_TITER],
                    marker='o', markersize=4, linewidth=1.5,
                    color=cmap(r / max(N - 1, 1)),
                    label=f'{reactors[r]} (ρ={titer_rho[r]:+.2f})')
        ax.set_xlabel('Normalised time')
        ax.set_ylabel('Titer (normalised)')
        ax.set_title(f'{label} — All Reactors')
        ax.legend(fontsize=6, loc='upper left')

    fig.suptitle('All Titer Trajectories Overlaid\n'
                 '(ρ = titer Spearman per reactor)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_5_all_titer.png', dpi=150)
    plt.close(fig)
    print('  saved explore_5_all_titer.png')


# ── 6. Titer residual autocorrelation ─────────────────────────────────────────

def plot_residual_autocorr(y_true, y_pred, reactors, out_dir):
    """Structured autocorrelation = model is systematically missing a pattern."""
    residuals = y_pred[:, :, IDX_TITER] - y_true[:, :, IDX_TITER]  # (N, T)
    N, T = residuals.shape
    titer_rho = titer_spearman(y_true, y_pred)
    sig_bound = 1.96 / np.sqrt(T)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for r in range(N):
        ax = axes[r]
        res = residuals[r]
        lags = range(T)
        acf = [1.0 if lag == 0
               else float(np.corrcoef(res[:T - lag], res[lag:])[0, 1])
               for lag in lags]
        ax.bar(list(lags), acf, color='#1565C0', alpha=0.7)
        ax.axhline(0,           color='black', linewidth=0.8)
        ax.axhline( sig_bound,  color='red',   linewidth=0.8, linestyle='--')
        ax.axhline(-sig_bound,  color='red',   linewidth=0.8, linestyle='--')
        name = reactors[r] if r < len(reactors) else f'R{r}'
        ax.set_title(f'{name}  (ρ={titer_rho[r]:+.2f})', fontsize=8)
        ax.set_xlabel('Lag', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle('Titer Residual Autocorrelation\n'
                 '(bars outside dashed lines = structured error, '
                 'not random noise)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_6_residual_autocorr.png', dpi=150)
    plt.close(fig)
    print('  saved explore_6_residual_autocorr.png')


# ── 7. IC space PCA colored by Spearman ──────────────────────────────────────

def plot_ic_pca(y_true, y_pred, reactors, out_dir):
    """Are problem reactors outliers in initial-condition space?"""
    ics = y_true[:, 0, :]   # (N, C) first timepoint ≈ ICs in normalised space
    titer_rho = titer_spearman(y_true, y_pred)

    pca = PCA(n_components=2)
    z   = pca.fit_transform(ics)

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(-1, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[:, 0], z[:, 1],
               c=titer_rho, cmap=cmap, norm=norm,
               s=160, edgecolors='black', linewidths=0.8, zorder=3)
    for r, name in enumerate(reactors):
        ax.annotate(name, (z[r, 0], z[r, 1]),
                    fontsize=8, textcoords='offset points', xytext=(5, 5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Titer Spearman ρ')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('IC Space PCA\n(color = titer Spearman, red=poor, green=good)')
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_7_ic_pca.png', dpi=150)
    plt.close(fig)

    var = pca.explained_variance_ratio_
    print(f'  saved explore_7_ic_pca.png  '
          f'(PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%)')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(here / 'improved_model.pt'))
    args = parser.parse_args()

    out_dir = here / 'figures'
    out_dir.mkdir(exist_ok=True)

    print('Loading model and data...')
    y_true, y_pred, latent_np, reactors, doe_arr = load_everything(
        args.model, here)
    print(f'  {len(reactors)} reactors, {y_true.shape[1]} timepoints, '
          f'{y_true.shape[2]} components\n')

    print('Generating exploration figures...')
    plot_nn_baseline(y_true, y_pred, reactors, out_dir)
    plot_latent_pca(latent_np, y_true, y_pred, reactors, out_dir)
    plot_doe_distance(y_true, y_pred, doe_arr, reactors, out_dir)
    plot_trajectory_clustering(y_true, y_pred, reactors, out_dir)
    plot_all_titer(y_true, y_pred, reactors, out_dir)
    plot_residual_autocorr(y_true, y_pred, reactors, out_dir)
    plot_ic_pca(y_true, y_pred, reactors, out_dir)

    print(f'\nDone. 7 figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
