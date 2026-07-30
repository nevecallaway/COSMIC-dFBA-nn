#!/usr/bin/env python3
"""
Exploratory data analysis for COSMIC-dFBA training data.

Generates:
  1. Metabolite concentration histograms (real vs synthetic)
  2. Inter-metabolite correlation heatmap
  3. DoE parameter heatmap (reactor × parameter)
  4. PCA / UMAP of timepoints coloured by phase state
  5. Q-Q plots for key metabolites
  6. Specific-rates correlation heatmap
  9. Model architecture diagram
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_experimental_data

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ─────────────────────────────────────────────────────────────────────────────
COMPONENT_NAMES = [
    'Cell Density', 'Cell Volume', 'Glucose', 'Lactate', 'NH4', 'Titer',
    'Glutamine', 'Glutamate', 'L-Asparagine', 'L-Aspartic acid', 'L-Serine',
    'Glycine', 'L-Alanine', 'L-Proline', 'L-Threonine', 'L-Histidine',
    'L-Lysine', 'L-Valine', 'L-Methionine', 'L-Arginine', 'L-Tyrosine',
    'L-Isoleucine', 'L-Leucine', 'L-Phenylalanine', 'L-Tryptophan',
]

PHASE_COLORS = {
    'growth':     '#2196F3',  # blue
    'transition': '#FF9800',  # orange
    'production': '#E91E63',  # pink/red
}


def phase_label(f):
    """Map continuous f to discrete label string."""
    if f < 0.2:
        return 'growth'
    elif f > 0.8:
        return 'production'
    return 'transition'


def load_data(here: Path):
    trajs, times, ics, meta = load_experimental_data(
        str(here / 'data' / 'data_2.csv'),
        doe_file=str(here / 'data' / 'data_1.csv'),
        rates_file=str(here / 'data' / 'data_3.csv'),
    )
    phases = meta['phases']           # (n_reactors, n_timepoints)
    components = meta['components']   # list of column names from CSV
    doe = meta.get('doe_params')      # (n_reactors, 3) or None
    rates = meta.get('specific_rates') # (n_reactors, 50) or None
    reactors = list(meta['reactors'])
    return trajs, times, phases, components, doe, rates, reactors


def load_synthetic(here: Path):
    npz = np.load(here / 'synthetic_training.npz')
    trajs = npz['trajectories']   # (N, T, C)
    phases = npz.get('phases', None)
    return trajs, phases


# ── 1. Metabolite histograms ──────────────────────────────────────────────────
def plot_histograms(real_flat, synth_flat, components, out_dir):
    n = len(components)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.flatten()

    for i, name in enumerate(components):
        ax = axes[i]
        ax.hist(synth_flat[:, i], bins=40, alpha=0.5, color='steelblue',
                density=True, label='synthetic')
        ax.hist(real_flat[:, i], bins=15, alpha=0.8, color='crimson',
                density=True, label='real')
        ax.set_title(name, fontsize=8)
        ax.set_xlabel('concentration', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Metabolite Distributions — Real vs Synthetic', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / '1_histograms.png', dpi=150)
    plt.close(fig)
    print('  saved 1_histograms.png')


# ── 2. Inter-metabolite correlation heatmap ───────────────────────────────────
def plot_correlation(real_flat, components, out_dir):
    df = pd.DataFrame(real_flat, columns=components)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(components)))
    ax.set_yticks(range(len(components)))
    ax.set_xticklabels(components, rotation=90, fontsize=7)
    ax.set_yticklabels(components, fontsize=7)
    ax.set_title('Inter-Metabolite Pearson Correlation (real data)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '2_correlation_heatmap.png', dpi=150)
    plt.close(fig)
    print('  saved 2_correlation_heatmap.png')


# ── 3. DoE parameter heatmap ──────────────────────────────────────────────────
def plot_doe_heatmap(doe, reactors, out_dir):
    if doe is None:
        print('  (skipping DoE heatmap — data not available)')
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(doe, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='level (−1/0/+1)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['O₂', 'AAs', 'Glc'])
    ax.set_yticks(range(len(reactors)))
    ax.set_yticklabels(reactors, fontsize=8)
    ax.set_title('DoE Parameter Levels by Reactor', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '3_doe_heatmap.png', dpi=150)
    plt.close(fig)
    print('  saved 3_doe_heatmap.png')


# ── 4. Specific-rates heatmap ─────────────────────────────────────────────────
def plot_rates_heatmap(rates, reactors, out_dir):
    if rates is None:
        print('  (skipping rates heatmap — data not available)')
        return

    n_rates = rates.shape[1] // 2
    labels = ([f'G{i}' for i in range(n_rates)] +
              [f'P{i}' for i in range(n_rates)])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(rates.T, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='standardised rate')
    ax.set_yticks(range(rates.shape[1]))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xticks(range(len(reactors)))
    ax.set_xticklabels(reactors, rotation=45, fontsize=8)
    ax.axhline(n_rates - 0.5, color='k', linewidth=1.5, linestyle='--')
    ax.text(len(reactors) + 0.1, n_rates / 2, 'Growth', va='center', fontsize=8)
    ax.text(len(reactors) + 0.1, n_rates + n_rates / 2, 'Prod', va='center', fontsize=8)
    ax.set_title('Phase-Specific Metabolic Rates (standardised)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '4_rates_heatmap.png', dpi=150)
    plt.close(fig)
    print('  saved 4_rates_heatmap.png')


# ── 5. PCA of timepoints coloured by phase ────────────────────────────────────
def plot_pca_phase(real_flat_all, phase_labels_all, out_dir):
    if not HAS_SKLEARN:
        print('  (skipping PCA — sklearn not available)')
        return

    X = StandardScaler().fit_transform(real_flat_all)
    pcs = PCA(n_components=2).fit_transform(X)

    color_map = {'growth': PHASE_COLORS['growth'],
                 'transition': PHASE_COLORS['transition'],
                 'production': PHASE_COLORS['production']}

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ['growth', 'transition', 'production']:
        mask = np.array(phase_labels_all) == label
        if mask.any():
            ax.scatter(pcs[mask, 0], pcs[mask, 1], c=color_map[label],
                       label=label, alpha=0.75, s=40, edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.set_title('PCA of Real Timepoints Coloured by Phase State', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '5_pca_phase.png', dpi=150)
    plt.close(fig)
    print('  saved 5_pca_phase.png')


# ── 6. UMAP of timepoints coloured by phase ───────────────────────────────────
def plot_umap_phase(real_flat_all, phase_labels_all, out_dir):
    if not HAS_UMAP:
        print('  (skipping UMAP — umap-learn not installed)')
        return
    if not HAS_SKLEARN:
        return

    X = StandardScaler().fit_transform(real_flat_all)
    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42).fit_transform(X)

    color_map = {'growth': PHASE_COLORS['growth'],
                 'transition': PHASE_COLORS['transition'],
                 'production': PHASE_COLORS['production']}

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ['growth', 'transition', 'production']:
        mask = np.array(phase_labels_all) == label
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color_map[label],
                       label=label, alpha=0.75, s=40, edgecolors='none')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    ax.set_title('UMAP of Real Timepoints Coloured by Phase State', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '6_umap_phase.png', dpi=150)
    plt.close(fig)
    print('  saved 6_umap_phase.png')


# ── 7. Q-Q plots for key metabolites ─────────────────────────────────────────
KEY_COMPONENTS = ['Cell Density', 'Glucose', 'Titer', 'Lactate', 'L-Arginine', 'Glutamine']

def plot_qq(real_flat, components, out_dir):
    targets = [c for c in KEY_COMPONENTS if c in components]
    if not targets:
        print('  (skipping Q-Q — key components not found)')
        return

    fig, axes = plt.subplots(2, (len(targets) + 1) // 2, figsize=(4 * ((len(targets) + 1) // 2), 6))
    axes = axes.flatten()

    for i, name in enumerate(targets):
        idx = components.index(name)
        data = real_flat[:, idx]
        data = data[np.isfinite(data)]
        stats.probplot(data, dist='norm', plot=axes[i])
        axes[i].set_title(f'Q-Q: {name}', fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(len(targets), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Normal Q-Q Plots — Real Metabolite Concentrations', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / '7_qq_plots.png', dpi=150)
    plt.close(fig)
    print('  saved 7_qq_plots.png')


# ── 8. Phase distribution across reactors ────────────────────────────────────
def plot_phase_distribution(phases, times, reactors, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True)
    axes = axes.flatten()

    for i, (reactor, ax) in enumerate(zip(reactors, axes)):
        t = times[i]
        f = phases[i]
        colors = [PHASE_COLORS[phase_label(fi)] for fi in f]
        ax.scatter(t, f, c=colors, s=20, zorder=3)
        ax.plot(t, f, color='gray', alpha=0.4, linewidth=1)
        ax.axhline(0.2, color='gray', linewidth=0.7, linestyle='--')
        ax.axhline(0.8, color='gray', linewidth=0.7, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(reactor, fontsize=9)
        ax.set_xlabel('Day', fontsize=7)
        if i % 5 == 0:
            ax.set_ylabel('f (phase)', fontsize=7)

    fig.suptitle('Phase Trajectories by Reactor (blue=growth, orange=transition, pink=production)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / '8_phase_trajectories.png', dpi=150)
    plt.close(fig)
    print('  saved 8_phase_trajectories.png')


# ── 9. Model architecture diagram ────────────────────────────────────────────
def plot_architecture(out_dir):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── helpers ───────────────────────────────────────────────────────────────
    def box(cx, cy, w, h, label, sublabel='', color='#E3F2FD', fontsize=8):
        rect = plt.Rectangle((cx - w/2, cy - h/2), w, h,
                              facecolor=color, edgecolor='#546E7A',
                              linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=3)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha='center', va='center', fontsize=6,
                    color='#546E7A', zorder=3)

    def arrow(x0, y0, x1, y1, label='', color='#455A64'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.2, connectionstyle='arc3,rad=0'))
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx + 0.08, my, label, fontsize=6, color='#37474F', va='center')

    def bracket(cx, y_top, y_bot, label, color='#B0BEC5'):
        ax.plot([cx, cx], [y_bot, y_top], color=color, lw=1, linestyle='--', zorder=1)
        ax.text(cx + 0.12, (y_top + y_bot)/2, label,
                fontsize=6.5, color='#546E7A', va='center', style='italic')

    BLUE   = '#E3F2FD'
    GREEN  = '#E8F5E9'
    ORANGE = '#FFF3E0'
    PURPLE = '#F3E5F5'
    PINK   = '#FCE4EC'
    TEAL   = '#E0F2F1'

    # ── Column x positions ────────────────────────────────────────────────────
    X_IN   = 1.3    # inputs
    X_ENC  = 3.5    # encoder
    X_DEC  = 7.0    # decoder internals (left)
    X_RATE = 9.5    # rate predictor
    X_BLEND= 11.2   # blending / integrator
    X_OUT  = 13.0   # outputs

    # ── Inputs ────────────────────────────────────────────────────────────────
    box(X_IN, 7.2, 1.9, 0.65, 'Initial Conditions', 'n_components = 25', BLUE)
    box(X_IN, 5.8, 1.9, 0.65, 'DoE Parameters', 'O₂, AAs, Glc  (3)', BLUE)
    box(X_IN, 4.4, 1.9, 0.65, 'Specific Rates', '25 growth + 25 prod  (50)', BLUE)
    box(X_IN, 2.5, 1.9, 0.65, 'Time Points', '(batch, T)', BLUE)

    # ── DynamicsEncoder ───────────────────────────────────────────────────────
    box(X_ENC, 6.0, 2.0, 2.6, 'DynamicsEncoder',
        'Linear(78→128)→ReLU→Drop\nLinear(128→128)→ReLU→Drop\nLinear(128→64)', GREEN)

    arrow(X_IN + 0.95, 7.2,  X_ENC - 1.0, 6.5,  'cat')
    arrow(X_IN + 0.95, 5.8,  X_ENC - 1.0, 6.0)
    arrow(X_IN + 0.95, 4.4,  X_ENC - 1.0, 5.5,  '')

    # latent vector
    box(X_ENC, 3.8, 1.4, 0.55, 'latent z', 'dim = 64', TEAL, fontsize=7)
    arrow(X_ENC, 4.7, X_ENC, 4.1)

    # ── MultiHeadTemporalDecoder ───────────────────────────────────────────────
    # Time embedding
    box(X_DEC, 2.5, 2.0, 0.65, 'Time Embed',
        'Linear(1→32)→ReLU\n→Linear(32→64)', ORANGE)
    arrow(X_IN + 0.95, 2.5, X_DEC - 1.0, 2.5, '')

    # Attention (cross-attention: time queries latent)
    box(X_DEC, 4.6, 2.0, 0.75, 'MHA',
        'time queries latent\n4 heads, dim=64', ORANGE)
    arrow(X_ENC + 1.0, 3.8, X_DEC - 0.6, 4.3,  'z')
    arrow(X_DEC, 3.15, X_DEC, 4.2, '')  # time embed → attention

    # RatePredictionHead
    box(X_RATE, 6.5, 2.2, 0.75, 'RatePredictionHead',
        'Linear(128→128)→ReLU→Drop\n→Linear(128→25)→Tanh\n× 2  (growth & prod)', PURPLE)
    arrow(X_ENC + 1.0, 3.8,  X_RATE - 1.1, 6.3,  'z')
    arrow(X_DEC + 1.0, 4.6,  X_RATE - 1.1, 6.7,  'attn')

    # StateWeightingLayer (phase predictor)
    box(X_RATE, 4.2, 2.2, 0.75, 'StateWeighting',
        'trigger = conc · W + b\nmodulation = MLP(z)\nf = σ(trigger + mod)', PINK)
    arrow(X_ENC + 1.0, 3.8,  X_RATE - 1.1, 4.2,  'z')

    # First integrator pass (f=0 seed)
    box(X_BLEND, 5.5, 1.8, 0.65, 'Integrator (seed)',
        'f=0 → pure growth rates\nIC + cumsum(r·dt)', TEAL)
    arrow(X_RATE + 1.1, 6.5,  X_BLEND - 0.9, 5.7,  'growth_r')

    # conc → state weighting
    arrow(X_BLEND - 0.1, 5.2, X_RATE + 1.1, 4.4,  'conc₀')

    # Final rate blend
    box(X_BLEND, 3.8, 1.8, 0.75, 'Rate Blend',
        '(1−f)·r_growth + f·r_prod', PURPLE)
    arrow(X_RATE + 1.1, 6.5,  X_BLEND - 0.9, 4.0,  'r_growth')
    arrow(X_RATE + 1.1, 5.5,  X_BLEND - 0.9, 3.9,  'r_prod', color='#AD1457')
    arrow(X_RATE + 1.1, 4.2,  X_BLEND - 0.9, 3.8,  'f', color='#AD1457')

    # Final integrator
    box(X_BLEND, 2.3, 1.8, 0.65, 'Integrator (final)',
        'IC + cumsum(blended_r·dt)', TEAL)
    arrow(X_BLEND, 3.42, X_BLEND, 2.65)

    # ── Outputs ───────────────────────────────────────────────────────────────
    box(X_OUT, 2.3, 1.6, 0.6, 'Concentrations', '(batch, T, 25)', '#C8E6C9', fontsize=8)
    box(X_OUT, 4.2, 1.6, 0.6, 'Phase f(t)', '(batch, T, 1)  ∈ [0,1]', '#F8BBD9', fontsize=8)

    arrow(X_BLEND + 0.9, 2.3,  X_OUT - 0.8, 2.3)
    arrow(X_RATE  + 1.1, 4.2,  X_OUT - 0.8, 4.2,  'f', color='#AD1457')

    # ── Decoder brace ─────────────────────────────────────────────────────────
    bracket(5.3, 7.2, 1.5, 'MultiHeadTemporalDecoder')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(7, 8.7, 'CosmicNNSurrogateEnhanced — Architecture',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#263238')

    fig.tight_layout()
    fig.savefig(out_dir / '9_architecture.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved 9_architecture.png')


# ─────────────────────────────────────────────────────────────────────────────
def main():
    here = Path(__file__).parent
    out_dir = here / 'figures'
    out_dir.mkdir(exist_ok=True)

    print('Loading real experimental data...')
    trajs, times, phases, components, doe, rates, reactors = load_data(here)
    # trajs: (n_reactors, n_timepoints, n_components)

    # Flatten timepoints × reactors → rows for scatter/histogram plots
    nr, nt, nc = trajs.shape
    real_flat = trajs.reshape(-1, nc)             # (nr*nt, nc)
    phase_flat = phases.reshape(-1)               # (nr*nt,)
    phase_labels_all = [phase_label(f) for f in phase_flat]

    print('Loading synthetic data...')
    synth_trajs, _ = load_synthetic(here)
    synth_flat = synth_trajs.reshape(-1, synth_trajs.shape[-1])

    # Trim synthetic to same number of components if mismatch
    synth_flat = synth_flat[:, :nc]

    print('\nGenerating plots...')
    plot_histograms(real_flat, synth_flat, components, out_dir)
    plot_correlation(real_flat, components, out_dir)
    plot_doe_heatmap(doe, reactors, out_dir)
    plot_rates_heatmap(rates, reactors, out_dir)
    plot_pca_phase(real_flat, phase_labels_all, out_dir)
    plot_umap_phase(real_flat, phase_labels_all, out_dir)
    plot_qq(real_flat, components, out_dir)
    plot_phase_distribution(phases, times, reactors, out_dir)
    plot_architecture(out_dir)

    print(f'\nDone. Figures saved to {out_dir}/')

    # Print separability summary
    if HAS_SKLEARN:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score as sk_f1
        mask = (phase_flat < 0.2) | (phase_flat > 0.8)
        X = StandardScaler().fit_transform(real_flat[mask])
        y = (phase_flat[mask] > 0.5).astype(int)
        if len(np.unique(y)) == 2:
            lr = LogisticRegression(max_iter=500).fit(X, y)
            f1 = sk_f1(y, lr.predict(X))
            print(f'\nLinear separability check (logistic regression on unambiguous timepoints):')
            print(f'  In-sample F1 = {f1:.4f}  '
                  f'(1.0 = perfectly linearly separable by concentration alone)')


if __name__ == '__main__':
    main()
