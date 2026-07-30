#!/usr/bin/env python3
"""
Plot training diagnostics from the CSV log produced by train.py.

Generates:
  1. Loss curve (train + val)
  2. Per-feature sigma traces over epochs
  3. Per-feature MSE traces over epochs

Usage:
    !python plot_training.py
    !python plot_training.py --log model_v2.csv
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FEATURE_NAMES = ['CellDensity', 'CellSize', 'Titer',
                 'Glucose', 'Glutamine', 'Asparagine', 'Serine', 'Glycine']


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=str(here / 'model_v2.csv'))
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    out_dir = Path(args.log).parent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(df['epoch'], df['train_loss'], label='Train')
    ax1.plot(df['epoch'], df['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()

    # Log scale version
    ax2.plot(df['epoch'], df['train_loss'], label='Train')
    ax2.plot(df['epoch'], df['val_loss'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log)')
    ax2.set_yscale('log')
    ax2.set_title('Loss Curve (log scale)')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / 'diag_loss.png', dpi=150, bbox_inches='tight')
    print(f'Saved {out_dir / "diag_loss.png"}')

    # Sigma traces
    sigma_cols = [c for c in df.columns if c.startswith('sigma_')]
    if sigma_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in sigma_cols:
            name = col.replace('sigma_', '')
            ax.plot(df['epoch'], df[col], label=name, lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sigma (exp(log_sigma))')
        ax.set_title('Per-feature Sigma Over Training')
        ax.legend(fontsize=8)
        ax.axhline(1.0, color='k', lw=0.5, ls=':')
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_sigma.png', dpi=150, bbox_inches='tight')
        print(f'Saved {out_dir / "diag_sigma.png"}')

    # Per-feature MSE traces
    mse_cols = [c for c in df.columns if c.startswith('mse_')]
    if mse_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in mse_cols:
            name = col.replace('mse_', '')
            ax.plot(df['epoch'], df[col], label=name, lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE (normalized)')
        ax.set_title('Per-feature MSE Over Training')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / 'diag_mse.png', dpi=150, bbox_inches='tight')
        print(f'Saved {out_dir / "diag_mse.png"}')


if __name__ == '__main__':
    main()
