#!/usr/bin/env python3
"""Production training script for supercomputer - Enhanced COSMIC-dFBA model"""
import sys
from nn.cosmic_nn_surrogate import (
    CosmicNNSurrogateEnhanced, CosmicNNSurrogate, 
    TrainingManager, dFBADataset
)
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
import json
from pathlib import Path

def main(model_type='enhanced', data_dir='data/matlab_exports', use_cuda=True):
    """
    Train COSMIC-dFBA model on real or synthetic data.
    
    Args:
        model_type: 'standard' or 'enhanced'
        data_dir: Directory containing .npz files from MATLAB
        use_cuda: Whether to use GPU if available
    """
    print(f"\n{'='*70}")
    print(f"COSMIC-dFBA Production Training ({model_type.upper()} Model)")
    print(f"{'='*70}")
    
    # Setup device
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Expected .npz files from MATLAB in this directory")
        return
    
    npz_files = list(data_dir.glob('*.npz'))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return
    
    print(f"Found {len(npz_files)} data files")
    
    # Load and combine data
    all_trajectories = []
    all_times = []
    all_ics = []
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            profiles = data['profiles']
            time = data['time'].flatten() if len(data['time'].shape) > 1 else data['time']
            
            all_trajectories.append(profiles)
            all_times.append(time)
            all_ics.append(profiles[0, :])
            print(f"  ✓ Loaded {npz_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {npz_file.name}: {e}")
    
    # Pad trajectories
    max_t = max(len(t) for t in all_times)
    n_sims = len(all_trajectories)
    n_comps = all_trajectories[0].shape[1]
    
    trajectories_padded = np.zeros((n_sims, max_t, n_comps))
    times_padded = np.zeros((n_sims, max_t))
    
    for i, (traj, t) in enumerate(zip(all_trajectories, all_times)):
        trajectories_padded[i, :len(t), :] = traj
        times_padded[i, :len(t)] = t
        # Pad with last value
        if len(t) < max_t:
            trajectories_padded[i, len(t):, :] = traj[-1, :]
            times_padded[i, len(t):] = t[-1]
    
    initial_conditions = np.array(all_ics)
    
    print(f"Data shape: {trajectories_padded.shape}")
    
    # Create dataset
    dataset = dFBADataset(
        trajectories_padded,
        times_padded,
        initial_conditions,
        parameters={},
        normalize=True
    )
    
    # Create dataloaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create model
    if model_type == 'enhanced':
        model = CosmicNNSurrogateEnhanced(
            n_components=dataset.n_components,
            n_params=0,
            latent_dim=64,
            n_heads=4
        )
    else:
        model = CosmicNNSurrogate(
            n_components=dataset.n_components,
            n_params=0,
            latent_dim=64,
            n_heads=4
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = TrainingManager(
        model,
        device=device,
        learning_rate=1e-3,
        model_type=model_type
    )
    
    best_loss = trainer.train(train_loader, val_loader, epochs=200, patience=30)
    
    # Save model
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / f'cosmic_model_{model_type}.pt'
    trainer.save(str(model_path))
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'best_val_loss': float(best_loss),
        'n_components': dataset.n_components,
        'n_training_samples': len(train_dataset),
        'n_validation_samples': len(val_dataset),
        'device': str(device),
        'training_time_seconds': sum(times_padded.max() for _ in range(len(npz_files))),
        'input_files': [str(f) for f in npz_files],
    }
    
    metadata_path = output_dir / f'metadata_{model_type}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model, dataset, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train COSMIC-dFBA model')
    parser.add_argument('--model', type=str, default='enhanced', choices=['standard', 'enhanced'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, default='data/matlab_exports',
                       help='Directory containing .npz data files')
    parser.add_argument('--gpu', type=bool, default=True,
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    main(model_type=args.model, data_dir=args.data, use_cuda=args.gpu)
