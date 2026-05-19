#!/usr/bin/env python3
"""
Unified Training Script for COSMIC-dFBA.
Combines standard and PINN-enhanced training logic for real and simulated data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import time

from model import CosmicNNSurrogateEnhanced, dFBADataset, dfba_collate_fn
from utils import load_experimental_data, ModelDiagnostics

class Trainer:
    """
    Unified trainer for COSMIC-dFBA that handles both standard and
    Physics-Informed Neural Network (PINN) losses.
    """
    def __init__(self, model, device, learning_rate=5e-4, model_type='enhanced',
                 scheduler_patience=5):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=scheduler_patience
        )
        self.losses = []

    def compute_loss(self, predictions, targets, ics, phases_batch=None):
        """
        Computes the loss based on the model type.
        For 'enhanced' models, it uses the fused PINN loss.
        """
        if self.model_type == 'standard':
            # Standard MSE + Smoothness
            conc_loss = nn.functional.mse_loss(predictions, targets)
            smoothness = 0.1 * torch.mean((predictions[:, 1:, :] - predictions[:, :-1, :]) ** 2)
            return conc_loss + smoothness, {'conc': conc_loss.item()}

        # Enhanced / PINN Fused Loss
        conc_pred = predictions['concentrations']
        phase_logits = predictions['phase_weights']
        growth_rates = predictions['growth_rates']
        prod_rates = predictions['prod_rates']

        # 1. Main Concentration MSE
        conc_loss = nn.functional.mse_loss(conc_pred, targets)

        # 2. IC constraint
        ic_loss = 0.1 * torch.mean((conc_pred[:, 0, :] - ics) ** 2)

        # 3. Non-flatness penalty (Increased to punish flat lines)
        conc_variance = torch.var(conc_pred, dim=1)
        flatness_penalty = 0.2 * torch.mean(1.0 / (1.0 + conc_variance))

        # 4. PINN: Non-negativity penalty
        non_neg_loss = 0.5 * torch.mean(torch.clamp(conc_pred, max=0)**2)

        # 5. PINN: Concentration Smoothness
        conc_smoothness = 0.1 * torch.mean((conc_pred[:, 1:, :] - conc_pred[:, :-1, :]) ** 2)

        # 6. Binary Phase Classification (with masking)
        phase_loss = torch.tensor(0.0, device=targets.device)
        if phases_batch is not None:
            batch_size, n_time = phases_batch.shape
            phase_targets = torch.zeros((batch_size, n_time), dtype=torch.long, device=phases_batch.device)
            mask = torch.zeros((batch_size, n_time), dtype=torch.bool, device=phases_batch.device)

            for b in range(batch_size):
                for t in range(n_time):
                    p = phases_batch[b, t].item()
                    if p < 0.2:
                        phase_targets[b, t] = 0
                        mask[b, t] = True
                    elif p > 0.8:
                        phase_targets[b, t] = 1
                        mask[b, t] = True

            phase_logits_flat = phase_logits.view(-1, 2)
            phase_targets_flat = phase_targets.view(-1)
            mask_flat = mask.view(-1)

            if mask_flat.sum() > 0:
                phase_loss = 0.5 * nn.functional.cross_entropy(
                    phase_logits_flat[mask_flat],
                    phase_targets_flat[mask_flat]
                )

        # 7. PINN: Rate-based constraints
        phase_probs = torch.softmax(phase_logits, dim=-1)
        f = phase_probs[:, :, 1 : 2]
        blended_rates = (1 - f) * growth_rates + f * prod_rates
        rate_smoothness = 0.1 * torch.mean((blended_rates[:, 1:, :] - blended_rates[:, :-1, :]) ** 2)
        rate_magnitude = 0.01 * (torch.mean(torch.abs(growth_rates)) + torch.mean(torch.abs(prod_rates)))

        # 8. Phase smoothness and exploration penalty
        phase_smoothness = 0.05 * torch.mean((phase_logits[:, 1:, :] - phase_logits[:, :-1, :]) ** 2)
        phase_penalty = 0.02 * torch.mean((phase_probs[:, :, 1 : 2] - 0.5) ** 2)

        total_loss = (conc_loss + ic_loss + flatness_penalty + phase_loss +
                      non_neg_loss + conc_smoothness + rate_smoothness +
                      rate_magnitude + phase_smoothness + phase_penalty)

        return total_loss, {
            'conc': conc_loss.item(),
            'ic': ic_loss.item(),
            'phase_ce': phase_loss.item() if isinstance(phase_loss, torch.Tensor) else phase_loss,
            'pinn_non_neg': non_neg_loss.item(),
            'pinn_rate_smooth': rate_smoothness.item(),
        }

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        components = {'conc': 0, 'ic': 0, 'phase_ce': 0, 'pinn_non_neg': 0, 'pinn_rate_smooth': 0}

        for batch in train_loader:
            ic = batch['initial_conditions'].to(self.device)
            time = batch['time'].to(self.device)
            params = batch['parameters'].to(self.device)
            target = batch['trajectory'].to(self.device)
            phases = batch.get('phases', None)
            if phases is not None:
                phases = phases.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(ic, time, params)
            loss, comp = self.compute_loss(predictions, target, ic, phases)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            for k in components:
                components[k] += comp.get(k, 0)

        epoch_loss /= len(train_loader)
        for k in components:
            components[k] /= len(train_loader)

        self.losses.append(epoch_loss)
        self.scheduler.step(epoch_loss)
        return epoch_loss, components

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_targets, all_preds, all_phase_targets, all_phase_preds = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                ic = batch['initial_conditions'].to(self.device)
                time = batch['time'].to(self.device)
                params = batch['parameters'].to(self.device)
                target = batch['trajectory'].to(self.device)

                predictions = self.model(ic, time, params)
                loss, _ = self.compute_loss(predictions, target, ic, None)
                val_loss += loss.item()

                all_targets.append(target.cpu().numpy())
                all_preds.append(predictions['concentrations'].cpu().numpy() if isinstance(predictions, dict) else predictions.cpu().numpy())
                if 'phases' in batch:
                    all_phase_targets.append(batch['phases'].cpu().numpy())
                    all_phase_preds.append(predictions['phase_weights'].cpu().numpy() if isinstance(predictions, dict) else None)

        val_loss /= len(val_loader)
        report = {"val_loss": val_loss}
        if all_targets:
            y_true = np.concatenate(all_targets, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            report["metrics"] = ModelDiagnostics.calculate_regression_metrics(y_true, y_pred)
            if all_phase_targets:
                p_true = np.concatenate(all_phase_targets, axis=0)
                p_pred = np.concatenate(all_phase_preds, axis=0)
                report["phase_metrics"] = ModelDiagnostics.calculate_phase_metrics(p_true, p_pred)

        return report

    def train(self, train_loader, val_loader, epochs=100, patience=20, verbose=True):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, comps = self.train_epoch(train_loader)
            report = self.validate(val_loader)
            val_loss = report["val_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                status = "(BEST)"
            else:
                patience_counter += 1
                status = ""

            if verbose and (epoch % 5 == 0 or epoch == 1):
                metric_str = ""
                if "metrics" in report:
                    m = report['metrics']
                    titer_r2 = m['component_r2'].get('comp_5', float('nan'))
                    r2_str = f"{m['global_r2']:.4f}" if not np.isnan(m['global_r2']) else "nan"
                    titer_str = f"{titer_r2:.4f}" if not np.isnan(titer_r2) else "nan"
                    metric_str = f" | R2: {r2_str} (Titer: {titer_str})"
                if "phase_metrics" in report:
                    metric_str += f" | F1: {report['phase_metrics']['phase_f1']:.4f}"

                if verbose and epoch % 20 == 0 and "metrics" in report:
                    comp_r2s = {k: v for k, v in report['metrics']['component_r2'].items()
                                if not np.isnan(v)}
                    if comp_r2s:
                        sorted_r2 = sorted(comp_r2s.items(), key=lambda x: x[1])
                        print(f"  -> Worst R2: {sorted_r2[0]} | Best R2: {sorted_r2[-1]}")

                print(f"Epoch {epoch:3d}: Train={train_loss:.6f} | Val={val_loss:.6f}{status}{metric_str} "
                      f"| Conc={comps['conc']:.4f} | IC={comps['ic']:.4f} | NonNeg={comps['pinn_non_neg']:.4f}")

            if patience is not None and patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        return best_val_loss


def load_synthetic_data(npz_path, real_dataset):
    """
    Load synthetic .npz and apply real dataset's normalization stats.
    Both datasets must be in the same normalized space so pre-trained
    weights transfer directly to fine-tuning without a scale mismatch.
    """
    data = np.load(npz_path, allow_pickle=True)
    trajectories = data['trajectories'].copy()        # (N, T, C)
    times        = data['times']                      # (N, T)
    ics          = data['ics'].copy()                 # (N, C)
    phases       = data['phases']                     # (N, T)

    # Apply real data's normalization (fit on 10 reactors) to synthetic
    trajectories = (trajectories - real_dataset.traj_min) / (real_dataset.traj_max - real_dataset.traj_min)
    trajectories = np.clip(trajectories, 0, 1)
    ics = (ics - real_dataset.ic_min) / (real_dataset.ic_max - real_dataset.ic_min)
    ics = np.clip(ics, 0, 1)

    return dFBADataset(trajectories, times, ics, parameters={}, normalize=False, phases=phases)


def load_data(data_path):
    """Flexible loader for CSV or NPZ data."""
    p = Path(data_path)
    if p.is_file() and p.suffix == '.csv':
        print(f"Loading real experimental data from {p}...")
        trajectories, time_points, ics, metadata = load_experimental_data(str(p))
        phases = metadata.get('phases', None)
        dataset = dFBADataset(trajectories, time_points, ics, parameters={}, normalize=True, phases=phases)
        return dataset

    elif p.is_dir():
        print(f"Loading production NPZ data from directory {p}...")
        npz_files = list(p.glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {p}")

        all_trajectories, all_times, all_ics = [], [], []
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            profiles = data['profiles']
            time = data['time'].flatten() if len(data['time'].shape) > 1 else data['time']
            all_trajectories.append(profiles)
            all_times.append(time)
            all_ics.append(profiles[0, :])

        max_t = max(len(t) for t in all_times)
        n_sims, n_comps = len(all_trajectories), all_trajectories[0].shape[1]
        trajectories_padded = np.zeros((n_sims, max_t, n_comps))
        times_padded = np.zeros((n_sims, max_t))
        for i, (traj, t) in enumerate(zip(all_trajectories, all_times)):
            trajectories_padded[i, :len(t), :] = traj
            times_padded[i, :len(t)] = t
            if len(t) < max_t:
                trajectories_padded[i, len(t):, :] = traj[-1, :]
                times_padded[i, len(t):] = t[-1]

        dataset = dFBADataset(trajectories_padded, times_padded, np.array(all_ics), parameters={}, normalize=True)
        return dataset
    else:
        raise ValueError(f"Unsupported data path: {data_path}. Provide a .csv file or a directory of .npz files.")


def main():
    print(f"\n{'='*70}")
    print("COSMIC-dFBA Unified Training")
    print(f"{'='*70}")

    script_dir = Path(__file__).parent
    DATA_PATH  = script_dir / "data_2.csv"
    SYNTH_PATH = script_dir / "synthetic_training.npz"

    LATENT_DIM = 64
    N_HEADS    = 4
    EPOCHS     = 200
    PATIENCE   = 40

    # Pre-training: 50 epochs on 20k synthetic samples with large batches
    # to saturate the T4 GPU on Colab. patience=None runs all epochs.
    PRETRAIN_EPOCHS = 50
    PRETRAIN_LR     = 5e-4

    # Fine-tuning: low LR to avoid catastrophic forgetting, but high enough
    # for the scheduler not to kill learning in the first 30 epochs.
    # scheduler_patience=15 delays LR halving vs the pre-training default of 5.
    FINETUNE_LR = 1e-4

    try:
        dataset = load_data(str(DATA_PATH))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=LATENT_DIM,
        n_heads=N_HEADS
    )

    # --- Phase 1: Pre-train on synthetic data (if available) ---
    if SYNTH_PATH.exists():
        print(f"\n{'='*70}")
        print("Phase 1: Pre-training on synthetic data")
        print(f"{'='*70}")
        # Apply real data normalization so both datasets share the same scale.
        synth_dataset    = load_synthetic_data(str(SYNTH_PATH), real_dataset=dataset)
        synth_train_size = int(0.9 * len(synth_dataset))
        synth_val_size   = len(synth_dataset) - synth_train_size
        synth_train, synth_val = random_split(synth_dataset, [synth_train_size, synth_val_size])
        synth_train_loader = DataLoader(synth_train, batch_size=256, shuffle=True,  collate_fn=dfba_collate_fn)
        synth_val_loader   = DataLoader(synth_val,   batch_size=256, shuffle=False, collate_fn=dfba_collate_fn)

        print(f"Device: {device} | Synthetic: Train={synth_train_size}, Val={synth_val_size}")
        pretrain_trainer = Trainer(model, device, learning_rate=PRETRAIN_LR, model_type='enhanced')
        # patience=None disables early stopping — run all PRETRAIN_EPOCHS
        pretrain_trainer.train(synth_train_loader, synth_val_loader,
                               epochs=PRETRAIN_EPOCHS, patience=None)
        print("Pre-training complete. Switching to real-data fine-tuning.")
    else:
        print(f"\nNo synthetic data at {SYNTH_PATH} — skipping pre-training.")
        print("Run: python generate_synthetic_training.py data_2.csv")

    # --- Phase 2: Leave-one-out fine-tuning ---
    # With only 10 reactors a random 70/30 split gives 3 val samples whose
    # R2 is dominated by which reactors happen to land there. LOO uses every
    # reactor as the held-out test once, averaging results across all 10 folds
    # for a stable metric that isn't luck-of-the-draw.
    print(f"\n{'='*70}")
    print("Phase 2: Leave-one-out fine-tuning on real data")
    print(f"{'='*70}")

    n_reactors  = len(dataset)
    loo_val_losses, loo_r2s, loo_titer_r2s, loo_f1s = [], [], [], []
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

    start_time = time.time()
    for fold in range(n_reactors):
        # Reset to pre-trained weights for each fold
        model.load_state_dict(pretrained_state)

        val_indices   = [fold]
        train_indices = [i for i in range(n_reactors) if i != fold]
        from torch.utils.data import Subset
        fold_train = Subset(dataset, train_indices)
        fold_val   = Subset(dataset, val_indices)

        fold_train_loader = DataLoader(fold_train, batch_size=4, shuffle=True,  collate_fn=dfba_collate_fn)
        fold_val_loader   = DataLoader(fold_val,   batch_size=1, shuffle=False, collate_fn=dfba_collate_fn)

        fold_trainer = Trainer(model, device, learning_rate=FINETUNE_LR, model_type='enhanced',
                               scheduler_patience=15)
        best_val = fold_trainer.train(fold_train_loader, fold_val_loader,
                                      epochs=EPOCHS, patience=PATIENCE, verbose=False)
        report = fold_trainer.validate(fold_val_loader)

        r2       = report['metrics']['global_r2']          if 'metrics'       in report else float('nan')
        titer_r2 = report['metrics']['component_r2'].get('comp_5', float('nan')) if 'metrics' in report else float('nan')
        f1       = report['phase_metrics']['phase_f1']     if 'phase_metrics' in report else float('nan')
        loo_val_losses.append(best_val)
        loo_r2s.append(r2)
        loo_titer_r2s.append(titer_r2)
        loo_f1s.append(f1)
        print(f"  Fold {fold+1:2d}/10 (val=reactor {fold}): "
              f"loss={best_val:.4f} | R2={r2:.4f} | TiterR2={titer_r2:.4f} | F1={f1:.4f}")

    elapsed = time.time() - start_time
    print(f"\n✓ LOO complete in {elapsed:.1f}s")
    print(f"  Mean val loss : {np.mean(loo_val_losses):.4f} ± {np.std(loo_val_losses):.4f}")
    print(f"  Mean R2       : {np.mean(loo_r2s):.4f} ± {np.std(loo_r2s):.4f}")
    print(f"  Mean Titer R2 : {np.mean(loo_titer_r2s):.4f} ± {np.std(loo_titer_r2s):.4f}")
    print(f"  Mean F1       : {np.mean(loo_f1s):.4f} ± {np.std(loo_f1s):.4f}")

    # Save final model (re-trained on all 10 reactors)
    model.load_state_dict(pretrained_state)
    all_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dfba_collate_fn)
    final_trainer = Trainer(model, device, learning_rate=FINETUNE_LR, model_type='enhanced',
                            scheduler_patience=15)
    final_trainer.train(all_loader, all_loader, epochs=EPOCHS // 2, patience=PATIENCE)
    torch.save({
        'model_state': model.state_dict(),
        'model_type': 'enhanced',
        'hyperparams': {'latent_dim': LATENT_DIM, 'n_heads': N_HEADS, 'n_components': dataset.n_components},
        'loo_mean_r2': float(np.mean(loo_r2s)),
        'loo_mean_titer_r2': float(np.mean(loo_titer_r2s)),
    }, 'improved_model.pt')
    print(f"✓ Model saved: improved_model.pt")

if __name__ == "__main__":
    main()
