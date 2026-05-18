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
    def __init__(self, model, device, learning_rate=5e-4, model_type='enhanced'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
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

    def train(self, train_loader, val_loader, epochs=100, patience=20):
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

            if epoch % 5 == 0 or epoch == 1:
                metric_str = ""
                if "metrics" in report:
                    m = report['metrics']
                    # Show global R2 and the R2 of the "Titer" (component 7)
                    titer_r2 = m['component_r2'].get('comp_7', 'N/A')
                    metric_str = f" | R2: {m['global_r2']:.4f} (Titer: {titer_r2:.4f} if exists)"
                if "phase_metrics" in report:
                    metric_str += f" | F1: {report['phase_metrics']['phase_f1']:.4f}"

                # Print a small summary of all component R2s every 20 epochs
                if epoch % 20 == 0 and "metrics" in report:
                    comp_r2s = report['metrics']['component_r2']
                    sorted_r2 = sorted(comp_r2s.items(), key=lambda x: x[1])
                    print(f"  -> Worst R2: {sorted_r2[0]} | Best R2: {sorted_r2[-1]}")

                print(f"Epoch {epoch:3d}: Train={train_loss:.6f} | Val={val_loss:.6f}{status}{metric_str} "
                      f"| IC={comps['ic']:.4f} | NonNeg={comps['pinn_non_neg']:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        return best_val_loss


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
    DATA_PATH = script_dir / "data_2.csv"

    LATENT_DIM = 64
    N_HEADS = 4
    LR = 1e-4
    EPOCHS = 200
    PATIENCE = 40

    try:
        dataset = load_data(str(DATA_PATH))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=dfba_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=dfba_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Samples: Train={len(train_dataset)}, Val={len(val_dataset)}")

    model = CosmicNNSurrogateEnhanced(
        n_components=dataset.n_components,
        n_params=0,
        latent_dim=LATENT_DIM,
        n_heads=N_HEADS
    )

    trainer = Trainer(model, device, learning_rate=LR, model_type='enhanced')

    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)
    elapsed = time.time() - start_time

    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")

    torch.save({
        'model_state': model.state_dict(),
        'model_type': 'enhanced',
        'hyperparams': {'latent_dim': LATENT_DIM, 'n_heads': N_HEADS, 'n_components': dataset.n_components},
        'best_val_loss': best_val_loss,
    }, 'improved_model.pt')
    print(f"✓ Model saved: improved_model.pt")

if __name__ == "__main__":
    main()
