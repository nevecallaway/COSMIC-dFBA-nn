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

from model import CosmicNNSurrogateEnhanced, CosmicNNSurrogateLSTM, dFBADataset, dfba_collate_fn
from utils import load_experimental_data, ModelDiagnostics

IDX_TITER = 5   # column index of Titer in the 25-component trajectory

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
        phase_pred = predictions['phase_weights']   # (batch, time, 1) regression in [0,1]
        growth_rates = predictions['growth_rates']
        prod_rates = predictions['prod_rates']

        # 1. Weighted concentration MSE — titer gets 8× weight
        comp_weights = torch.ones(targets.shape[-1], device=targets.device)
        comp_weights[IDX_TITER] = 5.0
        conc_loss = (((conc_pred - targets) ** 2) * comp_weights).mean()

        # 1a. Endpoint titer loss — directly penalise final-timepoint titer error
        endpoint_loss = 2.0 * nn.functional.mse_loss(
            conc_pred[:, -1, IDX_TITER], targets[:, -1, IDX_TITER])

        # 1b. Peak-time loss — soft argmax to align predicted titer peak with actual.
        # Replaces the monotonicity penalty, which was fighting the real rise-then-fall
        # dynamics seen in the data.
        T = conc_pred.shape[1]
        t_idx = torch.arange(T, dtype=torch.float32, device=targets.device) / max(T - 1, 1)
        pred_peak_w = torch.softmax(conc_pred[:, :, IDX_TITER] * 5.0, dim=1)
        true_peak_w = torch.softmax(targets[:, :, IDX_TITER] * 5.0, dim=1)
        pred_peak_t = (pred_peak_w * t_idx).sum(dim=1)   # (batch,)
        true_peak_t = (true_peak_w * t_idx).sum(dim=1)   # (batch,)
        peak_time_loss = 3.0 * torch.mean((pred_peak_t - true_peak_t) ** 2)

        # 2. IC constraint
        ic_loss = 0.1 * torch.mean((conc_pred[:, 0, :] - ics) ** 2)

        # 3. Non-flatness penalty (Increased to punish flat lines)
        conc_variance = torch.var(conc_pred, dim=1)
        flatness_penalty = 0.2 * torch.mean(1.0 / (1.0 + conc_variance))

        # 4. PINN: Non-negativity penalty
        non_neg_loss = 0.5 * torch.mean(torch.clamp(conc_pred, max=0)**2)

        # 5. PINN: Concentration Smoothness
        conc_smoothness = 0.1 * torch.mean((conc_pred[:, 1:, :] - conc_pred[:, :-1, :]) ** 2)

        # 6. Phase regression — MSE against continuous 0-1 fraction, all timepoints.
        # Weight raised to 3.0: f(t) trajectory accuracy is the primary metric and
        # was under-penalised at 0.5 relative to the concentration losses.
        phase_loss = torch.tensor(0.0, device=targets.device)
        if phases_batch is not None:
            phase_target = phases_batch.unsqueeze(-1)   # (batch, time, 1)
            phase_loss = 3.0 * nn.functional.mse_loss(phase_pred, phase_target)

        # 7. PINN: Rate-based constraints
        blended_rates = (1 - phase_pred) * growth_rates + phase_pred * prod_rates
        rate_smoothness = 0.1 * torch.mean((blended_rates[:, 1:, :] - blended_rates[:, :-1, :]) ** 2)
        rate_magnitude = 0.01 * (torch.mean(torch.abs(growth_rates)) + torch.mean(torch.abs(prod_rates)))

        # 8. Phase smoothness (encourage gradual rather than jittery transitions)
        phase_smoothness = 0.05 * torch.mean((phase_pred[:, 1:, :] - phase_pred[:, :-1, :]) ** 2)

        total_loss = (conc_loss + endpoint_loss + peak_time_loss +
                      ic_loss + flatness_penalty + phase_loss +
                      non_neg_loss + conc_smoothness + rate_smoothness +
                      rate_magnitude + phase_smoothness)

        return total_loss, {
            'conc': conc_loss.item(),
            'titer_endpoint': endpoint_loss.item(),
            'titer_peak': peak_time_loss.item(),
            'ic': ic_loss.item(),
            'phase_mse': phase_loss.item() if isinstance(phase_loss, torch.Tensor) else phase_loss,
            'pinn_non_neg': non_neg_loss.item(),
            'pinn_rate_smooth': rate_smoothness.item(),
        }

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        components = {'conc': 0, 'titer_endpoint': 0, 'titer_peak': 0,
                      'ic': 0, 'phase_mse': 0, 'pinn_non_neg': 0, 'pinn_rate_smooth': 0}

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
        all_targets, all_preds = [], []
        all_phase_targets, all_phase_preds, all_times_days = [], [], []

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
                    # Convert normalised [0,1] time back to actual days for metric
                    t_norm  = batch['time'].cpu().numpy()           # (B, T) in [0,1]
                    t_scale = batch['time_scale'].cpu().numpy()     # (B, 1) max day
                    all_times_days.append(t_norm * t_scale)        # (B, T) in days

        val_loss /= len(val_loader)
        report = {"val_loss": val_loss}
        if all_targets:
            y_true = np.concatenate(all_targets, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            report["metrics"] = ModelDiagnostics.calculate_regression_metrics(y_true, y_pred)
            report["spearman"] = ModelDiagnostics.calculate_spearman_metrics(y_true, y_pred)
            report["y_true"] = y_true
            report["y_pred"] = y_pred
            if all_phase_targets:
                p_true = np.concatenate(all_phase_targets, axis=0)
                p_pred = np.concatenate(all_phase_preds, axis=0)
                t_days = np.concatenate(all_times_days, axis=0)   # (N, T) in days
                report["phase_metrics"]   = ModelDiagnostics.calculate_phase_metrics(p_true, p_pred)
                report["transition"]      = ModelDiagnostics.calculate_transition_metrics(p_true, p_pred, t_days)

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
                if "phase_metrics" in report:
                    f1 = report['phase_metrics']['phase_f1']
                    mcc = report['phase_metrics'].get('mcc', float('nan'))
                    mcc_str = f"{mcc:.4f}" if not np.isnan(mcc) else "n/a"
                    metric_str = f" | F1: {f1:.4f} | MCC: {mcc_str}"
                if "transition" in report:
                    t_mae = report['transition']['transition_mae_days']
                    if not np.isnan(t_mae):
                        metric_str += f" | TransMAE: {t_mae:.2f}d"
                print(f"Epoch {epoch:3d}{status}{metric_str}")

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
    If the .npz contains 'doe_params' (N, 3) these are passed through as
    process parameters so the pre-trained model learns to use them.
    """
    data = np.load(npz_path, allow_pickle=True)
    trajectories = data['trajectories'].copy()        # (N, T, C)
    times        = data['times']                      # (N, T)
    ics          = data['ics'].copy()                 # (N, C)
    phases       = data['phases']                     # (N, T)

    # Apply real data's two-step normalization to synthetic data so both
    # datasets share the same scale.
    traj_norm = (trajectories - real_dataset.traj_min) / (real_dataset.traj_max - real_dataset.traj_min)
    trajectories = (traj_norm - real_dataset.traj_scale_min) / (real_dataset.traj_scale_max - real_dataset.traj_scale_min)

    ic_norm = (ics - real_dataset.ic_min) / (real_dataset.ic_max - real_dataset.ic_min)
    ics = (ic_norm - real_dataset.ic_scale_min) / (real_dataset.ic_scale_max - real_dataset.ic_scale_min)

    # Build parameters dict from stored DoE params + specific rates
    parameters = {}
    if 'doe_params' in data:
        dp = data['doe_params']       # (N, 3)
        parameters.update({'O2': dp[:, 0], 'AAs': dp[:, 1], 'Glc': dp[:, 2]})
    if 'specific_rates' in data:
        sr = data['specific_rates']   # (N, 50)
        for k in range(sr.shape[1]):
            parameters[f'rate_{k}'] = sr[:, k]

    return dFBADataset(trajectories, times, ics, parameters=parameters,
                       normalize=False, phases=phases)


def load_data(data_path):
    """Flexible loader for CSV or NPZ data."""
    p = Path(data_path)
    if p.is_file() and p.suffix == '.csv':
        print(f"Loading real experimental data from {p}...")
        doe_file   = str(p.parent / 'data_1.csv')
        rates_file = str(p.parent / 'data_3.csv')
        fba_file   = str(p.parent / 'data_4.csv')
        trajectories, time_points, ics, metadata = load_experimental_data(
            str(p), doe_file=doe_file, rates_file=rates_file, fba_file=fba_file)
        phases   = metadata.get('phases', None)
        doe_arr  = metadata.get('doe_params', None)        # (n_reactors, 3) or None
        rate_arr = metadata.get('specific_rates', None)    # (n_reactors, 50) or None
        fba_arr  = metadata.get('fba_efficiencies', None)  # (n_reactors, 22) or None
        parameters = {}
        if doe_arr is not None:
            parameters.update({'O2': doe_arr[:, 0], 'AAs': doe_arr[:, 1], 'Glc': doe_arr[:, 2]})
        if rate_arr is not None:
            for k in range(rate_arr.shape[1]):
                parameters[f'rate_{k}'] = rate_arr[:, k]
        if fba_arr is not None:
            for k in range(fba_arr.shape[1]):
                parameters[f'fba_{k}'] = fba_arr[:, k]
        dataset = dFBADataset(trajectories, time_points, ics, parameters=parameters,
                              normalize=True, phases=phases)
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


def shuffle_dataset(dataset):
    """
    Permutation baseline: independently shuffle inputs (ICs + params) and
    outputs (trajectories + phases) so there is no real signal to learn.
    The model trained on this data establishes chance-level performance.
    Data is already normalised so normalize=False is passed through.
    """
    N = len(dataset)
    idx_in  = np.random.permutation(N)   # shuffled input order
    idx_out = np.random.permutation(N)   # shuffled output order (independent)

    new_trajs  = dataset.trajectories[idx_out]
    new_times  = dataset.time_points[idx_out]
    new_ics    = dataset.initial_conditions[idx_in]
    new_phases = dataset.phases[idx_out] if dataset.phases is not None else None

    new_params = {}
    for key, val in dataset.parameters.items():
        if val is not None:
            new_params[key] = np.array([val[i] for i in idx_in])

    shuffled = dFBADataset(new_trajs, new_times, new_ics,
                           parameters=new_params, normalize=False,
                           phases=new_phases)
    # Copy normalisation stats so scale is identical to the real dataset
    for attr in ('traj_min', 'traj_max', 'traj_scale_min', 'traj_scale_max',
                 'ic_min', 'ic_max', 'ic_scale_min', 'ic_scale_max'):
        setattr(shuffled, attr, getattr(dataset, attr))
    return shuffled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-synthetic', action='store_true',
                        help='Skip pre-training on synthetic data, train on real data only')
    parser.add_argument('--lstm', action='store_true',
                        help='Use LSTM decoder instead of transformer attention')
    parser.add_argument('--shuffle', action='store_true',
                        help='Permutation baseline: shuffle inputs vs outputs before '
                             'training to establish chance-level performance')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("COSMIC-dFBA Unified Training")
    print(f"{'='*70}")

    USE_SYNTHETIC = not args.no_synthetic
    USE_LSTM      = args.lstm
    USE_SHUFFLE   = args.shuffle

    script_dir = Path(__file__).parent
    DATA_PATH  = script_dir / "data" / "data_2.csv"
    SYNTH_PATH = script_dir / "synthetic_training.npz"

    LATENT_DIM = 64
    N_HEADS    = 4
    EPOCHS     = 400
    PATIENCE   = 80

    # Pre-training: 500 epochs on GPU — more synthetic data and compute budget
    # means pre-training can converge more fully before fine-tuning begins.
    PRETRAIN_EPOCHS = 500
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

    if USE_SHUFFLE:
        print("\n*** PERMUTATION BASELINE: inputs and outputs independently shuffled ***")
        print("*** Train on this to establish chance-level performance             ***\n")
        dataset = shuffle_dataset(dataset)

    OUTPUT_PATH = 'shuffled_model.pt' if USE_SHUFFLE else 'improved_model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_params = dataset.n_params if hasattr(dataset, 'n_params') else 0
    N_LAYERS = 2  # used for both LSTM (actual layers) and transformer (stored for compat)
    if USE_LSTM:
        print(f"Model: LSTM  n_components={dataset.n_components}, n_params={n_params}, n_layers={N_LAYERS}")
        model = CosmicNNSurrogateLSTM(
            n_components=dataset.n_components,
            n_params=n_params,
            latent_dim=LATENT_DIM,
            n_layers=N_LAYERS,
        )
    else:
        print(f"Model: Transformer  n_components={dataset.n_components}, n_params={n_params}, n_heads={N_HEADS}")
        model = CosmicNNSurrogateEnhanced(
            n_components=dataset.n_components,
            n_params=n_params,
            latent_dim=LATENT_DIM,
            n_heads=N_HEADS,
        )

    # --- Phase 1: Pre-train on synthetic data (if available and enabled) ---
    if USE_SYNTHETIC and SYNTH_PATH.exists():
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
    loo_val_losses, loo_f1s = [], []
    loo_trans_mae  = []   # transition time MAE per fold (days)
    loo_mcc        = []
    loo_spec       = []   # specificity per fold  (paper: 0.780)
    loo_sens       = []   # sensitivity per fold  (paper: 0.681)
    loo_within10   = []   # fraction of reactors with final titer within 10%
    # Calibration data for conformal prediction: list of (y_true, y_pred, sigma) per fold
    conformal_cal = []
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

        f1        = report['phase_metrics']['phase_f1']               if 'phase_metrics' in report else float('nan')
        mcc       = report['phase_metrics'].get('mcc', float('nan'))  if 'phase_metrics' in report else float('nan')
        spec      = report['phase_metrics'].get('specificity', float('nan')) if 'phase_metrics' in report else float('nan')
        sens      = report['phase_metrics'].get('sensitivity', float('nan')) if 'phase_metrics' in report else float('nan')
        trans_mae = report['transition']['transition_mae_days']        if 'transition'    in report else float('nan')

        # Compute within-10% final titer in original (denormalized) units
        within10 = float('nan')
        if 'y_true' in report and 'y_pred' in report:
            y_t = report['y_true'][:, -1, IDX_TITER]
            y_p = report['y_pred'][:, -1, IDX_TITER]
            def _denorm(x):
                x = x * (dataset.traj_scale_max[IDX_TITER] - dataset.traj_scale_min[IDX_TITER]) + dataset.traj_scale_min[IDX_TITER]
                return x * (dataset.traj_max[IDX_TITER] - dataset.traj_min[IDX_TITER]) + dataset.traj_min[IDX_TITER]
            within10 = float(np.mean(np.abs(_denorm(y_p) - _denorm(y_t)) / (np.abs(_denorm(y_t)) + 1e-8) <= 0.10))

        loo_val_losses.append(best_val)
        loo_f1s.append(f1)
        loo_mcc.append(mcc)
        loo_spec.append(spec)
        loo_sens.append(sens)
        loo_trans_mae.append(trans_mae)
        loo_within10.append(within10)
        if 'y_true' in report and 'y_pred' in report:
            conformal_cal.append({
                'fold': fold,
                'y_true': report['y_true'],
                'y_pred': report['y_pred'],
                'sigma':  report.get('sigma'),
            })
        trans_str    = f"{trans_mae:.2f}d"        if not np.isnan(trans_mae) else "n/a"
        mcc_str      = f"{mcc:.3f}"               if not np.isnan(mcc)       else "n/a"
        spec_str     = f"{spec:.3f}"              if not np.isnan(spec)      else "n/a"
        sens_str     = f"{sens:.3f}"              if not np.isnan(sens)      else "n/a"
        within10_str = f"{within10*100:.0f}%"     if not np.isnan(within10)  else "n/a"
        print(f"  Fold {fold+1:2d}/10 (val=reactor {fold}): "
              f"TransMAE={trans_str} | MCC={mcc_str} | F1={f1:.4f} | "
              f"Spec={spec_str} | Sens={sens_str} | Within10%={within10_str}")

    elapsed = time.time() - start_time
    valid_within10 = [x for x in loo_within10 if not np.isnan(x)]
    within10_count = sum(1 for x in valid_within10 if x > 0.5)
    print(f"\n✓ LOO complete in {elapsed:.1f}s")
    print(f"  Mean Transition MAE : {np.nanmean(loo_trans_mae):.2f} ± {np.nanstd(loo_trans_mae):.2f} days  ← primary metric")
    print(f"  Mean MCC            : {np.nanmean(loo_mcc):.4f} ± {np.nanstd(loo_mcc):.4f}   (paper: 0.454)")
    print(f"  Mean F1             : {np.nanmean(loo_f1s):.4f} ± {np.nanstd(loo_f1s):.4f}   (paper: 0.731)")
    print(f"  Mean Specificity    : {np.nanmean(loo_spec):.4f} ± {np.nanstd(loo_spec):.4f}   (paper: 0.780)")
    print(f"  Mean Sensitivity    : {np.nanmean(loo_sens):.4f} ± {np.nanstd(loo_sens):.4f}   (paper: 0.681)")
    print(f"  Titer Within 10%    : {within10_count}/{len(valid_within10)} reactors")

    # Save final model (re-trained on all 10 reactors)
    model.load_state_dict(pretrained_state)
    all_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dfba_collate_fn)
    final_trainer = Trainer(model, device, learning_rate=FINETUNE_LR, model_type='enhanced',
                            scheduler_patience=15)
    final_trainer.train(all_loader, all_loader, epochs=EPOCHS, patience=PATIENCE)
    torch.save({
        'model_state': model.state_dict(),
        'model_type': 'enhanced',
        'hyperparams': {
            'arch': 'lstm' if USE_LSTM else 'transformer',
            'latent_dim': LATENT_DIM,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS if USE_LSTM else 2,
            'n_components': dataset.n_components,
            'n_params': dataset.n_params,
        },
        'loo_mean_trans_mae':  float(np.nanmean(loo_trans_mae)),
        'loo_mean_mcc':        float(np.nanmean(loo_mcc)),
        'loo_mean_f1':         float(np.nanmean(loo_f1s)),
        'loo_mean_spec':       float(np.nanmean(loo_spec)),
        'loo_mean_sens':       float(np.nanmean(loo_sens)),
        'loo_titer_within10':  within10_count,
        'conformal_cal': conformal_cal,
    }, OUTPUT_PATH)
    print(f"✓ Model saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
