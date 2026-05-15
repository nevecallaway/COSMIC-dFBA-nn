import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

class ModelDiagnostics:
    """
    Advanced diagnostic suite for COSMIC-dFBA surrogate models.
    Focuses on statistical robustness, modality dominance, and biological RCA.
    """

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes a robust set of regression metrics to detect if the model is
        dominated by a single modality or failing on specific components.
        """
        # Flatten for global metrics, then calculate per-component
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        mape = mean_absolute_percentage_error(y_true.flatten(), y_pred.flatten())

        # Per-component R2 to detect "modality dominance" (e.g. biomass is good, product is bad)
        comp_r2 = {}
        for i in range(y_true.shape[-1]):
            comp_r2[f"comp_{i}"] = r2_score(y_true[..., i], y_pred[..., i])

        return {
            "global_r2": r2,
            "global_mape": mape,
            "component_r2": comp_r2
        }

    @staticmethod
    def calculate_phase_metrics(y_true_phase: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the phase switch classifier using F1 and Confusion Matrix.
        """
        # Convert logits to binary prediction (0: growth, 1: production)
        y_pred_phase = np.argmax(y_pred_logits, axis=-1)

        # True phase: Since dFBA uses a sigmoid, we treat 0.5 as the threshold
        y_true_binary = (y_true_phase > 0.5).astype(int)

        f1 = f1_score(y_true_binary.flatten(), y_pred_phase.flatten())
        cm = confusion_matrix(y_true_binary.flatten(), y_pred_phase.flatten())

        return {
            "phase_f1": f1,
            "confusion_matrix": cm
        }

    @staticmethod
    def analyze_modality_dominance(model, ic, time_points, params, device='cpu'):
        """
        Detects if the model is dominated by one input modality using Saliency Maps.
        Calculates the gradient of the final titer with respect to each input.
        """
        model.eval()
        ic = torch.FloatTensor(ic).to(device).requires_grad_(True)
        params = torch.FloatTensor(params).to(device).requires_grad_(True)
        time_tensor = torch.FloatTensor(time_points).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(ic, time_tensor, params)

        # Target the final titer (assuming last component is product)
        final_titer = outputs['concentrations'][0, -1, -1]

        # Backward pass to get gradients
        final_titer.backward()

        # Saliency = absolute value of gradients
        ic_saliency = ic.grad.abs().mean().item()
        param_saliency = params.grad.abs().mean().item()

        return {
            "ic_importance": ic_saliency,
            "param_importance": param_saliency,
            "dominance_ratio": ic_saliency / (param_saliency + 1e-6)
        }

    @staticmethod
    def detect_drop_off_rca(concentrations: np.ndarray, time_points: np.ndarray, comp_idx: int = -1):
        """
        Root Cause Analysis: Detects when the titer starts to drop off
        and calculates the rate of decay.
        """
        c = concentrations[..., comp_idx] # (batch, time)
        t = time_points

        # First derivative (rate of change)
        dc_dt = np.diff(c, axis=-1) / np.diff(t)

        # Find the point where dc_dt becomes significantly negative
        drop_off_mask = dc_dt < -0.05 # Threshold for "significant drop"

        results = []
        for i in range(c.shape[0]):
            # Find the first time point where the drop starts
            drops = np.where(drop_off_mask[i])[0]
            if len(drops) > 0:
                first_drop_t = t[drops[0]]
                decay_rate = np.mean(dc_dt[i, drops])
                results.append({
                    "drop_start_time": first_drop_t,
                    "avg_decay_rate": decay_rate,
                    "is_crashing": True
                })
            else:
                results.append({"is_crashing": False})

        return results
