import torch
import numpy as np
from cosmic_nn_surrogate import DifferentiableIntegrator

def test_integrator():
    integrator = DifferentiableIntegrator()

    # Setup: Constant flux v=1.0, C0=0, 11 timepoints from 0 to 1
    batch_size = 1
    n_timepoints = 11
    n_components = 1

    initial_conditions = torch.zeros(batch_size, n_components) # (1, 1)
    blended_rates = torch.ones(batch_size, n_timepoints, n_components) # (1, 11, 1)
    time_points = torch.linspace(0, 1, n_timepoints).unsqueeze(0) # (1, 11)

    concentrations = integrator(initial_conditions, blended_rates, time_points)

    print("Input time points:\n", time_points)
    print("Predicted concentrations:\n", concentrations)

    # Expected: C(t) = 0 + sum(1.0 * dt)
    # At t=1, C should be 1.0
    expected_final = 1.0
    actual_final = concentrations[0, -1, 0].item()

    print(f"Expected Final: {expected_final}, Actual Final: {actual_final}")

    if np.isclose(actual_final, expected_final):
        print("✅ Integration test passed!")
    else:
        print("❌ Integration test failed!")

if __name__ == "__main__":
    test_integrator()
