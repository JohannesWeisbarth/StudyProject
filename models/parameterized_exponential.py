"""The parameterized exponential model is an exponential whose temperature plateau and time constant are
parameterized functions of the stress level and the test setup. This allows for interpolation of missing curves.

The temperature rise over the time t in s at the stress level sigma, a fraction of UTS, in the setup s is
    theta(t) = delta_T_stab(sigma, setup) * (1 - exp(-t / tau(sigma, setup)))
"""
from __future__ import annotations

import torch

from curve_sampling import CurveBatch
from models.simple_exponential import exponential

# The fibre materials as numbers for the network.
FIBRE_MATERIALS = {"carbon": 0.0, "glass": 1.0}


def setup_features(setup: dict) -> torch.Tensor:
    """Maps the setup of a paper to a tensor of shape (3,), which can then be given to the network.
    The following is given to the model:
    - the fibre material, 0 for carbon and 1 for glass
    - the fiber contribution. This depends on the "angle of attack" of the forces on the material.
      E.g. 1 for 0°, 0.25 for ±45°, 0 for 90°
    - the volume to surface ratio in mm
    """
    angles = torch.tensor(setup["angles_of_attack_deg"], dtype=torch.float64).deg2rad()
    return torch.tensor([FIBRE_MATERIALS[setup["fibre_material"]], float((angles.cos() ** 4).mean()),
                         setup["volume_to_surface_mm"]], dtype=torch.float64)


class ParameterizedExponential(torch.nn.Module):
    """Parameterized exponential function."""

    def __init__(
            self,
            initial_log_delta_T_stab: float,
            initial_log_tau: float,
            n_setup_features: int = 3,
            hidden_units: int = 16
        ) -> None:
        super().__init__()
        self.hidden_layer = torch.nn.Linear(1 + n_setup_features, hidden_units, dtype=torch.float64)
        self.output_layer = torch.nn.Linear(hidden_units, 2, dtype=torch.float64)
        # The network starts close to the given log delta_T_stab and log tau for every input.
        with torch.no_grad():
            self.output_layer.bias.copy_(torch.tensor([initial_log_delta_T_stab, initial_log_tau], dtype=torch.float64))

    def forward(self, batch: CurveBatch) -> torch.Tensor:
        """The temperature rise in K of every curve of the batch, of the same shape as batch["x"]."""
        features = torch.stack([setup_features(setup) for setup in batch["setup"]])
        inputs = torch.cat([batch["stress_level"].double()[:, None], features], dim=1)
        log_delta_T_stab, log_tau = self.output_layer(torch.tanh(self.hidden_layer(inputs))).unbind(-1)
        return exponential(batch["x"], log_delta_T_stab.exp()[:, None], log_tau.exp()[:, None])
