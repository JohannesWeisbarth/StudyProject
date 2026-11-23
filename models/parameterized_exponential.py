"""The parameterized exponential model is an exponential whose temperature plateau and time constant are
parameterized functions of the stress level and the test setup. This allows for interpolation of missing curves.

The temperature rise over the time t in s at the stress level sigma, a fraction of UTS, in the setup s is
    theta(t) = delta_T_stab(sigma, setup) * (1 - exp(-t / tau(sigma, setup)))
"""
from __future__ import annotations

import torch

from curve_dataset import Curve
from curve_sampling import CurveBatch
from models.simple_exponential import SimpleExponential, exponential

# The fibre materials as numbers for the network.
FIBRE_MATERIALS = {"carbon": 0.0, "glass": 1.0}


def setup_features(setup: dict) -> torch.Tensor:
    """Maps the setup of a paper to a tensor of shape (4,), which can then be given to the network.
    The following is given to the model:
    - the fibre material, 0 for carbon and 1 for glass
    - the fiber contribution. This depends on the "angle of attack" of the forces on the material.
      E.g. 1 for 0°, 0.25 for ±45°, 0 for 90°
    - the volume to surface ratio in mm
    - the logarithm of the loading frequency in Hz. The heat dissipated per second grows with the frequency, so
      log delta_T_stab, which the network predicts, is expected to grow linearly with log f.
    """
    angles = torch.tensor(setup["angles_of_attack_deg"], dtype=torch.float64).deg2rad()
    return torch.tensor([FIBRE_MATERIALS[setup["fibre_material"]], float((angles.cos() ** 4).mean()),
                         setup["volume_to_surface_mm"], float(torch.log(torch.tensor(setup["frequency_hz"])))],
                        dtype=torch.float64)


class ParameterizedExponential(torch.nn.Module):
    """Parameterized exponential function."""
    stress_mean: torch.Tensor
    stress_std: torch.Tensor

    def __init__(self, train_curves: list[Curve], n_setup_features: int = 4, hidden_units: int = 16) -> None:
        super().__init__()
        # The stress level is standardized with the mean and standard deviation of the training curves. Its raw range of
        # about 0.3 to 0.9 barely changes the hidden units at their initial weights, so the network starts almost linear in it.
        stress_levels = torch.stack([curve["stress_level"] for curve in train_curves]).double()
        self.register_buffer("stress_mean", stress_levels.mean())
        self.register_buffer("stress_std", stress_levels.std())
        self.hidden_layer = torch.nn.Linear(1 + n_setup_features, hidden_units, dtype=torch.float64)
        self.output_layer = torch.nn.Linear(hidden_units, 2, dtype=torch.float64)
        # The network starts at the mean of the starting values of the single-curve fits, for every input.
        guesses = [SimpleExponential(curve["x"], curve["y"]) for curve in train_curves]
        with torch.no_grad():
            self.output_layer.bias.copy_(torch.stack([torch.stack([guess.log_delta_T_stab for guess in guesses]).mean(),
                                                      torch.stack([guess.log_tau for guess in guesses]).mean()]))

    def predict_parameters(self, batch: CurveBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts delta_T_stab in K and tau in s for every curve of the batch, each a tensor of shape (n_curves,)."""
        features = torch.stack([setup_features(setup) for setup in batch["setup"]])
        stress = (batch["stress_level"].double() - self.stress_mean) / self.stress_std
        inputs = torch.cat([stress[:, None], features], dim=1)
        log_delta_T_stab, log_tau = self.output_layer(torch.tanh(self.hidden_layer(inputs))).unbind(-1)
        return log_delta_T_stab.exp(), log_tau.exp()

    def forward(self, batch: CurveBatch) -> torch.Tensor:
        """Predicts the temperature rise in K for every curve of the batch."""
        delta_T_stab, tau = self.predict_parameters(batch)
        return exponential(batch["x"], delta_T_stab[:, None], tau[:, None])
