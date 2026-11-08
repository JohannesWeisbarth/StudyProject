"""An exponential function fitted to each temperature curve in a plot separately.

The temperature rise over the time t in s is modelled as
    temperature(t) = delta_T_stab * (1 - exp(-t / tau))
It thus has two fitted parameters:
- the temperature rise plateau delta_T_stab in K
- the time constant tau in s.

For the fitted delta_T_stab and tau, it holds that temperature(0) = 0 and
temperature(infinity) = delta_T_stab. Please note that this is not physically
plausible. In the real world, the temperature does not stabilize forever, instead
you observe a second exponential before failure.
"""
from __future__ import annotations

import torch


def exponential(time: torch.Tensor, delta_T_stab: float | torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """The model theta(t) = delta_T_stab * (1 - exp(-t / tau)). It approaches the plateau delta_T_stab with the time constant tau."""
    return delta_T_stab * (1.0 - torch.exp(-time.double() / tau))


class SimpleExponential(torch.nn.Module):
    """The exponential with the learnable plateau delta_T_stab and time constant tau, stored as logarithms to keep them
    positive."""

    def __init__(self, time: torch.Tensor, temperature_rise: torch.Tensor) -> None:
        super().__init__()
        # Initialize from the time and temperature data.
        # - delta_T_stab is the plateau the curve approaches, so a good guess is the highest temperature observed in the data.
        # - tau is guessed as the first time at which the curve reaches 1 - 1/e = 63 % of that plateau: an exponential that
        #   starts at 0 reaches theta(tau) = delta_T_stab * (1 - 1/e) after one time constant.
        #   A guess from the slope at the first point fails: a first point at or below 0 gives an almost infinite tau, and the
        #   fit then gets stuck at a straight line.
        time, temperature_rise = time.double(), temperature_rise.double()
        initial_delta_T_stab = temperature_rise.max()
        reached = temperature_rise >= (1.0 - torch.exp(torch.tensor(-1.0, dtype=torch.float64))) * initial_delta_T_stab
        initial_tau = time[reached.nonzero()[0, 0]].clamp_min(time[time > 0].min())
        self.log_delta_T_stab = torch.nn.Parameter(torch.log(initial_delta_T_stab))
        self.log_tau = torch.nn.Parameter(torch.log(initial_tau))

    @property
    def delta_T_stab(self) -> torch.Tensor:
        return self.log_delta_T_stab.exp()

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def heating_rate(self) -> torch.Tensor:
        """The slope at the start in K per s."""
        return self.delta_T_stab / self.tau

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return exponential(time, self.delta_T_stab, self.tau)
