#!/usr/bin/env python3
"""Fit one exponential to every temperature curve.

The temperature rise over the cycles N is
    theta(N) = delta_t * (1 - exp(-heating_rate * N / delta_t))
It has two parameters: the heating rate in °C per cycle, the slope at the start, and the temperature rise plateau delta_t in
°C that the curve approaches. Both constraints hold for every heating rate and delta_t: theta(0) = 0 and
theta(infinity) = delta_t. The time constant is phi = delta_t / heating_rate.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

# Plots are never rendered
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from curve_dataset import CurveDataset

logger = logging.getLogger(__name__)

# All results of the fits are written to this directory, in the directory the script is started in, one folder per paper.
OUTPUT_DIRECTORY = Path("results") / "exponential_fits"
# The papers that are fitted: vector figures with exact points.
PAPERS = ("Huang2019_IJF", "Huang2020_EFM")
# The heating rate of the data is the slope of a straight line through the points of the first cycles.
INITIAL_CYCLES = 200
# The optimizer of the fit: Adam on the logarithms of the parameters, full batch (one curve has only a few hundred points).
LEARNING_RATE = 0.05
TRAINING_STEPS = 3000


def exponential(cycles: torch.Tensor, heating_rate: float | torch.Tensor, delta_t: float | torch.Tensor) -> torch.Tensor:
    """The model theta(N) = delta_t * (1 - exp(-heating_rate * N / delta_t)). Its slope at N = 0 is the heating rate in
    °C per cycle and it approaches the plateau delta_t. It is the form with phi = delta_t / heating_rate."""
    return delta_t * (1.0 - torch.exp(-heating_rate * cycles.double() / delta_t))


class ExponentialModel(torch.nn.Module):
    """The exponential with the learnable heating rate and plateau delta_t. Both are stored as logarithms, so they stay
    positive and the gradient steps are relative changes, which suits parameters that span several orders of magnitude."""

    def __init__(self, cycles: torch.Tensor, temperature_rise: torch.Tensor) -> None:
        super().__init__()
        # Initialize from the cycles and temperature data.
        # - delta_t is the plateau the curve approaches, so a good guess is the highest temperature observed in the data.
        # - The heating_rate is guessed from the initial slope.
        cycles, temperature_rise = cycles.double(), temperature_rise.double()
        self.log_delta_t = torch.nn.Parameter(torch.log(temperature_rise.max()))
        first = torch.where(cycles > 0, cycles, torch.inf).argmin()
        initial_slope = (temperature_rise[first] / cycles[first]).clamp_min(1e-9)
        self.log_heating_rate = torch.nn.Parameter(torch.log(initial_slope))

    @property
    def delta_t(self) -> torch.Tensor:
        return self.log_delta_t.exp()

    @property
    def heating_rate(self) -> torch.Tensor:
        return self.log_heating_rate.exp()

    def forward(self, cycles: torch.Tensor) -> torch.Tensor:
        return exponential(cycles, self.heating_rate, self.delta_t)


def fit_exponential(cycles: torch.Tensor, temperature_rise: torch.Tensor) -> tuple[float, float]:
    """Fit the exponential to a curve by gradient descent on the MSE loss. Returns the heating rate in °C per cycle and the
    temperature rise plateau delta_t in °C."""
    model = ExponentialModel(cycles, temperature_rise)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for step in range(TRAINING_STEPS):
        optimizer.zero_grad()
        prediction = model(cycles)
        loss = F.mse_loss(input=prediction, target=temperature_rise.double())
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            logger.debug("step %d: MSE %.4g, heating rate %.4g, delta_t %.4g",
                         step, loss.item(), model.heating_rate.item(), model.delta_t.item())
    return model.heating_rate.item(), model.delta_t.item()


def plot_fits(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """Plot one panel of extracted data and corresponding exponential fit for every stress level."""
    columns = 4
    rows = math.ceil(len(fitted_curves) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.4 * rows), sharex=True, squeeze=False)
    for axis, (curve, heating_rate, delta_t) in zip(axes.flat, fitted_curves):
        axis.scatter(curve["x"], curve["y"], s=3, linewidths=0, color="tab:blue", label="extracted")
        cycles = torch.linspace(0.0, float(curve["x"].max()), 300)
        axis.plot(cycles, exponential(cycles, heating_rate, delta_t), lw=1.5, color="tab:red", label="fit")
        axis.axhline(delta_t, lw=0.6, ls=":", color="tab:red")
        axis.set_title(f"{float(curve['stress_level']):g} % UTS: rate {1000 * heating_rate:.3g} °C/kcycle, ΔT_stab {delta_t:.3g} °C",
                       fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(fitted_curves):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Cycles", fontsize=8)
    figure.suptitle("Exponential Fit", fontsize=10)
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_parameters(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """Plot the fitted parameters over the stress level.."""
    stress_levels = [float(curve["stress_level"]) for curve, _, _ in fitted_curves]
    figure, (delta_t_axis, rate_axis) = plt.subplots(1, 2, figsize=(10, 4))
    delta_t_axis.plot(stress_levels, [delta_t for _, _, delta_t in fitted_curves], "o-", color="tab:red")
    delta_t_axis.set_ylabel("ΔT_stab [°C]")
    fitted_rates = [1000.0 * heating_rate for _, heating_rate, _ in fitted_curves]
    data_rates = []
    for curve, _, _ in fitted_curves:
        first_cycles = curve["x"] <= INITIAL_CYCLES
        data_rates.append(1000.0 * float(np.polyfit(curve["x"][first_cycles].numpy(), curve["y"][first_cycles].numpy(), 1)[0]))
    rate_axis.plot(stress_levels, fitted_rates, "o-", color="tab:green", label="fit")
    rate_axis.plot(stress_levels, data_rates, "s--", color="tab:gray", label=f"data: first {INITIAL_CYCLES} cycles")
    rate_axis.set_ylabel("Heating rate [°C per 1000 cycles]")
    rate_axis.legend(fontsize=8)
    for axis in (delta_t_axis, rate_axis):
        axis.set_xlabel("Stress level [% UTS]")
        axis.grid(True, alpha=0.3)
    # figure.suptitle(f"{paper}: parameters of θ = ΔT_stab * (1 - exp(-rate * N / ΔT_stab))")
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_fits_compact(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """All curves in one plot, one color for each stress level. The temperature rise is on a logarithmic axis."""
    stress_levels = [float(curve["stress_level"]) for curve, _, _ in fitted_curves]
    colors = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(min(stress_levels), max(stress_levels))
    figure, axis = plt.subplots(figsize=(9, 6))
    for curve, heating_rate, delta_t in fitted_curves:
        color = colors(norm(float(curve["stress_level"])))
        positive = curve["y"] > 0
        axis.scatter(curve["x"][positive], curve["y"][positive], s=2, linewidths=0, color=color)
        cycles = torch.linspace(5.0, float(curve["x"].max()), 300)
        axis.plot(cycles, exponential(cycles, heating_rate, delta_t), lw=1.5, color=color)
    axis.set_yscale("log")
    axis.set_xlabel("Cycles")
    axis.set_ylabel("Temperature rise [°C]")
    axis.grid(True, alpha=0.3)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colors), ax=axis, label="Stress level [% UTS]")
    figure.suptitle(f"{paper}: exponential fit of all curves")
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def main() -> None:
    dataset = CurveDataset()
    for paper in PAPERS:
        logger.info("%s\n stress level    rate [°C/kcycle]    delta_t [°C]    RMSE [°C]    RMSE / delta_t", paper)
        fitted_curves = []
        for curve in sorted((curve for curve in dataset.curves if curve["paper"] == paper),
                            key=lambda curve: float(curve["stress_level"])):
            heating_rate, delta_t = fit_exponential(curve["x"], curve["y"])
            # PyTorch has no RMSE loss, it is the square root of the MSE.
            rmse = F.mse_loss(exponential(curve["x"], heating_rate, delta_t), curve["y"].double()).sqrt().item()
            logger.info(" %9.1f %% %17.4g %15.3f %12.3f %14.2f",
                        float(curve["stress_level"]), 1000.0 * heating_rate, delta_t, rmse, rmse / delta_t)
            fitted_curves.append((curve, heating_rate, delta_t))
        for plot, file_name in ((plot_fits, "fits.png"), (plot_parameters, "parameters.png"),
                                (plot_fits_compact, "fits_compact.png")):
            plot(paper, fitted_curves, OUTPUT_DIRECTORY / paper / file_name)
            logger.info("Wrote %s.", OUTPUT_DIRECTORY / paper / file_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
