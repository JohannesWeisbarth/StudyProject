#!/usr/bin/env python3
"""Fit one exponential to every temperature curve.

The temperature rise over the cycles N is
    theta(N) = delta_t * (1 - exp(-N / tau))
It has two parameters: the time constant tau in cycles, which controls the rise, and the temperature rise delta_t in °C that
the curve approaches. Both constraints hold for every tau and delta_t: theta(0) = 0 and theta(infinity) = delta_t.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

# Plots are never rendered
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curve_dataset import CurveDataset

# All results of the fits are written to this directory, in the directory the script is started in.
OUTPUT_DIRECTORY = Path("results") / "exponential_fit"
# The papers that are fitted: vector figures with exact points.
PAPERS = ("Huang2019_IJF", "Huang2020_EFM")
# The heating rate of the data is the slope of a straight line through the points of the first cycles.
INITIAL_CYCLES = 200
# The time constants in cycles that are tried, spaced by 1 %, from a rise within a few cycles to one that is not reached in a test.
TIME_CONSTANTS = torch.logspace(math.log10(5), math.log10(50000), 1000, dtype=torch.float64)


def fit_exponential(cycles: torch.Tensor, temperature_rise: torch.Tensor) -> tuple[float, float]:
    """Fit theta(N) = delta_t * (1 - exp(-N / tau)) to a curve by least squares. Return the time constant tau in cycles and
    the temperature rise delta_t in °C. For a given tau the best delta_t follows from a linear least squares, so the fit
    tries all time constants and takes the one with the smallest error."""
    cycles, temperature_rise = cycles.double(), temperature_rise.double()
    rise = 1.0 - torch.exp(-cycles[:, None] / TIME_CONSTANTS[None, :])                # (points, time constants)
    delta_t = (rise * temperature_rise[:, None]).sum(0) / (rise * rise).sum(0)        # best delta_t for each tau
    squared_error = ((delta_t[None, :] * rise - temperature_rise[:, None]) ** 2).sum(0)
    best = int(squared_error.argmin())
    return TIME_CONSTANTS[best].item(), delta_t[best].item()


def exponential(cycles: torch.Tensor, time_constant: float, delta_t: float) -> torch.Tensor:
    """The model theta(N) = delta_t * (1 - exp(-N / tau))."""
    return delta_t * (1.0 - torch.exp(-cycles.double() / time_constant))


def plot_fits(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """One panel for every curve: the extracted points and the fitted exponential on top.
    fitted_curves are the curves of the dataset with their time constant and temperature rise."""
    columns = 4
    rows = math.ceil(len(fitted_curves) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.4 * rows), sharex=True, squeeze=False)
    for axis, (curve, time_constant, delta_t) in zip(axes.flat, fitted_curves):
        # Every extracted point is drawn, as a point.
        axis.scatter(curve["x"], curve["y"], s=3, linewidths=0, color="tab:blue", label="extracted")
        cycles = torch.linspace(0.0, float(curve["x"].max()), 300)
        axis.plot(cycles, exponential(cycles, time_constant, delta_t), lw=1.5, color="tab:red", label="fit")
        axis.axhline(delta_t, lw=0.6, ls=":", color="tab:red")
        axis.set_title(f"{float(curve['stress_level']):g} % UTS: tau {time_constant:.0f}, dT {delta_t:.3g} °C", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(fitted_curves):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Cycles", fontsize=8)
    figure.suptitle(f"{paper}: theta = dT * (1 - exp(-N / tau))", fontsize=10)
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_parameters(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """The fitted parameters over the stress level: the time constant tau, the temperature rise delta_t and the heating rate
    at the start, from the fit (delta_t / tau) and from the data (the slope of the first INITIAL_CYCLES cycles)."""
    stress_levels = [float(curve["stress_level"]) for curve, _, _ in fitted_curves]
    figure, (tau_axis, delta_t_axis, rate_axis) = plt.subplots(1, 3, figsize=(15, 4))
    tau_axis.plot(stress_levels, [time_constant for _, time_constant, _ in fitted_curves], "o-", color="tab:blue")
    tau_axis.set_ylabel("Time constant tau [cycles]")
    delta_t_axis.plot(stress_levels, [delta_t for _, _, delta_t in fitted_curves], "o-", color="tab:red")
    delta_t_axis.set_ylabel("Temperature rise delta_t [°C]")
    delta_t_axis.set_yscale("log")
    fitted_rates = [1000.0 * delta_t / time_constant for _, time_constant, delta_t in fitted_curves]
    data_rates = []
    for curve, _, _ in fitted_curves:
        first_cycles = curve["x"] <= INITIAL_CYCLES
        data_rates.append(1000.0 * float(np.polyfit(curve["x"][first_cycles].numpy(), curve["y"][first_cycles].numpy(), 1)[0]))
    rate_axis.plot(stress_levels, fitted_rates, "o-", color="tab:green", label="fit: delta_t / tau")
    rate_axis.plot(stress_levels, data_rates, "s--", color="tab:gray", label=f"data: first {INITIAL_CYCLES} cycles")
    rate_axis.set_ylabel("Heating rate at the start [°C per 1000 cycles]")
    rate_axis.set_yscale("log")
    rate_axis.legend(fontsize=8)
    for axis in (tau_axis, delta_t_axis, rate_axis):
        axis.set_xlabel("Stress level [% UTS]")
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"{paper}: parameters of theta = delta_t * (1 - exp(-N / tau))")
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_all_fits(paper: str, fitted_curves: list[tuple[dict, float, float]], plot_path: str | Path) -> None:
    """All curves in one plot, coloured by the stress level: the extracted points and the fitted exponential as a line.
    The temperature rise is on a logarithmic axis, because the curves span a factor of 100 or more."""
    stress_levels = [float(curve["stress_level"]) for curve, _, _ in fitted_curves]
    colors = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(min(stress_levels), max(stress_levels))
    figure, axis = plt.subplots(figsize=(9, 6))
    for curve, time_constant, delta_t in fitted_curves:
        color = colors(norm(float(curve["stress_level"])))
        positive = curve["y"] > 0
        axis.scatter(curve["x"][positive], curve["y"][positive], s=2, linewidths=0, color=color)
        cycles = torch.linspace(5.0, float(curve["x"].max()), 300)
        axis.plot(cycles, exponential(cycles, time_constant, delta_t), lw=1.5, color=color)
    axis.set_yscale("log")
    axis.set_xlabel("Cycles")
    axis.set_ylabel("Temperature rise [°C]")
    axis.grid(True, alpha=0.3)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colors), ax=axis, label="Stress level [% UTS]")
    figure.suptitle(f"{paper}: extracted points and fits (lines) of all curves")
    figure.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def main() -> None:
    dataset = CurveDataset()
    for paper in PAPERS:
        print(f"\n{paper}\n stress level    tau [cycles]    delta_t [°C]    RMSE [°C]    RMSE / delta_t")
        fitted_curves = []
        for curve in sorted((curve for curve in dataset.curves if curve["paper"] == paper),
                            key=lambda curve: float(curve["stress_level"])):
            time_constant, delta_t = fit_exponential(curve["x"], curve["y"])
            rmse = float(((exponential(curve["x"], time_constant, delta_t) - curve["y"].double()) ** 2).mean().sqrt())
            print(f" {float(curve['stress_level']):9.1f} % {time_constant:13.0f} {delta_t:15.3f} {rmse:12.3f} {rmse / delta_t:14.2f}")
            fitted_curves.append((curve, time_constant, delta_t))
        for plot, file_name in ((plot_fits, f"exponential_fit_{paper}.png"),
                                (plot_parameters, f"exponential_parameters_{paper}.png"),
                                (plot_all_fits, f"exponential_fit_all_{paper}.png")):
            plot(paper, fitted_curves, OUTPUT_DIRECTORY / file_name)
            print(f"Wrote {OUTPUT_DIRECTORY / file_name}.")


if __name__ == "__main__":
    main()
