"""The plots of the learned/fitted parameters over the stress level."""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Plots are never rendered
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curve_dataset import Curve

# The heating rate of the data is the slope of a straight line through the points of the first seconds: 200 cycles at 5 Hz.
INITIAL_TIME = 40.0


def plot_temperature_plateau(paper: str, stress_levels: list[float], delta_T_stabs: list[float], plot_path: Path) -> None:
    """Plots temperature plateau over stress level in % UTS."""
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(stress_levels, delta_T_stabs, "o-", color="tab:red")
    axis.set_xlabel("Stress level [% UTS]")
    axis.set_ylabel("ΔT_stab [K]")
    axis.set_title(paper)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_time_constant(paper: str, stress_levels: list[float], taus: list[float], plot_path: Path) -> None:
    """The fitted time constant tau over the stress level in % UTS."""
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(stress_levels, taus, "o-", color="tab:blue")
    axis.set_xlabel("Stress level [% UTS]")
    axis.set_ylabel("τ [s]")
    axis.set_title(paper)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_heating_rate(paper: str, stress_levels: list[float], heating_rates: list[float], curves: list[Curve],
                      fitted_temperatures: list[torch.Tensor], plot_path: Path) -> None:
    """The fitted heating rate delta_T_stab / tau over the stress level in % UTS, and the slopes of straight lines through the
    first INITIAL_TIME seconds of the data and of the fit. Both lines are flattened alike by the window, so a gap between them
    shows a mismatch of the fit at the start. Curves without points in that window (Jia2023) have no lines."""
    data_rates, fit_rates = [], []
    for curve, fitted in zip(curves, fitted_temperatures):
        first_seconds = curve["x"] <= INITIAL_TIME
        if first_seconds.sum() < 2:
            data_rates.append(float("nan"))
            fit_rates.append(float("nan"))
        else:
            time = curve["x"][first_seconds].numpy()
            data_rates.append(float(np.polyfit(time, curve["y"][first_seconds].numpy(), 1)[0]))
            fit_rates.append(float(np.polyfit(time, fitted[first_seconds].numpy(), 1)[0]))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(stress_levels, heating_rates, "o-", color="tab:green", label="heating rate")
    axis.plot(stress_levels, fit_rates, "^--", color="tab:green", alpha=0.6, label=f"fit: line through first {INITIAL_TIME:g} s")
    axis.plot(stress_levels, data_rates, "s--", color="tab:gray", label=f"data: line through first {INITIAL_TIME:g} s")
    axis.set_xlabel("Stress level [% UTS]")
    axis.set_ylabel("Heating rate [K/s]")
    axis.set_title(paper)
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_learned_parameters(paper: str, stress_levels: list[float], delta_T_stabs: list[float], taus: list[float],
                            train_stress_levels: list[float], plot_path: Path) -> None:
    """The delta_T_stab and tau learned by the network over the stress level in % UTS, on a grid much finer than the
    curves. The stress levels of the training curves are ticks at the bottom: a bend between two ticks is not backed by a
    training curve and hints at overfitting."""
    figure, (plateau_axis, tau_axis) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    plateau_axis.plot(stress_levels, delta_T_stabs, color="tab:red")
    plateau_axis.set_ylabel("ΔT_stab [K]")
    plateau_axis.set_title(f"{paper}: learned parameters")
    tau_axis.plot(stress_levels, taus, color="tab:blue")
    tau_axis.set_ylabel("τ [s]")
    tau_axis.set_xlabel("Stress level [% UTS]")
    for axis in (plateau_axis, tau_axis):
        # The ticks sit at the bottom of the axis whatever its y scale.
        axis.plot(train_stress_levels, [0.0] * len(train_stress_levels), "|", color="tab:orange", markersize=20,
                  markeredgewidth=2, transform=axis.get_xaxis_transform(), label="training curve")
        axis.grid(True, alpha=0.3)
    plateau_axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_losses(paper: str, train_losses: list[float], validation_losses: list[float], plot_path: Path) -> None:
    """The training loss and the validation loss on the held-out curves over the optimizer steps, in one plot."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(train_losses) + 1), train_losses, color="tab:blue", label="training")
    if validation_losses:
        axis.plot(range(1, len(validation_losses) + 1), validation_losses, color="tab:orange", label="validation (held out)")
    axis.set_xlabel("Step")
    axis.set_ylabel("Relative MSE")
    # The loss spans orders of magnitude from the start to the end of the training.
    axis.set_yscale("log")
    axis.set_title(paper)
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def plot_interpolation(paper: str, train_curves: list[Curve], test_curves: list[Curve], predictions: list[torch.Tensor],
                       baseline_predictions: list[torch.Tensor], plot_path: Path) -> None:
    """The held-out curves of a paper, measured (solid), predicted by the network (dashed) and by the linear interpolation
    baseline (dotted) in the same color, over the training curves in gray."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for curve in train_curves:
        axis.plot(curve["x"], curve["y"], color="lightgray", linewidth=1)
    for index, (curve, prediction, baseline) in enumerate(zip(test_curves, predictions, baseline_predictions)):
        color = plt.cm.tab10(index % 10)
        stress_level = 100 * float(curve["stress_level"])
        axis.plot(curve["x"], curve["y"], color=color, linewidth=1.5, label=f"{stress_level:g} % UTS measured")
        axis.plot(curve["x"], prediction, color=color, linewidth=1.5, linestyle="--", label=f"{stress_level:g} % UTS network")
        axis.plot(curve["x"], baseline, color=color, linewidth=1.5, linestyle=":",
                  label=f"{stress_level:g} % UTS linear interpolation")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Temperature rise [K]")
    axis.set_title(f"{paper}: held-out curves, training curves in gray")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
