"""Fit a simple exponential to every curve of the selected papers on its own, with Adam on the MSE loss, and plot the fitted
parameters over the stress level: the temperature plateau delta_T_stab, the time constant tau and the heating rate
delta_T_stab / tau.

    python fit_simple_exponential.py --papers Huang2019_IJF Huang2020_EFM
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from curve_dataset import Curve, CurveDataset
from models.simple_exponential import SimpleExponential, exponential
from plots import plot_heating_rate, plot_temperature_plateau, plot_time_constant

logger = logging.getLogger(__name__)

# All results are written to this directory, in the directory the script is started in, one folder per paper.
OUTPUT_DIRECTORY = Path("results") / "simple_exponential_fits"


def fit_simple_exponential(curve: Curve, steps: int, learning_rate: float) -> SimpleExponential:
    """Fit a simple exponential to one curve with Adam on the MSE loss, full batch."""
    model = SimpleExponential(curve["x"], curve["y"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(input=model(curve["x"]), target=curve["y"].double())
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            logger.debug("step %d: MSE %.4g K², delta_T_stab %.4g K, tau %.4g s", step, loss.item(), model.delta_T_stab.item(),
                         model.tau.item())
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", nargs="+", help="the papers to fit (default: all)")
    parser.add_argument("--steps", type=int, default=3000, help="number of optimizer steps per curve")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="learning rate")
    args = parser.parse_args()

    # The curves of a paper are sorted by stress level in the dataset.
    dataset = CurveDataset(papers=args.papers)
    for paper in dataset.paper_names:
        curves = [curve for curve in dataset.curves if curve["paper_name"] == paper]
        stress_levels, delta_T_stabs, taus = [], [], []
        logger.info("%s\n stress level    delta_T_stab [K]    tau [s]    RMSE [K]    RMSE / delta_T_stab", paper)
        for curve in curves:
            model = fit_simple_exponential(curve, args.steps, args.learning_rate)
            stress_level, delta_T_stab, tau = 100 * float(curve["stress_level"]), model.delta_T_stab.item(), model.tau.item()
            # PyTorch has no RMSE loss, it is the square root of the MSE.
            rmse = F.mse_loss(exponential(curve["x"], delta_T_stab, tau), curve["y"].double()).sqrt().item()
            logger.info(" %9.1f %% %19.3f %10.1f %11.3f %22.2f", stress_level, delta_T_stab, tau, rmse, rmse / delta_T_stab)
            stress_levels.append(stress_level)
            delta_T_stabs.append(delta_T_stab)
            taus.append(tau)

        plot_temperature_plateau(paper, stress_levels, delta_T_stabs, OUTPUT_DIRECTORY / paper / "temperature_plateau.png")
        plot_time_constant(paper, stress_levels, taus, OUTPUT_DIRECTORY / paper / "time_constant.png")
        heating_rates = [delta_T_stab / tau for delta_T_stab, tau in zip(delta_T_stabs, taus)]
        fitted_temperatures = [exponential(curve["x"], delta_T_stab, tau) for curve, delta_T_stab, tau in zip(curves, delta_T_stabs, taus)]
        plot_heating_rate(paper, stress_levels, heating_rates, curves, fitted_temperatures,
                          OUTPUT_DIRECTORY / paper / "heating_rate.png")
        logger.info("Wrote the plots of %s to %s.", paper, OUTPUT_DIRECTORY / paper)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
