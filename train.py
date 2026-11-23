#!/usr/bin/env python3
"""Train the parameterized exponential on the temperature curves and plot its training loss, its validation loss on the
held-out curves and its predictions of the held-out curves.

    python train.py --papers Huang2020_EFM --n-held-out 2 --hold-out-mode interior --mode curve_interpolation

The parameterized exponential learns delta_T_stab and tau as functions of the stress level and the setup. One model is trained
on the curves of all selected papers. A held-out curve is predicted at its stress level and setup.
The simple exponential, a curve fit and not a learned model, is fitted in fit_simple_exponential.py.

The baseline is linear interpolation: the simple exponential is fitted to every training curve of a paper, and log delta_T_stab
and log tau of a held-out curve are interpolated linearly over the stress level between its neighbouring training curves.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from curve_dataset import Curve, CurveDataset, curve_id
from curve_sampling import CurveBatch, InterpolationSampler, RandomCurveSampler, collate_curves
from curve_split import split_curves
from fit_simple_exponential import fit_simple_exponential
from models.parameterized_exponential import ParameterizedExponential
from models.simple_exponential import SimpleExponential, exponential
from plots import plot_interpolation, plot_losses

logger = logging.getLogger(__name__)

# All results are written to this directory, in the directory the script is started in, one folder per paper.
# The simple exponential of fit_simple_exponential.py writes to the folder simple_exponential next to it.
OUTPUT_DIRECTORY = Path("results") / "training" / "parameterized_exponential"
# The samplers that --mode chooses from, by task.
SAMPLERS = {"curve_interpolation": InterpolationSampler, "plot_prediction": RandomCurveSampler}
# The seed of the network initialization, the batch order and the choice of the held-out curves.
SEED = 0


def relative_mse(prediction: torch.Tensor, batch: CurveBatch) -> torch.Tensor:
    """The mean squared error of every curve relative to its largest measured temperature rise, averaged over the curves.
    Within a paper the plateaus span orders of magnitude, so an MSE in K² would only fit the curves of the highest stress
    levels. Relative to its own scale, every curve counts the same, independent of its number of points."""
    scale = batch["y"].double().abs().amax(dim=1).clamp_min(1e-6)
    squared = ((prediction - batch["y"].double()) / scale[:, None]) ** 2
    mask = batch["mask"]
    return ((squared * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def linear_interpolation_baseline(train_curves: list[Curve], batch: CurveBatch) -> torch.Tensor:
    """The temperature rise in K of every curve of the batch, of the same shape as batch["x"], predicted by the baseline from
    the training curves of the same paper. The simple exponential is fitted to every training curve with the defaults of
    fit_simple_exponential.py, and log delta_T_stab and log tau are interpolated linearly over the stress level, which follows
    their fast growth with the stress level better than interpolating the values themselves. Outside the stress levels of the
    training curves, the values of the outermost training curve are used."""
    fits = [fit_simple_exponential(curve, steps=3000, learning_rate=0.05) for curve in train_curves]
    # The curves of a paper are sorted by stress level in the dataset, as np.interp requires.
    stress_levels = [float(curve["stress_level"]) for curve in train_curves]
    log_delta_T_stab = np.interp(batch["stress_level"].numpy(), stress_levels, [fit.log_delta_T_stab.item() for fit in fits])
    log_tau = np.interp(batch["stress_level"].numpy(), stress_levels, [fit.log_tau.item() for fit in fits])
    return exponential(batch["x"], torch.from_numpy(np.exp(log_delta_T_stab))[:, None], torch.from_numpy(np.exp(log_tau))[:, None])


def train_model(train: CurveDataset, test: CurveDataset, args: argparse.Namespace
                ) -> tuple[ParameterizedExponential, list[float], dict[str, list[float]]]:
    """Trains one model with Adam on batches of the training curves. Returns the model, its training loss after every step and,
    for every paper with held-out curves, the validation loss on them after every step."""
    # The network starts at the mean of the starting values of the single-curve fits.
    guesses = [SimpleExponential(curve["x"], curve["y"]) for curve in train.curves]
    model = ParameterizedExponential(initial_log_delta_T_stab=sum(guess.log_delta_T_stab.item() for guess in guesses) / len(guesses),
                                     initial_log_tau=sum(guess.log_tau.item() for guess in guesses) / len(guesses))
    dataloader = DataLoader(train, batch_sampler=SAMPLERS[args.mode](train, batch_size=args.batch_size, seed=SEED),
                        collate_fn=collate_curves)
    validation_batches = {paper: collate_curves([curve for curve in test.curves if curve["paper_name"] == paper])
                          for paper in test.paper_names}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_losses: list[float] = []
    validation_losses: dict[str, list[float]] = {paper: [] for paper in validation_batches}
    step = 0
    while step < args.steps:
        for batch in dataloader:
            optimizer.zero_grad()
            loss = relative_mse(model(batch), batch)
            loss.backward()
            optimizer.step()
            step += 1
            train_losses.append(loss.item())
            with torch.no_grad():
                for paper, validation_batch in validation_batches.items():
                    validation_losses[paper].append(relative_mse(model(validation_batch), validation_batch).item())
            if step % 5000 == 0:
                logger.info("step %d: training loss %.4g, validation loss %s", step, train_losses[-1],
                            {paper: f"{losses[-1]:.4g}" for paper, losses in validation_losses.items()} or "none")
            if step == args.steps:
                break
    return model, train_losses, validation_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10000, help="number of optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--papers", nargs="+", help="the papers to use (default: all)")
    parser.add_argument("--n-held-out", type=int, default=0, help="number of curves held out of every selected paper (default: 0)")
    parser.add_argument("--hold-out-mode", choices=["random", "interior"], default="random",
                        help="sets which curves are held out. Options are: randomly selected, interior (not the outmost curves)")
    parser.add_argument("--mode", choices=["curve_interpolation", "plot_prediction"], default="curve_interpolation", help="how training batches are formed")
    parser.add_argument("--batch-size", type=int, default=8, help="number of curves per batch, for both samplers")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    dataset = CurveDataset(papers=args.papers)
    # A paper with a setup value None does not provide complete input. It is skipped.
    skipped_papers = sorted({curve["paper_name"] for curve in dataset.curves if None in curve["setup"].values()})
    if skipped_papers:
        logger.info("Skipped for missing setup values: %s", skipped_papers)
        dataset = CurveDataset.from_curves([curve for curve in dataset.curves if curve["paper_name"] not in skipped_papers])
    if args.n_held_out > 0:
        split = split_curves(dataset, args.n_held_out, args.hold_out_mode, SEED)
        train, test = split.train, split.test
    else:
        train, test = dataset, CurveDataset.from_curves([])
    logger.info("%d training curves, held out: %s", len(train), [curve_id(curve) for curve in test.curves] or "none")

    model, train_losses, validation_losses = train_model(train, test, args)

    for paper in train.paper_names:
        plot_losses(paper, train_losses, validation_losses.get(paper, []), OUTPUT_DIRECTORY / paper / "loss.png")
        logger.info("Wrote %s.", OUTPUT_DIRECTORY / paper / "loss.png")
        paper_test = [curve for curve in test.curves if curve["paper_name"] == paper]
        if paper_test:
            paper_train = [curve for curve in train.curves if curve["paper_name"] == paper]
            test_batch = collate_curves(paper_test)
            with torch.no_grad():
                network = model(test_batch)
            baseline = linear_interpolation_baseline(paper_train, test_batch)
            logger.info("%s, relative MSE on the held-out curves: network %.4g, linear interpolation %.4g", paper,
                        relative_mse(network, test_batch).item(), relative_mse(baseline, test_batch).item())
            # The predictions without the padding of the batch.
            network_predictions = [network[row, :len(curve["x"])] for row, curve in enumerate(paper_test)]
            baseline_predictions = [baseline[row, :len(curve["x"])] for row, curve in enumerate(paper_test)]
            plot_interpolation(paper, paper_train, paper_test, network_predictions, baseline_predictions,
                               OUTPUT_DIRECTORY / paper / "interpolation.png")
            logger.info("Wrote %s.", OUTPUT_DIRECTORY / paper / "interpolation.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
