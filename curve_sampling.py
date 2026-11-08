"""The dataset class loads single curves. The curves must be batched for training / validation.

To this end, two samplers are provided, for two different tasks:
- InterpolationSampler: Only samples curves from a single paper. N many curves are held out (split_curves in curve_split.py).
  The task then is to interpolate the curves.
- RandomCurveSampler: Mixes curves from different papers. N many papers are held out (split_plots in curve_split.py). The task,
  then, is to predict heating behavior of the PMC across test setups.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TypedDict

import torch

from curve_dataset import Curve, CurveDataset


class RandomCurveSampler:
    """Random batches of batch_size many curves from any paper. The last batch of an epoch may be smaller."""

    def __init__(self, dataset: CurveDataset, batch_size: int = 8, seed: int = 0) -> None:
        self.size, self.batch_size = len(dataset), batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.size, generator=self.generator).tolist()
        for start in range(0, self.size, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


class InterpolationSampler:
    """Restricts batch sampling to one paper, so all curves in the batch share their test setup."""

    def __init__(self, dataset: CurveDataset, batch_size: int | None = 8, seed: int = 0) -> None:
        # This is a map paper_name -> curve_indices
        self.indices: dict[str, list[int]] = {paper_name: [] for paper_name in dataset.paper_names}
        for index, curve in enumerate(dataset.curves):
            # Group the indices of the curves by paper name.
            self.indices[curve["paper_name"]].append(index)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[list[int]]:
        batches = []
        for indices in self.indices.values():
            batch_size = self.batch_size or len(indices)
            shuffled_indices = torch.randperm(len(indices), generator=self.generator).tolist()
            shuffled = [indices[position] for position in torch.randperm(len(indices), generator=self.generator).tolist()]
            # Collect the batches. Note that the last batch is smaller!
            batches += [shuffled[start:start + batch_size] for start in range(0, len(shuffled), batch_size)]
        for position in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[position]

    def __len__(self) -> int:
        n_batches = 0
        for indices in self.indices.values():
            batch_size = self.batch_size or len(indices)
            n_batches += math.ceil(len(indices) / batch_size)
        return n_batches


class CurveBatch(TypedDict):
    """The curves of a batch, zero-padded to the longest one."""
    # The time in seconds, a tensor of shape (n_curves, n_points of the longest curve).
    x: torch.Tensor
    # The temperature rise in K, a tensor of the same shape as x.
    y: torch.Tensor
    # True where a point is real, False where it is padding, a tensor of the same shape as x.
    mask: torch.Tensor
    # The stress level of every curve as a fraction of UTS, a tensor of shape (n_curves,).
    stress_level: torch.Tensor
    # The paper of every curve.
    paper_name: list[str]
    # The setup parameters of every curve's paper.
    setup: list[dict[str, float | str | list[float] | None]]


def collate_curves(curves: list[Curve]) -> CurveBatch:
    """Zero-pads the curves of a batch to the longest one."""
    longest = max(len(curve["x"]) for curve in curves)
    x = torch.zeros(len(curves), longest, dtype=curves[0]["x"].dtype)
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for row, curve in enumerate(curves):
        length = len(curve["x"])
        x[row, :length], y[row, :length], mask[row, :length] = curve["x"], curve["y"], True
    return {"x": x, "y": y, "mask": mask,
            "stress_level": torch.stack([curve["stress_level"] for curve in curves]),
            "paper_name": [curve["paper_name"] for curve in curves],
            "setup": [curve["setup"] for curve in curves]}
