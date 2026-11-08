"""Hold curves out of a CurveDataset: train and test sets for the interpolation and the plot prediction task.

Both tasks are the same kind of split; they differ only in how many curves of a test plot remain in the training set:
- all but a few: interpolation within the plot (split_curves, folds)
- n_context: few-shot plot prediction, the model has seen n_context curves of the new setup (split_plots)
- none: zero-shot plot prediction (split_plots with n_context = 0)
Whole curves are held out, never points from inside a curve. Every split records which curves were held out, so that a result
can be traced back to the test it belongs to.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass

from curve_dataset import CurveDataset, curve_id

# How the curves to hold out are drawn.
# - random: any curve.
# - interior: not the lowest or highest stress level of its paper, so the held-out curve lies between training curves.
# - edge: only the lowest or highest stress level of its paper, so the held-out curve lies outside the training curves.
# - explicit: the curves given by name.
MODES = ("random", "interior", "edge", "explicit")


@dataclass
class CurveSplit:
    """A training and a test set of curves. held_out lists the names of the test curves."""
    train: CurveDataset
    test: CurveDataset
    held_out: list[str]


def _candidates(dataset: CurveDataset, mode: str) -> dict[str, list[str]]:
    """The names of the curves that may be held out in this mode, for every paper."""
    candidates: dict[str, list[str]] = {paper: [] for paper in dataset.paper_names}
    for paper in dataset.paper_names:
        paper_curves = [curve for curve in dataset.curves if curve["paper_name"] == paper]
        stress_levels = [float(curve["stress_level"]) for curve in paper_curves]
        lowest, highest = min(stress_levels), max(stress_levels)
        for curve in paper_curves:
            is_edge = float(curve["stress_level"]) in (lowest, highest)
            if mode == "random" or (mode == "edge" and is_edge) or (mode == "interior" and not is_edge):
                candidates[paper].append(curve_id(curve))
    return candidates


def _make_split(dataset: CurveDataset, held_out: list[str], every_paper_trained: bool = True) -> CurveSplit:
    """Train on everything but the held-out curves, test on the held-out curves. With every_paper_trained, every paper must keep
    a training curve (interpolation); without, a paper may be held out completely (zero-shot plot prediction)."""
    test_names = set(held_out)
    train = [curve for curve in dataset.curves if curve_id(curve) not in test_names]
    test = [curve for curve in dataset.curves if curve_id(curve) in test_names]
    starved = set(dataset.paper_names) - {curve["paper_name"] for curve in train}
    if every_paper_trained and starved:
        raise ValueError(f"The split leaves no training curve for {sorted(starved)}, lower n_out.")
    if not train:
        raise ValueError("The split has no training curves.")
    if not test:
        raise ValueError("The split has no test curves.")
    return CurveSplit(CurveDataset.from_curves(train), CurveDataset.from_curves(test), sorted(test_names))


def split_curves(dataset: CurveDataset, n_out: int, mode: str = "random", seed: int = 0,
                 explicit: list[str] | None = None) -> CurveSplit:
    """Hold out n_out curves of every paper, drawn in the given mode with the seed, so that every paper has test curves. In the
    explicit mode the curves are the names in explicit ("Jia2023@87.5") and n_out is ignored."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}, available: {MODES}")
    if mode == "explicit":
        names = set(curve_id(curve) for curve in dataset.curves)
        missing = [name for name in explicit or () if name not in names]
        if not explicit or missing:
            raise ValueError(f"explicit needs curve names from the dataset, missing: {missing}")
        return _make_split(dataset, list(explicit))
    rng = random.Random(seed)
    held_out = []
    for paper, candidates in _candidates(dataset, mode).items():
        if n_out > len(candidates):
            raise ValueError(f"n_out={n_out} but only {len(candidates)} curves of {paper} can be held out in mode {mode!r}")
        held_out += rng.sample(candidates, n_out)
    return _make_split(dataset, held_out)


def folds(dataset: CurveDataset, n_out: int, mode: str = "random", max_folds: int | None = None,
          seed: int = 0) -> Iterator[CurveSplit]:
    """Every hold-out of n_out curves of the whole dataset in turn. With max_folds, a random sample of that many folds, the same
    for the same seed."""
    candidates = [name for paper_candidates in _candidates(dataset, mode).values() for name in paper_candidates]
    combinations = list(itertools.combinations(candidates, n_out))
    if max_folds is not None and max_folds < len(combinations):
        combinations = random.Random(seed).sample(combinations, max_folds)
    for held_out in combinations:
        yield _make_split(dataset, list(held_out))


def split_plots(dataset: CurveDataset, test_papers: list[str], n_context: int = 0, seed: int = 0) -> CurveSplit:
    """Hold out the plots of test_papers. n_context randomly drawn curves of each held-out plot stay in the training set as its
    context, the other curves of the plot are the test curves. Papers from the same lab or material belong in the same
    test_papers, otherwise the plot is not really unseen."""
    unknown = set(test_papers) - set(dataset.paper_names)
    if unknown:
        raise ValueError(f"Papers not in the dataset: {sorted(unknown)}")
    rng = random.Random(seed)
    held_out = []
    for paper in test_papers:
        names = [curve_id(curve) for curve in dataset.curves if curve["paper_name"] == paper]
        if n_context >= len(names):
            raise ValueError(f"n_context={n_context} leaves no test curve of {paper}, which has {len(names)} curves.")
        context = set(rng.sample(names, n_context))
        held_out += [name for name in names if name not in context]
    return _make_split(dataset, held_out, every_paper_trained=False)
