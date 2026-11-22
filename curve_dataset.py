"""PyTorch dataset of the extracted temperature curves. This dataset does not
impose any grouping or restrictions on the sampled data, this is handled by separate samplers!
"""
from __future__ import annotations

import json
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class Curve(TypedDict):
    """The temperature rise over time at one stress level of one paper."""
    # Name of the paper the curve was extracted from.
    paper_name: str
    # The stress level as a fraction of UTS, e.g. 0.875 for 87.5 % UTS. NaN for a paper that gives σ_max in MPa but no UTS.
    stress_level: torch.Tensor
    # The setup parameters of the paper.
    setup: dict[str, float | str | list[float] | None]
    # The time in seconds, a tensor of shape (n_points,).
    x: torch.Tensor
    # The temperature rise in K at the discrete times in x, a tensor of shape (n_points,).
    y: torch.Tensor


def curve_id(curve: Curve) -> str:
    """Creates a unique name for each curve from paper_name@stress_level, e.g. "Jia2023@87.5"."""
    return f"{curve['paper_name']}@{100.0 * float(curve['stress_level']):g}"


class CurveDataset(Dataset):
    """Represents the dataset extracted from the papers
    - Montesano 2013
    - Huang 2019 and Huang 2020
    - Jia 2023
    - Najd 2022 (pristine and smart specimen)
    - Dolbachin 2025."""

    def __init__(self, papers: list[str] | None = None, filepath: str = "results/digitization/extracted_curves.json", dtype: torch.dtype = torch.float32) -> None:
        """The curves of the given papers, all papers if None, in the order of the papers in the file and, within a paper,
        sorted by stress level. A paper gives its stress levels in % UTS or as σ_max in MPa, both are converted to a
        fraction of UTS."""
        with open(filepath) as file:
            extracted_curves = json.load(file)
        if papers is None:
            papers = list(extracted_curves)
        for paper_name in papers:
            if paper_name not in extracted_curves:
                raise ValueError(f"Unknown paper {paper_name!r}. Choose papers from: {sorted(extracted_curves)}!")
        self.curves: list[Curve] = []
        for paper_name, paper_data in extracted_curves.items():
            if paper_name not in papers:
                continue
            if paper_data["stress_unit"] == "% UTS":
                stress_scale = 100.0
            else:
                uts_mpa = paper_data["setup"]["uts_mpa"]
                stress_scale = uts_mpa if uts_mpa is not None else float("nan")
            for stress_level, curve in sorted(paper_data["curves"].items(), key=lambda item: float(item[0])):
                self.curves.append({"paper_name": paper_name,
                                    "stress_level": torch.tensor(float(stress_level) / stress_scale, dtype=dtype),
                                    "setup": paper_data["setup"],
                                    "x": torch.tensor(curve["x"], dtype=dtype),
                                    "y": torch.tensor(curve["y"], dtype=dtype)})

    @classmethod
    def from_curves(cls, curves: list[Curve]) -> CurveDataset:
        """Creates a dataset from the given curves. Does not copy the curves."""
        dataset = cls.__new__(cls)
        dataset.curves = list(curves)
        return dataset

    @property
    def paper_names(self) -> list[str]:
        """Lists the papers in the dataset, in alphanumerical order."""
        return sorted({curve["paper_name"] for curve in self.curves})

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, index: int) -> Curve:
        return self.curves[index]
