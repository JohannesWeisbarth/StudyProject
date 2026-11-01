"""PyTorch dataset for the extracted temperature curves, read from the results/extracted_curves.json that data_extractor.py writes.

A curve is the temperature rise y in °C over the cycle number x at one stress level in % UTS of one paper.
A sample is one curve: a dict with the paper, the stress level, the cycles x and the temperature rise y of the curve.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


class CurveDataset(Dataset):
    """The data extracted from the papers
    - Montesano 2013
    - Huang 2020
    - Jia 2023."""

    def __init__(self, filepath: str = "results/extracted_curves.json", dtype: torch.dtype = torch.float32) -> None:
        with open(filepath) as file:
            extracted_curves = json.load(file)
        self.curves: list[dict] = []
        for paper, paper_curves in extracted_curves.items():
            for stress_level, curve in paper_curves.items():
                self.curves.append({"paper": paper, "stress_level": torch.tensor(float(stress_level), dtype=dtype),
                                    "x": torch.tensor(curve["x"], dtype=dtype), "y": torch.tensor(curve["y"], dtype=dtype)})

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, index: int) -> dict:
        return self.curves[index]
