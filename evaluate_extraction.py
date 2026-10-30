"""Evaluation of an extraction: the statistics of the extracted curves.

The statistics have the same columns for every figure, so the tables of several papers can be put together:
- paper_pdf: name of the PDF file of the paper,
- stress_level: stress level of the curve in % UTS, read from its name,
- data_points: number of extracted points of the curve,
- delta_t_plateau: temperature in the plateau, the median of the curve in the plateau,
- temperature_jitter_plateau: standard deviation of the curve around a straight line through the plateau.
Both temperatures are given in the unit of the y axis of the figure.

The plateau is the last quarter of the x range of the figure. A curve that does not reach it, for example a curve
that ends early because the specimen failed, has no plateau and gets no temperatures.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# The plateau is this last fraction of the x range of the figure.
PLATEAU_FRACTION = 0.25
# A curve with fewer points in the plateau has no plateau.
MIN_PLATEAU_POINTS = 10


def parse_stress_level(curve_name: str) -> float:
    """Read the stress level in % UTS from the name of a curve, e.g. 85 from '85% UTS'. NaN if the name has none."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*%", curve_name)
    return float(match.group(1)) if match else float("nan")


def compute_statistics(extracted_data: pd.DataFrame, paper_pdf: str) -> pd.DataFrame:
    """Compute the statistics of every extracted curve, sorted by stress level. The columns are described above."""
    plateau_start = extracted_data.x.max() - PLATEAU_FRACTION * (extracted_data.x.max() - extracted_data.x.min())
    rows = []
    for curve_name, curve in extracted_data.groupby("series", sort=False):
        plateau = curve[curve.x >= plateau_start]
        delta_t_plateau, temperature_jitter_plateau = float("nan"), float("nan")
        if len(plateau) >= MIN_PLATEAU_POINTS:
            delta_t_plateau = float(plateau.y.median())
            # The jitter is measured around a straight line, so that a slow rise or fall in the plateau does not count.
            trend = np.polyval(np.polyfit(plateau.x, plateau.y, 1), plateau.x)
            temperature_jitter_plateau = float(np.std(plateau.y - trend))
        rows.append({"paper_pdf": paper_pdf, "stress_level": parse_stress_level(str(curve_name)),
                     "data_points": len(curve), "delta_t_plateau": delta_t_plateau,
                     "temperature_jitter_plateau": temperature_jitter_plateau})
    return pd.DataFrame(rows).sort_values("stress_level", kind="stable", ignore_index=True)
