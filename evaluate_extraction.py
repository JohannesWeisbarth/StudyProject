"""Evaluation of an extraction. For each extracted plot, some data is collected:

- paper_pdf: name of the PDF file of the paper,
- stress_level: stress level of the curve in % UTS,
- data_points: number of extracted points for the curve,
- cycles: number of cycles for each stress_level
- delta_t_plateau: temperature rise in the plateau, the median of the curve in the plateau,
- x_accuracy_estimate, y_accuracy_estimate: uncertainty of the x and y values of the figure, in cycles and °C. It is the same
  for all curves of a figure and comes from the calibration and the resolution of the extraction.
The curves are the normalized ones from normalize_data.py: x in cycles, y as temperature rise.

The plateau is the last quarter of the x range of the figure. A curve that does not reach it, for example a curve
that ends early because the specimen failed, has no plateau and gets no temperatures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The plateau is this last fraction of the x range of the figure.
PLATEAU_FRACTION = 0.25
# A curve with fewer points in the plateau has no plateau.
MIN_PLATEAU_POINTS = 10


def compute_statistics(extracted_data: dict[float, np.ndarray], paper_pdf: str, x_accuracy: float, y_accuracy: float) -> pd.DataFrame:
    """Compute the statistics of every extracted line, sorted by stress level. The columns are described above."""
    x_min, x_max = min(line[:, 0].min() for line in extracted_data.values()), max(line[:, 0].max() for line in extracted_data.values())
    plateau_start = x_max - PLATEAU_FRACTION * (x_max - x_min)
    rows = []
    for stress_level, line in sorted(extracted_data.items()):
        in_plateau = line[:, 0] >= plateau_start
        plateau_x, plateau_y = line[in_plateau, 0], line[in_plateau, 1]
        delta_t_plateau, temperature_jitter_plateau = float("nan"), float("nan")
        if len(plateau_x) >= MIN_PLATEAU_POINTS:
            delta_t_plateau = float(np.median(plateau_y))
            # The jitter is measured around a straight line, so that a slow rise or fall in the plateau does not count.
            trend = np.polyval(np.polyfit(plateau_x, plateau_y, 1), plateau_x)
            temperature_jitter_plateau = float(np.var(plateau_y - trend))
        rows.append({"paper_pdf": paper_pdf, "stress_level": stress_level, "data_points": len(line),
                     "cycles": float(line[:, 0].max()), "delta_t_plateau": delta_t_plateau,
                     "temperature_jitter_plateau": temperature_jitter_plateau,
                     "x_accuracy": x_accuracy, "y_accuracy": y_accuracy})
    return pd.DataFrame(rows)
