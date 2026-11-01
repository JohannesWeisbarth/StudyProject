"""Common functionality for extraction. Defines result return types, and shared constants and functions.

The steps are:
1. extract_vector_plots.py, extract_raster_plots.py: extract the data of a figure, returns an ExtractionResult,
2. normalize_data.py: convert the curves to cycles on the x axis and the temperature rise on the y axis,
3. evaluate_extraction.py: compute statistics of the normalized curves,
4. save_results.py: write the data, the normalized curves, the plots and the statistics,
5. data_extractor.py: runs all steps for all papers. Its extract_figure returns a FigureResult.

curve_dataset.py is not part of these steps. It reads the written extracted_curves.json for training.
"""
from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pandas as pd

# Maximum distance between a figure and its caption in pt.
MAX_GAP = 150


def parse_stress_level(curve_label: str) -> float:
    """Parse the stress level in % UTS from the label of a curve, e.g. 85.0 from '85% UTS'. NaN if the name has none."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*%", curve_label)
    return float(match.group(1)) if match else float("nan")


class PixelCalibration(NamedTuple):
    """Linearly maps pixel column -> x and pixel row -> y for the original image,
    so that the extracted data can be drawn onto it."""
    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float


class ExtractionResult(NamedTuple):
    """Result of an extraction function."""
    extracted_data: dict[float, np.ndarray]               # Maps stress level (% UTS) -> line, an array (n, 2) of x and y, sorted by x
    original_plot: np.ndarray                       # The original figure from the paper, RGB pixels
    calibration_data: PixelCalibration              # Maps pixels of original_plot to data values
    line_colors: dict[float, np.ndarray] | None     # Line colours from the paper (RGB in 0..1) for the extracted plot, None if all lines are black
    x_accuracy: float                               # uncertainty of the x values in the unit of the x axis
    y_accuracy: float                               # uncertainty of the y values in the unit of the y axis


class FigureResult(NamedTuple):
    """Collects all results for a figure."""
    paper: str                                      # name of the PDF file without .pdf
    fig: int                                        # number of the figure in its caption
    extraction_result: ExtractionResult             # what the extraction function returned, in the units of the figure
    normalized_data: dict[float, np.ndarray]              # stress level -> line (n, 2) with x in cycles and y as temperature rise
    statistics: pd.DataFrame                        # one row per curve
