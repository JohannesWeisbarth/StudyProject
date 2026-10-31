"""The results that are passed from one step of the extraction to the next, and the constants that several steps share.

The steps are:
1. extract_vector_plots.py, extract_raster_plots.py: extract the data of a figure, returns an ExtractionResult,
2. evaluate_extraction.py: compute the statistics of the extracted curves,
3. save_results.py: write the data, the plots and the statistics,
4. data_extractor.py: runs all steps for all papers. Its extract_figure returns a FigureResult.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# Maximum distance between a figure and its caption in pt.
MAX_GAP = 150


class PixelCalibration(NamedTuple):
    """Linear maps pixel column -> x and pixel row -> y of the original image,
    so that the extracted data can be drawn onto it."""
    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float


class ExtractionResult(NamedTuple):
    """What the function of a plot returns."""
    extracted_data: pd.DataFrame                    # columns series, x, y
    original_plot: np.ndarray                       # the figure as it appears in the paper, RGB pixels
    calibration_data: PixelCalibration              # maps pixels of original_plot to data values
    line_colors: dict[str, np.ndarray] | None       # RGB in 0..1 per series for the extracted plot, None for default colours
    x_accuracy: float                               # uncertainty of the x values in the unit of the x axis
    y_accuracy: float                               # uncertainty of the y values in the unit of the y axis


class FigureResult(NamedTuple):
    """Everything that is known about an extracted figure. save_results.py writes the files from it."""
    paper: str                                      # name of the PDF file without .pdf
    fig: int                                        # number of the figure in its caption
    extracted_data: pd.DataFrame                    # columns series, x, y
    original_plot: np.ndarray                       # the figure as it appears in the paper, RGB pixels
    calibration_data: PixelCalibration              # maps pixels of original_plot to data values
    line_colors: dict[str, np.ndarray] | None       # RGB in 0..1 per series for the extracted plot, None for default colours
    statistics: pd.DataFrame                        # one row per curve
