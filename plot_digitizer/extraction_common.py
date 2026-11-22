"""Common functionality for extraction. Defines result return types, and shared constants and functions.

What is known about a figure before the extraction is its FigureMetadata, hardcoded in main of data_extractor.py. What the
extraction computes is its ExtractionResult. Every step is a function without side effects that returns only its own result.
extract_figure in data_extractor.py runs the steps one after the other, collects their results and writes them into the
ExtractionResult at the end:
1. extract_vector_plots.py, extract_raster_plots.py: extract the data of a figure, return an ExtractedFigure,
2. normalize_curves (below): the curves with the time in s on the x axis and the temperature rise on the y axis,
3. x_to_seconds (below): the x accuracy in s.
save_results.py then writes the data, the normalized curves and the plots.

evaluate_extraction.py is a separate script. It computes statistics from the written extracted_curves.json.
"""
from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

# Maximum distance between a figure and its caption in pt.
MAX_GAP = 150


class ExtractionError(Exception):
    """A figure could not be extracted, for example because the PDF or the figure does not look as expected."""


def hex_to_rgb(hex_color: str) -> np.ndarray:
    """Convert a hex number to an RGB array in 0..255, e.g. '#8c6365' -> array([140., 99., 101.])."""
    return np.array([int(hex_color.lstrip("#")[index:index + 2], 16) for index in (0, 2, 4)], dtype=float)


def rgb_to_hex(color: np.ndarray) -> str:
    """Convert an RGB array in 0..255 to a hex number, e.g. array([140., 99., 101.]) -> '#8c6365'."""
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in color)


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


class FigureMetadata(NamedTuple):
    """What is known about a figure before it is extracted, hardcoded in main of data_extractor.py."""
    paper_name: str                                 # name of the PDF file without .pdf
    figure: int                                     # number of the figure in its caption
    x_unit: str                                     # the unit of the x axis: "s", "h" or "cycles"
    y_is_absolute: bool                             # the y axis is the absolute temperature (True) or the temperature rise (False)
    setup: dict[str, float | str | list[float] | None]  # the test setup of the paper, None where the paper does not state it
    stress_unit: str = "% UTS"                      # the unit of the stress levels: "% UTS" or "MPa" (σ_max)
    dataset_name: str | None = None                 # the name of the curves in extracted_curves.json, if not paper_name


class ExtractedFigure(NamedTuple):
    """What the extraction function of a figure returns, in the units of the figure."""
    extracted_data: dict[float, np.ndarray]         # Maps stress level -> line, an array (n, 2) of x and y, sorted by x
    original_plot: np.ndarray                       # The original figure from the paper, RGB pixels
    calibration_data: PixelCalibration              # Maps pixels of original_plot to data values
    line_colors: dict[float, np.ndarray] | None     # Line colours from the paper (RGB in 0..1) for the extracted plot, None if all lines are black
    x_accuracy: float                               # uncertainty of the x values in the unit of the x axis
    y_accuracy: float                               # uncertainty of the y values in the unit of the y axis
    # The x value at which each stress level starts, for a figure that shows the whole test as one line over the time.
    # None for a figure in which every stress level starts at x = 0.
    block_starts: dict[float, float] | None = None


class ExtractionResult(NamedTuple):
    """All results of the extraction of a figure, collected by extract_figure at the end."""
    extracted_data: dict[float, np.ndarray]         # Maps stress level -> line, an array (n, 2) of x and y, sorted by x
    original_plot: np.ndarray                       # The original figure from the paper, RGB pixels
    calibration_data: PixelCalibration              # Maps pixels of original_plot to data values
    line_colors: dict[float, np.ndarray] | None     # Line colours from the paper (RGB in 0..1) for the extracted plot, None if all lines are black
    normalized_data: dict[float, np.ndarray]        # stress level -> line (n, 2) with x in s and y as temperature rise in K
    normalized_x_accuracy: float                    # uncertainty of the normalized x values in s
    normalized_y_accuracy: float                    # uncertainty of the normalized y values in K


def x_to_seconds(x: np.ndarray | float, metadata: FigureMetadata) -> np.ndarray | float:
    """Convert x values of the figure, an array or a number, to s. A cycle axis is divided by the loading frequency."""
    if metadata.x_unit == "s":
        return x
    if metadata.x_unit == "h":
        return 3600.0 * x
    return x / metadata.setup["frequency_hz"]


def normalize_curves(extracted_data: dict[float, np.ndarray], metadata: FigureMetadata,
                     block_starts: dict[float, float] | None) -> dict[float, np.ndarray]:
    """Return the lines with x in s and y as temperature rise.
    If the y axis of the figure is the absolute temperature, not the rise, the ambient temperature is the lowest first
    temperature of all curves, and it is subtracted from all curves. If the figure shows the whole test as one line, the
    time of every stress level is counted from the start of its block. Times below 0 are set to 0."""
    # One common ambient temperature for all curves. A curve does not start at 0: a test with several stress levels starts
    # each level at the temperature that the previous level left in the specimen.
    ambient_temperature = min(line[0, 1] for line in extracted_data.values()) if metadata.y_is_absolute else 0.0
    normalized_data: dict[float, np.ndarray] = {}
    for stress_level, line in extracted_data.items():
        block_start = block_starts[stress_level] if block_starts else 0.0
        x = x_to_seconds(line[:, 0] - block_start, metadata)
        # The calibration puts the first point of some curves slightly before the start of the test (Huang: -0.07 s,
        # less than the accuracy of 0.12 s). It belongs to time 0.
        x = np.maximum(x, 0.0)
        y = line[:, 1] - ambient_temperature
        normalized_data[stress_level] = np.column_stack([x, y])
    return normalized_data
