"""Normalization of the extracted curves: every curve gets the cycle number on the x axis and the temperature rise on the y axis.

The figures of the papers differ in what they plot. A figure with a time axis in s is converted with the loading frequency
of the test. A figure that plots the absolute temperature is converted to the rise over one ambient temperature for all
curves, the lowest first temperature of all curves. A curve does not start at 0 then, because a test with several stress
levels in a row starts each level at the temperature that the previous level left in the specimen.
"""
from __future__ import annotations

import numpy as np


def normalize_curves(extracted_data: dict[float, np.ndarray], frequency_hz: float | None, y_is_absolute: bool) -> dict[float, np.ndarray]:
    """Return the lines with x in cycles and y as temperature rise.
    frequency_hz is the loading frequency if the x axis of the figure is the time in s, None if it is already the cycles.
    y_is_absolute says that the y axis of the figure is the temperature, not the rise. The ambient temperature is then the
    lowest first temperature of all curves, and it is subtracted from all curves. Cycles below 0 are set to 0."""
    # One common ambient temperature for all curves. A curve does not start at 0: a test with several stress levels starts
    # each level at the temperature that the previous level left in the specimen.
    ambient_temperature = min(line[0, 1] for line in extracted_data.values()) if y_is_absolute else 0.0
    normalized_data: dict[float, np.ndarray] = {}
    for stress_level, line in extracted_data.items():
        x = line[:, 0] if frequency_hz is None else line[:, 0] * frequency_hz
        # The calibration puts the first point of some curves slightly before the start of the test (Huang: -0.36 cycles,
        # less than the accuracy of 0.6 cycles). It belongs to cycle 0.
        x = np.maximum(x, 0.0)
        y = line[:, 1] - ambient_temperature
        normalized_data[stress_level] = np.column_stack([x, y])
    return normalized_data


def x_to_cycles(x_value: float, frequency_hz: float | None) -> float:
    """Convert a value of the x axis, for example its accuracy, to cycles."""
    return x_value if frequency_hz is None else x_value * frequency_hz
