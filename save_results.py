"""Writing of the results of the extraction: the data, the plots and the statistics.
All results are written to the results directory, in the directory the script is started in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Plots are never rendered
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extraction_common import FigureResult, PixelCalibration

# Area of the extracted points in the overlay in pt^2. There is about one point per pixel column, so larger points
# merge into a solid line and hide the colour of the original line.
MARKER_SIZE = 0.3


def save_original_plot(original_plot: np.ndarray, plot_path: str | Path) -> None:
    """Store the figure as it appears in the paper (RGB pixels), for comparison with the extracted plot."""
    plt.imsave(plot_path, original_plot)


def save_extracted_plot(extracted_data: pd.DataFrame, plot_path: str | Path,
                        line_colors: dict[str, np.ndarray] | None = None) -> None:
    """Plot of the extracted series only.
    With line_colors (RGB in 0..1 per series name), each series is drawn in that colour."""
    figure, data_axes = plt.subplots(figsize=(7, 5))
    for series_name, series_data in extracted_data.groupby("series", sort=False):
        color = line_colors.get(str(series_name)) if line_colors else None
        data_axes.plot(series_data.x, series_data.y, lw=0.8, label=series_name, color=color)
    data_axes.legend(fontsize=6, ncol=2)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def data_to_pixels(extracted_data: pd.DataFrame, calibration_data: PixelCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Convert the data values to the column and row of the original image by inverting the calibration."""
    columns = (extracted_data.x.to_numpy() - calibration_data.x_intercept) / calibration_data.x_slope
    rows = (extracted_data.y.to_numpy() - calibration_data.y_intercept) / calibration_data.y_slope
    return columns, rows


def save_overlay_plot(extracted_data: pd.DataFrame, original_plot: np.ndarray, calibration_data: PixelCalibration,
                      plot_path: str | Path) -> None:
    """Draw the original figure with its coloured lines and the extracted data on top as small black points."""
    columns, rows = data_to_pixels(extracted_data, calibration_data)
    height, width = original_plot.shape[:2]
    figure, axes = plt.subplots(figsize=(width / 100, height / 100))
    axes.imshow(original_plot)
    axes.scatter(columns, rows, s=MARKER_SIZE, linewidths=0, color="black")
    axes.set_xlim(-0.5, width - 0.5)
    axes.set_ylim(height - 0.5, -0.5)
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)


def save_figure(figure: FigureResult) -> None:
    """Write the data and the plots of one figure."""
    # Every result file of this figure starts with this path, e.g. results/plot_data/Jia2023_fig14_data.csv
    plot_path = Path("results") / "plot_data" / f"{figure.paper}_fig{figure.fig}"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure.extracted_data.to_csv(f"{plot_path}_data.csv", index=False)
    # The result of the exttraction and the original plot are stored for comparison.
    save_extracted_plot(figure.extracted_data, f"{plot_path}_extracted.png", figure.line_colors)
    save_original_plot(figure.original_plot, f"{plot_path}_original.png")
    save_overlay_plot(figure.extracted_data, figure.original_plot, figure.calibration_data, f"{plot_path}_overlay.png")
    print(f"Wrote the files of {figure.paper} figure {figure.fig} to {plot_path.parent}.")


def save_statistics(figures: list[FigureResult]) -> None:
    """Join the statistics of all figures into one table and write it."""
    statistics = pd.concat([figure.statistics for figure in figures], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    statistics.to_csv(Path("results") / "statistics.csv", index=False, float_format="%.4g")
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.3g}"))
    print("\nDone writing results/statistics.csv.")
