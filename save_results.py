"""Writing of the results of the extraction: the data, the plots and the statistics.
All results are written to the results directory, in the directory the script is started in.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Plots are never rendered
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extraction_common import FigureResult, PixelCalibration

# Size of the extracted points in the overlay in pt^2. Points are tiny so the original line is still visible.
MARKER_SIZE = 0.3


def save_original_plot(
        original_plot: np.ndarray,
        plot_save_path: str | Path
    ) -> None:
    """Save the original figure, for comparison."""
    plt.imsave(plot_save_path, original_plot)


# Save the extracted points as an image, for visualization.
def save_extracted_plot(
        extracted_data: dict[float, np.ndarray],
        plot_save_path: str | Path,
        line_colors: dict[float, np.ndarray] | None = None
    ) -> None:
    """Save the extracted points as an image, for visualization. Line colors from the original plot are kept, if it had any."""
    figure, data_axes = plt.subplots(figsize=(7, 5))
    for stress_level, line in extracted_data.items():
        color = line_colors.get(stress_level) if line_colors else None
        data_axes.plot(line[:, 0], line[:, 1], lw=0.8, label=f"{stress_level:g}% UTS", color=color)
    data_axes.legend(fontsize=6, ncol=2)
    figure.savefig(plot_save_path, dpi=150)
    plt.close(figure)


def data_to_pixels(line: np.ndarray, calibration_data: PixelCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Convert the data values of a line to the column and row of the original image by inverting the calibration."""
    columns = (line[:, 0] - calibration_data.x_intercept) / calibration_data.x_slope
    rows = (line[:, 1] - calibration_data.y_intercept) / calibration_data.y_slope
    return columns, rows


def save_overlay_plot(extracted_data: dict[float, np.ndarray], original_plot: np.ndarray, calibration_data: PixelCalibration,
                      plot_path: str | Path) -> None:
    """Draw the original figure with its coloured lines and the extracted data on top as small black points."""
    height, width = original_plot.shape[:2]
    figure, axes = plt.subplots(figsize=(width / 100, height / 100))
    axes.imshow(original_plot)
    for line in extracted_data.values():
        columns, rows = data_to_pixels(line, calibration_data)
        axes.scatter(columns, rows, s=MARKER_SIZE, linewidths=0, color="black")
    axes.set_xlim(-0.5, width - 0.5)
    axes.set_ylim(height - 0.5, -0.5)
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)


def lines_to_frame(lines: dict[float, np.ndarray]) -> pd.DataFrame:
    """The lines as one table with the columns stress_level, x, y."""
    return pd.concat([pd.DataFrame({"stress_level": stress_level, "x": line[:, 0], "y": line[:, 1]})
                      for stress_level, line in lines.items()], ignore_index=True)


def save_figure(figure: FigureResult) -> None:
    """Write the data and the plots of one figure."""
    # Every result file of this figure starts with this path, e.g. results/plot_data/Jia2023_fig14_data.csv
    plot_path = Path("results") / "plot_data" / f"{figure.paper}_fig{figure.fig}"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    lines_to_frame(figure.extraction_result.extracted_data).to_csv(f"{plot_path}_data.csv", index=False)
    # The result of the exttraction and the original plot are stored for comparison.
    save_extracted_plot(figure.extraction_result.extracted_data, f"{plot_path}_extracted.png", figure.extraction_result.line_colors)
    save_original_plot(figure.extraction_result.original_plot, f"{plot_path}_original.png")
    save_overlay_plot(figure.extraction_result.extracted_data, figure.extraction_result.original_plot, figure.extraction_result.calibration_data,
                      f"{plot_path}_overlay.png")
    print(f"Wrote the files of {figure.paper} figure {figure.fig} to {plot_path.parent}.")


def save_curves(figures: list[FigureResult]) -> None:
    """Write the normalized curves of all figures to results/extracted_curves.json, nested as {paper: {stress level: {n, x, y}}}.
    For every curve n is the number of the point, x the cycle number and y the temperature rise. The stress level is the
    key as text in % UTS, e.g. "87.5"."""
    curves: dict[str, dict[str, dict[str, list]]] = {}
    for figure in figures:
        paper_curves = curves.setdefault(figure.paper, {})
        for stress_level, line in figure.normalized_data.items():
            paper_curves[f"{stress_level:g}"] = {"n": np.arange(len(line)).tolist(), "x": line[:, 0].tolist(),
                                                 "y": line[:, 1].tolist()}
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / "extracted_curves.json", "w") as curves_file:
        json.dump(curves, curves_file)
    print("Wrote results/extracted_curves.json.")


def save_statistics(figures: list[FigureResult]) -> None:
    """Join the statistics of all figures into one table and write it."""
    statistics = pd.concat([figure.statistics for figure in figures], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    statistics.to_csv(Path("results") / "statistics.csv", index=False, float_format="%.4g")
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.3g}"))
    print("\nDone writing results/statistics.csv.")
