"""Extraction of the known vector figures, one function for each. Run through data_extractor.py.

The curves are paths in the PDF, so their points are exact. The names of the curves and, where the figure has
no tick marks, the axis limits are known and hardcoded in the function of the figure.
"""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import pymupdf

from extraction_common import MAX_GAP, ExtractionResult, PixelCalibration, parse_stress_level

# A path with fewer points is not a curve.
MIN_POINTS = 10
# Resolution of the stored original figure.
RENDER_DPI = 200
# The part of the page around the axes box that is stored as the original figure in pt, as (left, top, right, bottom).
# It holds the tick labels, the titles and the labels of the curves.
RENDER_MARGIN = (60, 15, 110, 40)


def find_plot_frame(page: pymupdf.Page, caption_rect: pymupdf.Rect, max_gap: float = MAX_GAP,
                    search_area: pymupdf.Rect | None = None) -> pymupdf.Rect:
    """Find the rectangle of the plot frame just above the caption.
    search_area limits the search to the drawings inside it, for a figure with several panels."""
    page_area: float = page.rect.get_area()
    best_candidate: pymupdf.Rect | None = None
    best_candidate_key: tuple[bool, float] | None = None
    for drawing in page.get_drawings():
        drawing_rect: pymupdf.Rect = drawing["rect"]
        if drawing_rect.y1 > caption_rect.y0 + 2 or caption_rect.y0 - drawing_rect.y1 > max_gap:
            continue
        if search_area is not None and not search_area.contains(drawing_rect):
            continue
        if drawing_rect.width < 100 or drawing_rect.height < 60 or drawing_rect.get_area() > 0.7 * page_area:
            continue
        # A genuine rectangle path beats larger invisible background paths
        # (e.g. white bounding frames drawn by some export tools).
        is_rect: bool = len(drawing["items"]) == 1 and drawing["items"][0][0] in ("re", "qu")
        key: tuple[bool, float] = (is_rect, drawing_rect.get_area())
        if best_candidate_key is None or key > best_candidate_key:
            best_candidate, best_candidate_key = drawing_rect, key
    if best_candidate is None:
        sys.exit("No plot frame found above the caption.")
    return best_candidate


def collect_path_points(drawing: dict) -> np.ndarray | None:
    """Collects the points belonging to a line, None if none found."""
    points = []
    for item in drawing["items"]:
        if item[0] == "l":
            sequence = [item[1], item[2]]
        elif item[0] == "c":  # bezier: keep the end point only
            sequence = [item[1], item[4]]
        else:
            continue
        for point in sequence:
            if not points or (abs(points[-1][0] - point.x) > 1e-6 or abs(points[-1][1] - point.y) > 1e-6):
                points.append((point.x, point.y))
    return np.array(points) if points else None


def collect_curves(page: pymupdf.Page, plot_frame: pymupdf.Rect, tolerance: float = 1.0,
                   join_distance: float = 0.6) -> tuple[list[np.ndarray], list[str]]:
    """Collect the curves in the plot frame in the order in which they are drawn. A curve has to lie inside the frame
    enlarged by the tolerance in pt, which is also what lets a curve through that is drawn a little over the frame.
    A path is joined to the one before it if it has the same style and starts within join_distance in pt of its end.
    Return the points of every curve in page coordinates and its stroke colour as hex colour."""
    plot_area = pymupdf.Rect(plot_frame.x0 - tolerance, plot_frame.y0 - tolerance,
                             plot_frame.x1 + tolerance, plot_frame.y1 + tolerance)
    curve_chains = []
    for drawing in page.get_drawings():
        drawing_rect = drawing["rect"]
        is_frame = abs(drawing_rect.width - plot_frame.width) < 2 and abs(drawing_rect.height - plot_frame.height) < 2
        if not plot_area.contains(drawing_rect) or is_frame:
            continue  # outside the plot, or the frame itself
        path_points = collect_path_points(drawing)
        style_key = (drawing.get("color"), drawing.get("width"))
        # Exporters often split one curve into several consecutive paths:
        # stitch a path onto the previous one if it starts where that ended.
        if path_points is not None and len(path_points) >= 2 and curve_chains \
                and curve_chains[-1]["style_key"] == style_key \
                and np.hypot(*(curve_chains[-1]["points"][-1] - path_points[0])) < join_distance:
            curve_chains[-1]["points"] = np.vstack([curve_chains[-1]["points"], path_points[1:]])
        elif path_points is not None and len(path_points) >= MIN_POINTS:
            curve_chains.append({"style_key": style_key, "points": path_points})
    stroke_colors = ["#" + "".join(f"{round(channel * 255):02x}" for channel in chain["style_key"][0])
                     for chain in curve_chains]
    return [chain["points"] for chain in curve_chains], stroke_colors


def render_figure(page: pymupdf.Page, plot_frame: pymupdf.Rect, caption_rect: pymupdf.Rect) -> tuple[np.ndarray, pymupdf.Rect]:
    """Render the figure region (plot frame plus labels around it, down to the caption) of the page.
    Return the RGB pixels and the rendered page region."""
    left, top, right, bottom = RENDER_MARGIN
    region = pymupdf.Rect(plot_frame.x0 - left, plot_frame.y0 - top, plot_frame.x1 + right,
                          min(plot_frame.y1 + bottom, caption_rect.y0 - 2)) & page.rect
    pixmap = page.get_pixmap(dpi=RENDER_DPI, clip=region, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3), region


def pixel_calibration(to_data: Callable[[np.ndarray], np.ndarray], region_start: float) -> tuple[float, float]:
    """Express a map from page position to data value for the pixels of the rendered figure. Return slope and intercept.
    Pixel i covers page positions region_start + [i, i + 1] / scale and its centre is at i + 0.5."""
    scale = RENDER_DPI / 72.0
    value_at_0, value_at_1 = to_data(np.array([0.0, 1.0]))
    slope = value_at_1 - value_at_0  # the map is linear
    return slope / scale, value_at_0 + slope * (region_start + 0.5 / scale)


def build_result(page: pymupdf.Page, caption_rect: pymupdf.Rect, plot_frame: pymupdf.Rect,
                 curves: list[np.ndarray], curve_names: list[str],
                 x_to_data: Callable[[np.ndarray], np.ndarray], y_to_data: Callable[[np.ndarray], np.ndarray],
                 x_accuracy: float, y_accuracy: float,
                 line_colors: dict[float, np.ndarray] | None = None) -> ExtractionResult:
    """Convert the curves to data values and add the rendered original figure and the accuracy of the values.
    line_colors are the colours of the curves in the paper, if they have different ones."""
    lines: dict[float, np.ndarray] = {}
    for curve_name, curve_points in zip(curve_names, curves):
        x_values, y_values = x_to_data(curve_points[:, 0]), y_to_data(curve_points[:, 1])
        # The path of a curve does not always run from left to right, so sort its points by x.
        order = np.argsort(x_values, kind="stable")
        lines[parse_stress_level(curve_name)] = np.column_stack([x_values[order], y_values[order]])
    original_pixels, region = render_figure(page, plot_frame, caption_rect)
    calibration = PixelCalibration(*pixel_calibration(x_to_data, region.x0), *pixel_calibration(y_to_data, region.y0))
    return ExtractionResult(lines, original_pixels, calibration, line_colors, x_accuracy, y_accuracy)


def linear_map(position_a: float, value_a: float, position_b: float, value_b: float) -> Callable[[np.ndarray], np.ndarray]:
    """Make the map from a position on the page to a data value that runs through two known points."""
    return lambda position: value_a + (position - position_a) / (position_b - position_a) * (value_b - value_a)


def extract_huang2020_fig7(_document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractionResult:
    """Digitize Fig. 7 of Huang2020: the temperature increase in °C over the time in s for 12 stress levels."""
    plot_frame = find_plot_frame(page, caption_rect)

    # The positions of the axis limits on the page in pt: 0 and 1200 s, 0 and 60 °C. They come from a fit through the 31
    # tick marks of each axis. The tick marks are short lines in the path of the plot frame, 2 °C and 40 s apart. The frame
    # lies at the limits within 0.02 pt (0.007 °C and 0.1 s).
    x_to_data = linear_map(118.78463, 0, 396.52744, 1200)
    y_to_data = linear_map(221.04067, 0, 58.68721, 60)

    # Every curve has its own stroke colour. The colours run through a palette in the order of the stress levels.
    curve_names = {
        "#231f20": "30% UTS", "#644587": "35% UTS", "#0090d0": "40% UTS", "#00aded": "45% UTS",
        "#00aec5": "50% UTS", "#00adab": "55% UTS", "#00a967": "60% UTS", "#d2dc3c": "65% UTS",
        "#fde21e": "70% UTS", "#e9922a": "75% UTS", "#ef5f2b": "80% UTS", "#ed2c82": "85% UTS",
    }
    curves, stroke_colors = collect_curves(page, plot_frame)
    print(f"Found {len(curves)} curves")
    if sorted(stroke_colors) != sorted(curve_names):
        sys.exit(f"The stroke colours of the curves are {stroke_colors}, but the figure is hardcoded for {list(curve_names)}.")
    # The accuracy of the values is the largest deviation of a tick mark from the fit: 0.02 pt = 0.007 °C and
    # 0.027 pt = 0.12 s. The coordinates of the curves are exact to 0.001 pt, which is 0.0004 °C.
    # The extracted plot keeps the colours of the curves in the paper.
    line_colors = {parse_stress_level(curve_names[color]): np.array([int(color[index:index + 2], 16) for index in (1, 3, 5)]) / 255.0
                   for color in stroke_colors}
    return build_result(page, caption_rect, plot_frame, curves, [curve_names[color] for color in stroke_colors],
                        x_to_data, y_to_data, x_accuracy=0.12, y_accuracy=0.01, line_colors=line_colors)


def extract_huang2019_fig10(_document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractionResult:
    """Digitize Fig. 10(a) of Huang2019: the temperature rise in °C over the cycles for 13 stress levels, 30 to 90 % UTS.
    The figure has two panels, only panel (a) on the left is digitized."""
    plot_frame = find_plot_frame(page, caption_rect, search_area=pymupdf.Rect(0, 0, page.rect.x1 / 2, caption_rect.y0 + 2))

    # The positions of the axis limits on the page in pt: 0 and 6000 cycles, 0 and 4.5 °C. They come from a fit through the
    # 7 and 10 tick marks, which are short lines of their own. The plot frame lies at the limits within 0.04 pt (1 cycle).
    x_to_data = linear_map(73.55428, 0, 293.75600, 6000)
    y_to_data = linear_map(691.71408, 0, 560.11233, 4.5)

    # The stroke colour of every curve in the order in which they are drawn, and its stress level. Three curves have the same
    # colour, so the colours alone do not give the names. The names agree with the arrows of the legend for 30 to 65 %: the
    # tip of each arrow ends within 0.4 pt of the right end of the curve. 70 to 90 % are labelled at their curves.
    curve_levels = [("#e79651", "80% UTS"), ("#ed1c24", "30% UTS"), ("#ffc20d", "35% UTS"), ("#a6bd84", "45% UTS"),
                    ("#238e85", "50% UTS"), ("#0089c9", "60% UTS"), ("#2375a7", "65% UTS"), ("#4c576f", "70% UTS"),
                    ("#8c6b8b", "75% UTS"), ("#231f20", "85% UTS"), ("#7195be", "90% UTS"), ("#7195be", "55% UTS"),
                    ("#7195be", "40% UTS")]
    # The 90 % curve is drawn in two paths that overlap by 5 cycles and miss each other by 1.5 pt. It also runs up to 8 pt
    # over the top of the frame (to 5.2 °C), the curve is drawn there and is kept.
    curves, stroke_colors = collect_curves(page, plot_frame, tolerance=10.0, join_distance=2.0)
    print(f"Found {len(curves)} curves")
    if stroke_colors != [color for color, _ in curve_levels]:
        sys.exit(f"The stroke colours of the curves are {stroke_colors}, but the figure is hardcoded for "
                 f"{[color for color, _ in curve_levels]}.")
    line_colors = {parse_stress_level(name): np.array([int(color[index:index + 2], 16) for index in (1, 3, 5)]) / 255.0
                   for color, name in curve_levels}
    # The accuracy of the values is the largest deviation of a tick mark from the fit: 0.99 cycles and 0.0011 °C.
    # The coordinates of the curves are exact to 0.001 pt.
    return build_result(page, caption_rect, plot_frame, curves, [name for _, name in curve_levels], x_to_data, y_to_data,
                        x_accuracy=1.0, y_accuracy=0.002, line_colors=line_colors)


def extract_montesano2013_fig6(_document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractionResult:
    """Digitize Fig. 6 of Montesano2013: the temperature in °C over the cycles for 12 stress levels."""
    plot_frame = find_plot_frame(page, caption_rect)

    # The positions of the axis limits on the page in pt: 0 and 7000 cycles, 24 and 40 °C. They come from a fit through all
    # tick marks of the PDF. The plot frame is not at the limits, it lies 0.08 pt lower in y.
    x_to_data = linear_map(159.58091, 0, 395.35708, 7000)
    y_to_data = linear_map(565.97543, 24, 409.53054, 40)

    # All curves are black. They are drawn in the order of their stress levels.
    curve_names = ["30% UTS", "35% UTS", "40% UTS", "45% UTS", "50% UTS", "55% UTS",
                   "60% UTS", "65% UTS", "70% UTS", "75% UTS", "80% UTS", "85% UTS"]
    curves, _ = collect_curves(page, plot_frame)
    print(f"Found {len(curves)} curves")
    if len(curves) != len(curve_names):
        sys.exit(f"Found {len(curves)} curves, but the figure is hardcoded for {len(curve_names)} curves.")
    # The accuracy of the values is the residual of the fit through the tick marks: 0.4 cycles and 0.011 °C.
    # The coordinates of the curves are exact to 0.001 pt, which is below 0.0001 °C.
    return build_result(page, caption_rect, plot_frame, curves, curve_names, x_to_data, y_to_data,
                        x_accuracy=0.4, y_accuracy=0.011)
