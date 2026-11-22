"""Extraction of the known raster figures, one function for each. Run through data_extractor.py.

The colours and the names of the curves are known and hardcoded in the function of the figure, so the curves
are found by their colours. The axes are calibrated from the tick labels, which are read by OCR:
    pip install rapidocr-onnxruntime
"""
from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pymupdf
from PIL import Image

from extraction_common import (MAX_GAP, ExtractionError, ExtractedFigure, PixelCalibration, hex_to_rgb, parse_stress_level,
                           rgb_to_hex)

logger = logging.getLogger(__name__)

# A number, for example a tick label. The minus can be the typographic minus or the en dash.
NUMBER_PATTERN = re.compile(r"^[-−–]?\d+(\.\d+)?$")

OCR_SCALE = 2                # enlarge the image before OCR, small tick labels are read better
DARK_THRESHOLD = 110         # all colour channels below this count as black (frame, ticks, text)
FOREGROUND_DISTANCE = 120    # min RGB distance from white for a pixel to count as curve ink
FRAME_MARGIN = 3             # px ignored next to the frame
TICK_MARGIN = 8              # px next to the left/bottom frame where black tick marks are ignored
MIN_CURVE_COLUMNS = 30       # min pixel columns for a colour cluster to count as a curve
_OCR_ENGINE = None
LABEL_GAP = 6                # px between the frame and the tick label band
LABEL_BAND = 30              # px width/height of the band in which tick labels are searched
LABEL_INK_THRESHOLD = 160    # a pixel with any channel below this counts as label ink
TICK_LABEL_SCALE = 4         # enlargement of a single tick label crop before recognition
COLOR_TOLERANCE = 45         # max RGB distance between a pixel and its curve colour
MAX_TRACK_GAP = 150          # max pixel columns a curve may be hidden behind other curves
MAX_STEP = 3                 # px a curve may move vertically per column beyond its own thickness
GAP_SLOPE = 0.5              # px of extra vertical movement allowed per hidden column
JUMP_COST = 0.2              # track score lost per px of vertical jump
GAP_COST = 0.1               # track score lost per hidden column; low, so short visible pieces of a mostly hidden curve still connect
# Dolbachinn and Najd show figures of a whole step test: one red temperature line over the time, the blocks are marked by grey load bars.
GREY_SATURATION = 20         # max difference between the colour channels of a grey pixel
LIGHT_THRESHOLD = 235        # all colour channels above this count as white
MIN_LOAD_PIXELS = 25         # min grey pixels in a pixel column of a load bar
MIN_BLOCK_COLUMNS = 20       # min pixel columns of a block
TICK_LENGTH = 6              # px right of the temperature axis in which its tick marks are searched
LINE_JUMP = 40               # max px a line moves vertically per pixel column where it is hidden


def normalize_minus(text: str) -> str:
    """Convert typographic minus and en dash to ASCII '-', so that float() can parse the label."""
    return text.replace("−", "-").replace("–", "-")


def fit_axis(label_positions: list[float] | np.ndarray, label_values: list[float] | np.ndarray,
             axis: str) -> tuple[float, float]:
    """Linear fit position -> data value from tick labels -> (slope, intercept)."""
    positions, values = np.asarray(label_positions, dtype=float), np.asarray(label_values, dtype=float)
    slope, intercept = np.polyfit(positions, values, 1)
    max_residual = np.max(np.abs(slope * positions + intercept - values)) if len(positions) > 2 else 0.0
    logger.info(f"{axis}-axis: {len(positions)} labels, max residual {max_residual:.3g}")
    return slope, intercept


def find_figure_image(page: pymupdf.Page, caption_rect: pymupdf.Rect) -> tuple[int, pymupdf.Rect]:
    """Embedded image directly above the caption (overlapping its columns) -> (xref, image rect)."""
    best_gap, best_xref, best_rect = None, None, None
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        for image_rect in page.get_image_rects(xref):
            horizontal_overlap = min(image_rect.x1, caption_rect.x1) - max(image_rect.x0, caption_rect.x0)
            gap = caption_rect.y0 - image_rect.y1
            if horizontal_overlap <= 0 or gap < -2 or gap > MAX_GAP:
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_xref, best_rect = gap, xref, image_rect
    if best_xref is None or best_rect is None:
        raise ExtractionError("No embedded image found directly above the caption.")
    return best_xref, best_rect


def load_image_pixels(document: pymupdf.Document, xref: int) -> np.ndarray:
    """RGB pixels of an embedded image, shape (height, width, 3)."""
    pixmap = pymupdf.Pixmap(document, xref)
    if pixmap.colorspace is None or pixmap.colorspace.n != 3 or pixmap.alpha:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    return pixels[:, :, :3].copy()


def get_ocr_engine():
    """The OCR engine, created on first use."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ExtractionError("OCR engine missing. Install it with: pip install rapidocr-onnxruntime")
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def read_text(pixels: np.ndarray) -> list[tuple[str, float, float, float, float]]:
    """OCR the image -> list of (text, x0, y0, x1, y1) in image pixels."""
    height, width = pixels.shape[:2]
    enlarged = np.array(Image.fromarray(pixels).resize((width * OCR_SCALE, height * OCR_SCALE), Image.Resampling.LANCZOS))
    result, _ = get_ocr_engine()(enlarged)
    text_boxes = []
    for corners, text, _ in result or []:
        corners = np.array(corners, dtype=float) / OCR_SCALE
        text_boxes.append((text.strip(), corners[:, 0].min(), corners[:, 1].min(),
                           corners[:, 0].max(), corners[:, 1].max()))
    return text_boxes


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, end) indices of the longest run of True values in a 1D mask."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0] - 1
    if len(starts) == 0:
        raise ExtractionError("No axes frame found in the image.")
    longest = np.argmax(ends - starts)
    return int(starts[longest]), int(ends[longest])


def find_image_frame(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """Axes frame from the long black lines -> (x_left, x_right, y_top, y_bottom)."""
    height, width = pixels.shape[:2]
    dark = pixels.max(axis=2) < DARK_THRESHOLD
    row_counts, column_counts = dark.sum(axis=1), dark.sum(axis=0)
    y_bottom = height // 2 + int(np.argmax(row_counts[height // 2:]))
    x_left = int(np.argmax(column_counts[:width // 2]))
    _, x_right = longest_run(dark[y_bottom])
    y_top, _ = longest_run(dark[:, x_left])
    logger.info(f"Plot frame (px): x {x_left}..{x_right}, y {y_top}..{y_bottom}")
    return x_left, x_right, y_top, y_bottom


def dark_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """(start, end) of runs of True in a 1D mask; runs closer than max_gap are merged."""
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    runs = [[indices[0], indices[0]]]
    for index in indices[1:]:
        if index - runs[-1][1] <= max_gap:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [(int(start), int(end)) for start, end in runs]


def read_labels_by_projection(pixels: np.ndarray, frame: tuple[int, int, int, int], axis: str) -> tuple[list[float], list[float]]:
    """Fallback when text detection misses the (often single-digit) tick labels:
    locate each label glyph group next to the axis by projecting dark pixels,
    then recognise only that small crop."""
    x_left, x_right, y_top, y_bottom = frame
    ink = pixels.min(axis=2) < LABEL_INK_THRESHOLD
    if axis == "x":
        # Below the axis, the first rows with ink are the axis and its tick marks, the next rows are the labels.
        row_runs = dark_runs(ink[y_bottom:, x_left:x_right + 1].any(axis=1), max_gap=1)
        if len(row_runs) < 2:
            return [], []
        band = (slice(y_bottom + row_runs[1][0] - 3, y_bottom + row_runs[1][1] + 4), slice(x_left - 20, x_right + 21))
        runs = dark_runs(ink[band].any(axis=0), max_gap=8)
        offset = x_left - 20
    else:
        band = (slice(max(y_top - 12, 0), y_bottom + 13), slice(max(x_left - LABEL_GAP - LABEL_BAND, 0), x_left - LABEL_GAP))
        runs = dark_runs(ink[band].any(axis=1), max_gap=3)
        offset = max(y_top - 12, 0)

    crops, centers = [], []
    for start, end in runs:
        if end - start < 4:
            continue  # speck, not a glyph group
        if axis == "x":
            crop = pixels[band[0], offset + start - 3:offset + end + 4]
        else:
            crop = pixels[offset + start - 3:offset + end + 4, band[1]]
        enlarged = np.array(Image.fromarray(crop).resize(
            (crop.shape[1] * TICK_LABEL_SCALE, crop.shape[0] * TICK_LABEL_SCALE), Image.Resampling.LANCZOS))
        crops.append(np.pad(enlarged, ((24, 24), (24, 24), (0, 0)), constant_values=255))
        centers.append(offset + (start + end) / 2)
    if not crops:
        return [], []

    # The recogniser is called directly: the OCR wrapper returns nothing for lone digits
    # (it needs the text detector, which does not fire on them) and drops low-confidence results.
    recognised, _ = get_ocr_engine().text_recognizer(crops)
    positions, values = [], []
    for center, (text, _) in zip(centers, recognised):
        text = normalize_minus(unicodedata.normalize("NFKC", text).strip())
        if NUMBER_PATTERN.match(text):
            positions.append(center); values.append(float(text))
    return positions, values


def calibrate_axis(text_boxes: list, frame: tuple[int, int, int, int], axis: str,
                   pixels: np.ndarray) -> tuple[float, float]:
    """Fit pixel position -> data value from numeric tick labels -> (slope, intercept)."""
    x_left, x_right, y_top, y_bottom = frame
    label_positions, label_values = [], []
    for text, x0, y0, x1, y1 in text_boxes:
        text = normalize_minus(text)
        if not NUMBER_PATTERN.match(text):
            continue
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        if axis == "x" and y_bottom < center_y <= y_bottom + 40 and x_left - 15 <= center_x <= x_right + 15:
            label_positions.append(center_x); label_values.append(float(text))
        if axis == "y" and x_left - 60 <= center_x < x_left and y_top - 10 <= center_y <= y_bottom + 10:
            label_positions.append(center_y); label_values.append(float(text))
    if len(label_positions) < 2:
        label_positions, label_values = read_labels_by_projection(pixels, frame, axis)
        logger.info(f"{axis}-axis: text detection found <2 labels, recognised {len(label_positions)} by projection")
    if len(label_positions) < 2:
        raise ExtractionError(f"Found <2 numeric tick labels for the {axis} axis (OCR read: "
                              f"{[box[0] for box in text_boxes]}).")
    return fit_axis(label_positions, label_values, axis)


def curve_ink_mask(pixels: np.ndarray, frame: tuple[int, int, int, int], text_boxes: list) -> np.ndarray:
    """Boolean mask over the whole image: True for pixels that belong to a curve."""
    x_left, x_right, y_top, y_bottom = frame
    distance_from_white = np.linalg.norm(255.0 - pixels.astype(float), axis=2)
    mask = distance_from_white > FOREGROUND_DISTANCE
    # Keep only the inside of the frame.
    inside = np.zeros_like(mask)
    inside[y_top + FRAME_MARGIN:y_bottom - FRAME_MARGIN, x_left + FRAME_MARGIN:x_right - FRAME_MARGIN] = True
    mask &= inside
    # Black tick marks stick into the plot along the left and bottom frame.
    is_black = pixels.max(axis=2) < DARK_THRESHOLD
    mask[:, :x_left + TICK_MARGIN + FRAME_MARGIN] &= ~is_black[:, :x_left + TICK_MARGIN + FRAME_MARGIN]
    mask[y_bottom - TICK_MARGIN - FRAME_MARGIN:, :] &= ~is_black[y_bottom - TICK_MARGIN - FRAME_MARGIN:, :]
    # Text inside the plot (curve labels).
    for _, x0, y0, x1, y1 in text_boxes:
        mask[max(int(y0) - 2, 0):int(y1) + 3, max(int(x0) - 2, 0):int(x1) + 3] = False
    return mask


def nearest_color(colors: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the closest palette colour for each colour -> (indices, RGB distances)."""
    distances = np.linalg.norm(colors[:, None, :].astype(float) - palette[None, :, :], axis=2)
    indices = np.argmin(distances, axis=1)
    return indices, distances[np.arange(len(colors)), indices]


def track_curve(columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Longest consistent path through the pixels of one colour -> (column, row) array.

    Per pixel column the pixels form vertical runs: the curve itself, plus stray pixels of
    other curves with a similar colour. The path picks at most one run per column and may
    skip columns where the curve is hidden behind others, paying for vertical jumps and
    skipped columns (dynamic programming over the runs). Hidden stretches stay empty."""
    order = np.lexsort((rows, columns))
    sorted_rows, sorted_columns = rows[order], columns[order]
    run_starts = np.concatenate(([0], np.where((np.diff(sorted_columns) != 0) | (np.diff(sorted_rows) > 2))[0] + 1))
    run_sizes = np.diff(np.concatenate((run_starts, [len(sorted_rows)])))
    run_columns = sorted_columns[run_starts]
    run_rows = np.add.reduceat(sorted_rows.astype(float), run_starts) / run_sizes

    score, previous_run = np.ones(len(run_starts)), np.full(len(run_starts), -1)
    for run in range(len(run_starts)):
        first = np.searchsorted(run_columns, run_columns[run] - MAX_TRACK_GAP)
        last = np.searchsorted(run_columns, run_columns[run])  # runs in earlier columns only
        if first == last:
            continue
        column_steps = run_columns[run] - run_columns[first:last]
        jumps = np.abs(run_rows[run] - run_rows[first:last])
        reachable = jumps <= (run_sizes[run] + run_sizes[first:last]) / 2 + MAX_STEP + GAP_SLOPE * (column_steps - 1)
        candidate_scores = np.where(reachable, score[first:last] - JUMP_COST * jumps - GAP_COST * (column_steps - 1), -np.inf)
        best = int(np.argmax(candidate_scores))
        if candidate_scores[best] > 0:
            score[run] += candidate_scores[best]
            previous_run[run] = first + best

    path = [int(np.argmax(score))]
    while previous_run[path[-1]] >= 0:
        path.append(int(previous_run[path[-1]]))
    path = path[::-1]
    return np.column_stack([run_columns[path], run_rows[path]])


def trace_curves(pixels: np.ndarray, mask: np.ndarray, curve_colors: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Trace one curve for every curve colour. Return the curves as (column, row) arrays and their colours.

    A pixel belongs to the closest curve colour, if it is close enough to it. Pixels where curves overlap
    are ignored. If a curve is hidden behind another curve, it has a gap there."""
    rows, columns = np.where(mask)
    assignment, distances = nearest_color(pixels[rows, columns], curve_colors)
    curves, colors = [], []
    for color_index, color in enumerate(curve_colors):
        in_color = (assignment == color_index) & (distances < COLOR_TOLERANCE)
        if len(np.unique(columns[in_color])) < MIN_CURVE_COLUMNS:
            logger.warning(f"colour {rgb_to_hex(color)} covers <{MIN_CURVE_COLUMNS} pixel columns, skipped.")
            continue
        curves.append(track_curve(columns[in_color], rows[in_color]))
        colors.append(color)
    return curves, colors


def extract_jia2023_fig14(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractedFigure:
    """Digitize Fig. 14 of Jia2023: the temperature increase in °C over the cycles for 20 stress levels.
    The figure is an image. The curves are found by their colours."""
    # The colours of the curves in the image and their names, from top to bottom at the right edge of the plot.
    # The arrows of the labels 70 %, 72.5 % and 75 % end within a few pixels of each other. The names of these three
    # curves were checked in the figure: magenta is 75 %, green is 72.5 % and the ochre curve is 70 %.
    curve_names = {
        "#8c6365": "87.5% UTS", "#7b7b7c": "85% UTS", "#8fa73e": "82.5% UTS", "#394084": "77.5% UTS",
        "#47babc": "80% UTS", "#a12a6e": "75% UTS", "#25b546": "72.5% UTS", "#8a6227": "70% UTS",
        "#2b8b8a": "67.5% UTS", "#6c066b": "57.5% UTS", "#355c8b": "65% UTS", "#7c7a16": "62.5% UTS",
        "#6c0304": "60% UTS", "#6f08db": "55% UTS", "#010165": "52.5% UTS", "#097109": "50% UTS",
        "#dd13dc": "45% UTS", "#0302c9": "40% UTS", "#d40a0a": "35% UTS", "#000000": "30% UTS",
    }
    xref, _ = find_figure_image(page, caption_rect)
    pixels = load_image_pixels(document, xref)
    logger.info(f"Figure image: xref {xref}, {pixels.shape[1]}x{pixels.shape[0]} px")

    # The axes are calibrated with the tick labels, which are read by OCR.
    text_boxes = read_text(pixels)
    frame = find_image_frame(pixels)
    x_slope, x_intercept = calibrate_axis(text_boxes, frame, "x", pixels)
    y_slope, y_intercept = calibrate_axis(text_boxes, frame, "y", pixels)

    # Text inside the plot, for example the label of a curve, is not part of a curve.
    mask = curve_ink_mask(pixels, frame, [box for box in text_boxes if not NUMBER_PATTERN.match(box[0])])
    curves, colors = trace_curves(pixels, mask, np.array([hex_to_rgb(color) for color in curve_names]))
    logger.info(f"Found {len(curves)} of {len(curve_names)} curves")
    names = [curve_names[rgb_to_hex(color)] for color in colors]

    lines: dict[float, np.ndarray] = {
        parse_stress_level(curve_name): np.column_stack([x_slope * curve[:, 0] + x_intercept, y_slope * curve[:, 1] + y_intercept])
        for curve_name, curve in zip(names, curves)}
    if not lines:
        raise ExtractionError("Nothing extracted.")

    line_colors = {parse_stress_level(curve_name): color / 255.0 for curve_name, color in zip(names, colors)}
    # The accuracy of the values is 2.2 px: the calibration is good to 1 px, and the centre of a line that is about
    # 4 px thick is good to 2 px. Its size in the data follows from the calibration. The plateaus of this figure differ
    # from the stabilised temperatures in Fig. 16 of the paper by 0.11 °C (RMS), which also includes how the paper
    # defines them.
    pixel_accuracy = 2.2
    return ExtractedFigure(lines, pixels, PixelCalibration(x_slope, x_intercept, y_slope, y_intercept), line_colors,
                            x_accuracy=pixel_accuracy * abs(x_slope), y_accuracy=pixel_accuracy * abs(y_slope))


def is_red(pixels: np.ndarray) -> np.ndarray:
    """Boolean mask over the image: True for red pixels."""
    channels = pixels.astype(int)
    return (channels[:, :, 0] > 200) & (channels[:, :, 1] < 100) & (channels[:, :, 2] < 100)


def find_step_test_frame(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """Axes frame of a step test figure -> (x_left, x_right, y_top, y_bottom). The black time axis at the bottom gives
    the left and right end, the red temperature axis at its right end gives the top."""
    dark = pixels.max(axis=2) < DARK_THRESHOLD
    height = dark.shape[0]
    y_bottom = height // 2 + int(np.argmax(dark.sum(axis=1)[height // 2:]))
    x_left, x_right = longest_run(dark[y_bottom])
    y_top, _ = longest_run(is_red(pixels)[:, x_right])
    logger.info(f"Plot frame (px): x {x_left}..{x_right}, y {y_top}..{y_bottom}")
    return x_left, x_right, y_top, y_bottom


def calibrate_temperature_axis(red: np.ndarray, frame: tuple[int, int, int, int],
                               tick_values: list[float]) -> tuple[float, float]:
    """Fit pixel row -> temperature from the tick marks on the right of the red temperature axis -> (slope, intercept).
    The tick values, from bottom to top, are read from the figure. OCR misses most of the single digit labels."""
    _, x_right, y_top, y_bottom = frame
    is_tick = red[y_top - 2:y_bottom + 3, x_right + 2:x_right + 2 + TICK_LENGTH].sum(axis=1) >= TICK_LENGTH - 1
    tick_rows = [y_top - 2 + (start + end) / 2 for start, end in dark_runs(is_tick, max_gap=1)]
    if len(tick_rows) != len(tick_values):
        raise ExtractionError(f"Found {len(tick_rows)} tick marks on the temperature axis, expected {len(tick_values)}.")
    # The rows grow downwards, the values upwards.
    return fit_axis(tick_rows[::-1], tick_values, "temperature")


def trace_single_line(mask: np.ndarray) -> np.ndarray:
    """The only line of a colour, one point per pixel column -> (column, row) array.

    The point is the centre of the run of pixels in the column, so a steep part of the line gives the middle of its
    vertical run. The line continues with the run closest to the previous run. A run that is further away than
    LINE_JUMP px per column is not the line, for example the legend in a column where other lines hide the line. Columns
    without the line are skipped."""
    points: list[tuple[int, float]] = []
    previous_column, previous_start, previous_end = -1, 0, 0
    for column in np.where(mask.any(axis=0))[0]:
        runs = np.array(dark_runs(mask[:, column], max_gap=2))
        if points:
            # The vertical distance between the runs, 0 if they overlap.
            distances = np.maximum(0, np.maximum(runs[:, 0] - previous_end, previous_start - runs[:, 1]))
            closest = int(np.argmin(distances))
            if distances[closest] > LINE_JUMP * (column - previous_column):
                continue
        else:
            closest = 0
        previous_column, (previous_start, previous_end) = column, runs[closest]
        points.append((int(column), (previous_start + previous_end) / 2))
    return np.array(points)


def find_load_blocks(pixels: np.ndarray, frame: tuple[int, int, int, int], text_boxes: list,
                     line_gap: int) -> list[tuple[int, int]]:
    """First and last pixel column of every block, from the grey load bars -> list of (start, end), in time order.

    The columns with grey load lines that are at most line_gap columns apart form a block. The thin load ramps before
    and after a block are cut off: a column of the block holds at least half the grey of the fullest column."""
    x_left, x_right, y_top, y_bottom = frame
    channels = pixels.astype(int)
    grey = ((channels.max(axis=2) - channels.min(axis=2) < GREY_SATURATION) & (channels.max(axis=2) > DARK_THRESHOLD)
            & (channels.max(axis=2) < LIGHT_THRESHOLD))
    # The edges of text are grey, too.
    for _, x0, y0, x1, y1 in text_boxes:
        grey[max(int(y0) - 2, 0):int(y1) + 3, max(int(x0) - 2, 0):int(x1) + 3] = False
    first_column = x_left + FRAME_MARGIN
    counts = grey[y_top:y_bottom - FRAME_MARGIN, first_column:x_right - 2 * FRAME_MARGIN].sum(axis=0)
    blocks = []
    for start, end in dark_runs(counts >= MIN_LOAD_PIXELS, line_gap):
        full = np.where(counts[start:end + 1] >= counts[start:end + 1].max() / 2)[0]
        if full[-1] - full[0] >= MIN_BLOCK_COLUMNS:
            blocks.append((first_column + start + int(full[0]), first_column + start + int(full[-1])))
    return blocks


def extract_step_test(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect,
                      tick_values: list[float], stresses: list[float], line_gap: int,
                      block_length: float) -> ExtractedFigure:
    """Digitize a figure that shows a whole step test: the temperature as one red line over the time, the load as grey
    bars, one per block. The figure is an image. The red line is cut into one curve per block, its x values stay the
    time of the figure, the start of every block is returned with it.

    tick_values: the values of the ticks on the temperature axis, from bottom to top.
    stresses: the σ_max of the blocks in time order, from the text of the paper.
    line_gap: the largest gap in pixel columns between the grey load lines of one block.
    block_length: the time of the cycles of one block in the unit of the time axis. The grey bars end a few pixels after
    the cycles, when the line already drops, so a block ends at the end of its bar or after block_length."""
    xref, _ = find_figure_image(page, caption_rect)
    pixels = load_image_pixels(document, xref)
    logger.info(f"Figure image: xref {xref}, {pixels.shape[1]}x{pixels.shape[0]} px")

    text_boxes = read_text(pixels)
    frame = find_step_test_frame(pixels)
    # Text detection misses most of the time labels, and two labels next to each other would give a poor fit. Every
    # label is recognised on its own instead.
    label_positions, label_values = read_labels_by_projection(pixels, frame, "x")
    if len(label_positions) < 2:
        raise ExtractionError(f"Recognised {len(label_positions)} time labels, at least 2 are needed.")
    x_slope, x_intercept = fit_axis(label_positions, label_values, "x")
    red = is_red(pixels)
    y_slope, y_intercept = calibrate_temperature_axis(red, frame, tick_values)

    blocks = find_load_blocks(pixels, frame, [box for box in text_boxes if not NUMBER_PATTERN.match(box[0])], line_gap)
    if len(blocks) != len(stresses):
        raise ExtractionError(f"Found {len(blocks)} load blocks, expected {len(stresses)}: {blocks}")

    # The red line inside the frame, without the red temperature axis on its right.
    x_left, x_right, y_top, y_bottom = frame
    inside = np.zeros_like(red)
    inside[y_top:y_bottom - FRAME_MARGIN, x_left + FRAME_MARGIN:x_right - 2 * FRAME_MARGIN] = True
    curve = trace_single_line(red & inside)

    lines: dict[float, np.ndarray] = {}
    block_starts: dict[float, float] = {}
    for stress, (start, end) in zip(stresses, blocks):
        end = min(end, start + block_length / x_slope)
        in_block = (curve[:, 0] >= start) & (curve[:, 0] <= end)
        lines[stress] = np.column_stack([x_slope * curve[in_block, 0] + x_intercept, y_slope * curve[in_block, 1] + y_intercept])
        block_starts[stress] = x_slope * start + x_intercept
    # The accuracy is 2 px: the ticks give the calibration to 1 px, the centre of a line that is 2 to 4 px thick is good to
    # 1 to 2 px, and the start of a block is good to 1 px.
    pixel_accuracy = 2.0
    return ExtractedFigure(lines, pixels, PixelCalibration(x_slope, x_intercept, y_slope, y_intercept), None,
                           x_accuracy=pixel_accuracy * abs(x_slope), y_accuracy=pixel_accuracy * abs(y_slope),
                           block_starts=block_starts)


def extract_najd2022_fig13(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractedFigure:
    """Digitize Fig. 13 of Najd2022: the temperature rise in °C over the time in h of a whole step test on a pristine
    specimen. σ_max starts at 50 MPa and grows by 10 MPa per block. A block is 5000 cycles at 2 Hz. The specimen fails in
    the 13th block."""
    return extract_step_test(document, page, caption_rect, tick_values=list(range(11)),
                             stresses=[50.0 + 10.0 * block for block in range(13)], line_gap=10, block_length=2500.0 / 3600.0)


def extract_najd2022_fig12(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractedFigure:
    """Digitize Fig. 12 of Najd2022: like Fig. 13, on a smart specimen with embedded PVDF transducers. The blue
    capacitance line is not extracted."""
    return extract_step_test(document, page, caption_rect, tick_values=list(range(9)),
                             stresses=[50.0 + 10.0 * block for block in range(13)], line_gap=10, block_length=2500.0 / 3600.0)


def extract_dolbachin2025_fig7(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractedFigure:
    """Digitize Fig. 7 of Dolbachin2025: the temperature rise in °C over the time in s of a whole step test in four-point
    bending. F_max starts at 187.5 N and grows by 187.5 N per block, the outer fibre stress σ = F L / (b h²) with
    L = 150 mm, b = 50 mm, h = 5 mm is 22.5 MPa per block. A block is 5000 cycles at 5 Hz. The figure shows the first 10
    blocks, the specimen fails in the 11th. The blue capacitance line is not extracted."""
    return extract_step_test(document, page, caption_rect, tick_values=list(range(6)),
                             stresses=[22.5 * block for block in range(1, 11)], line_gap=6, block_length=1000.0)
