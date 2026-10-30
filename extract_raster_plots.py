"""Extraction of the known raster figures, one function for each. Run through data_extractor.py.

The colours and the names of the curves are known and hardcoded in the function of the figure, so the curves
are found by their colours. The axes are calibrated from the tick labels, which are read by OCR:
    pip install rapidocr-onnxruntime
"""
from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

from extraction_common import MAX_GAP, ExtractionResult, PixelCalibration

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


def normalize_minus(text: str) -> str:
    """Convert typographic minus and en dash to ASCII '-', so that float() can parse the label."""
    return text.replace("−", "-").replace("–", "-")


def fit_axis(label_positions: list[float] | np.ndarray, label_values: list[float] | np.ndarray,
             axis: str) -> tuple[float, float]:
    """Linear fit position -> data value from tick labels -> (slope, intercept)."""
    positions, values = np.asarray(label_positions, dtype=float), np.asarray(label_values, dtype=float)
    slope, intercept = np.polyfit(positions, values, 1)
    max_residual = np.max(np.abs(slope * positions + intercept - values)) if len(positions) > 2 else 0.0
    print(f"{axis}-axis: {len(positions)} labels, max residual {max_residual:.3g}")
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
        sys.exit("No embedded image found directly above the caption.")
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
            sys.exit("OCR engine missing. Install it with: pip install rapidocr-onnxruntime")
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
        sys.exit("No axes frame found in the image.")
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
    print(f"Plot frame (px): x {x_left}..{x_right}, y {y_top}..{y_bottom}")
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
        band = (slice(y_bottom + LABEL_GAP, y_bottom + LABEL_GAP + LABEL_BAND), slice(x_left - 20, x_right + 21))
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
        print(f"{axis}-axis: text detection found <2 labels, recognised {len(label_positions)} by projection")
    if len(label_positions) < 2:
        sys.exit(f"Found <2 numeric tick labels for the {axis} axis (OCR read: "
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
            print(f"WARNING: colour {to_hex(color)} covers <{MIN_CURVE_COLUMNS} pixel columns, skipped.")
            continue
        curves.append(track_curve(columns[in_color], rows[in_color]))
        colors.append(color)
    return curves, colors


def to_hex(color: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in color)


def from_hex(hex_colors: list[str]) -> np.ndarray:
    """['#8c6365', '#7b7a7b'] -> RGB array."""
    return np.array([[int(hex_color.lstrip("#")[index:index + 2], 16) for index in (0, 2, 4)]
                     for hex_color in hex_colors], dtype=float)


def extract_jia2023_fig14(document: pymupdf.Document, page: pymupdf.Page, caption_rect: pymupdf.Rect) -> ExtractionResult:
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
    print(f"Figure image: xref {xref}, {pixels.shape[1]}x{pixels.shape[0]} px")

    # The axes are calibrated with the tick labels, which are read by OCR.
    text_boxes = read_text(pixels)
    frame = find_image_frame(pixels)
    x_slope, x_intercept = calibrate_axis(text_boxes, frame, "x", pixels)
    y_slope, y_intercept = calibrate_axis(text_boxes, frame, "y", pixels)

    # Text inside the plot, for example the label of a curve, is not part of a curve.
    mask = curve_ink_mask(pixels, frame, [box for box in text_boxes if not NUMBER_PATTERN.match(box[0])])
    curves, colors = trace_curves(pixels, mask, from_hex(list(curve_names)))
    print(f"Found {len(curves)} of {len(curve_names)} curves")
    names = [curve_names[to_hex(color)] for color in colors]

    rows = []
    for curve_name, curve in zip(names, curves):
        for column, row in curve:
            rows.append((curve_name, x_slope * column + x_intercept, y_slope * row + y_intercept))
    if not rows:
        sys.exit("Nothing extracted.")

    data_frame = pd.DataFrame(rows, columns=["series", "x", "y"])
    line_colors = {curve_name: color / 255.0 for curve_name, color in zip(names, colors)}
    return ExtractionResult(data_frame, pixels, PixelCalibration(x_slope, x_intercept, y_slope, y_intercept), line_colors)
