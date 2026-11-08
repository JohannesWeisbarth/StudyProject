#!/usr/bin/env python3
"""Extract data from figures in PDFs: extract the data of every figure, then save the results.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Callable

import pymupdf

from extract_raster_plots import extract_jia2023_fig14
from extract_vector_plots import extract_huang2019_fig10, extract_huang2020_fig7, extract_montesano2013_fig6
from extraction_common import (ExtractedFigure, ExtractionError, ExtractionResult, FigureMetadata, normalize_curves,
                               x_to_seconds)
from save_results import save_curves, save_figure

logger = logging.getLogger(__name__)

# The path to the papers directory
PAPERS_DIRECTORY = Path(__file__).resolve().parent / ".." / "papers"

def open_document(pdf_file: Path) -> pymupdf.Document:
    """Open the paper, raising an ExtractionError if that fails."""
    try:
        return pymupdf.open(pdf_file)
    except (FileNotFoundError, RuntimeError) as error:
        # PyMuPDF raises its own FileNotFoundError (a RuntimeError) or a plain RuntimeError, depending on version.
        raise ExtractionError(f"Could not open PDF '{pdf_file}': {error}") from error


def find_caption(document: pymupdf.Document, fig: int) -> tuple[pymupdf.Page, pymupdf.Rect]:
    """Locate the caption of the to-be-digitized figure."""
    # Look for a text block that starts with "Fig. N" or "Figure N", followed by "." or ":".
    figure_pattern: re.Pattern[str] = re.compile(rf"^\s*(Fig\.|Figure)\s*{fig}\s*[.:]", re.I)
    for page in document.pages():
        for block in page.get_text("blocks"):
            if figure_pattern.match(block[4]):
                return page, pymupdf.Rect(block[:4])
    raise ExtractionError(f"Caption for figure {fig} not found in PDF {Path(document.name or '').name}!")


def extract_figure(metadata: FigureMetadata,
                   # The extraction function for this plot.
                   extraction_function: Callable[[pymupdf.Document, pymupdf.Page, pymupdf.Rect], ExtractedFigure]) -> ExtractionResult:
    """Extract the data of one figure and normalize it: the time in s on the x axis, the temperature rise on the y axis."""
    pdf_file = PAPERS_DIRECTORY / f"{metadata.paper_name}.pdf"

    # Open the document and try to find a caption to localize this figure.
    document: pymupdf.Document = open_document(pdf_file)
    page, caption = find_caption(document, metadata.figure)
    logger.info(f"Caption for Fig. {metadata.figure} found on page {(page.number or 0) + 1}")

    # Every step returns its own result.
    logger.info(f"Extracting figure {metadata.figure} of {metadata.paper_name}.")
    extracted = extraction_function(document, page, caption)
    logger.info(f"Normalizing the axes of {pdf_file.name} figure {metadata.figure}.")
    normalized_data = normalize_curves(extracted.extracted_data, metadata)
    normalized_x_accuracy = float(x_to_seconds(extracted.x_accuracy, metadata))
    # Normalizing y only subtracts the ambient temperature, which does not change the uncertainty.
    normalized_y_accuracy = extracted.y_accuracy

    return ExtractionResult(extracted_data=extracted.extracted_data, original_plot=extracted.original_plot,
                            calibration_data=extracted.calibration_data, line_colors=extracted.line_colors,
                            normalized_data=normalized_data, normalized_x_accuracy=normalized_x_accuracy,
                            normalized_y_accuracy=normalized_y_accuracy)


def main() -> None:
    # The known metadata of every figure to digitize:
    # - the name of the PDF file
    # - the number of the figure
    # - whether its x axis is the time in s (True) or the cycles (False).
    # - whether its y axis is the absolute temperature or delta T.
    # - the loading frequency in Hz.
    # - the test setup of the paper, None where the paper does not state it:
    #   - fibre_material
    #   - angles_of_attack_deg: the angle between the load and the fibres of every ply, in stacking order
    #   - volume_to_surface_mm: the volume of the gauge section divided by its lateral surface, w·t / (2·(w + t)) for the
    #     width w and thickness t
    # Montesano2013 does not state the braid angle or the thickness.
    montesano2013 = FigureMetadata("Montesano2013_CS", 6, x_is_time=False, y_is_absolute=True, frequency_hz=10.0,
                                   setup={"fibre_material": "carbon", "angles_of_attack_deg": None,
                                          "volume_to_surface_mm": None})
    huang2019 = FigureMetadata("Huang2019_IJF", 10, x_is_time=False, y_is_absolute=False, frequency_hz=5.0,
                               setup={"fibre_material": "carbon", "angles_of_attack_deg": [0.0] * 8,
                                      "volume_to_surface_mm": 15.0 * 1.04 / (2 * (15.0 + 1.04))})
    # The thickness of Huang2020 is 16 plies of 0.128 mm.
    huang2020 = FigureMetadata("Huang2020_EFM", 7, x_is_time=True, y_is_absolute=False, frequency_hz=5.0,
                               setup={"fibre_material": "carbon", "angles_of_attack_deg": [45.0, -45.0] * 8,
                                      "volume_to_surface_mm": 25.0 * 2.048 / (2 * (25.0 + 2.048))})
    jia2023 = FigureMetadata("Jia2023", 14, x_is_time=False, y_is_absolute=False, frequency_hz=5.0,
                             setup={"fibre_material": "carbon",
                                    "angles_of_attack_deg": [0.0, 45.0, 90.0, -45.0, -45.0, 90.0, 45.0, 0.0],
                                    "volume_to_surface_mm": 25.0 * 1.009 / (2 * (25.0 + 1.009))})

    figures = [(montesano2013, extract_figure(montesano2013, extract_montesano2013_fig6)),
               (huang2019, extract_figure(huang2019, extract_huang2019_fig10)),
               (huang2020, extract_figure(huang2020, extract_huang2020_fig7)),
               (jia2023, extract_figure(jia2023, extract_jia2023_fig14))]

    for metadata, result in figures:
        save_figure(metadata, result)
    save_curves(figures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except ExtractionError as error:
        sys.exit(f"Extraction failed: {error}")
