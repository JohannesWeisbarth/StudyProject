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

from extract_raster_plots import (extract_dolbachin2025_fig7, extract_jia2023_fig14, extract_najd2022_fig12,
                                  extract_najd2022_fig13)
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
    normalized_data = normalize_curves(extracted.extracted_data, metadata, extracted.block_starts)
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
    # - the unit of its x axis: the time in s or h, or the cycles.
    # - whether its y axis is the absolute temperature or delta T.
    # - the test setup of the paper, None where the paper does not state it:
    #   - frequency_hz: the loading frequency, which also converts a cycle axis to s
    #   - fibre_material
    #   - angles_of_attack_deg: the angle between the load and the fibres of every ply, in stacking order
    #   - volume_to_surface_mm: the volume of the gauge section divided by its lateral surface, w·t / (2·(w + t)) for the
    #     width w and thickness t
    #   - uts_mpa: the static strength in MPa, which converts a stress in MPa to a fraction of UTS
    # - the unit of the stress levels, % UTS or σ_max in MPa.
    # Montesano2013 does not state the braid angle, the thickness or the UTS.
    montesano2013 = FigureMetadata("Montesano2013_CS", 6, x_unit="cycles", y_is_absolute=True,
                                   setup={"frequency_hz": 10.0, "fibre_material": "carbon", "angles_of_attack_deg": None,
                                          "volume_to_surface_mm": None, "uts_mpa": None})
    huang2019 = FigureMetadata("Huang2019_IJF", 10, x_unit="cycles", y_is_absolute=False,
                               setup={"frequency_hz": 5.0, "fibre_material": "carbon", "angles_of_attack_deg": [0.0] * 8,
                                      "volume_to_surface_mm": 15.0 * 1.04 / (2 * (15.0 + 1.04)), "uts_mpa": 1487.8})
    # The thickness of Huang2020 is 16 plies of 0.128 mm.
    huang2020 = FigureMetadata("Huang2020_EFM", 7, x_unit="s", y_is_absolute=False,
                               setup={"frequency_hz": 5.0, "fibre_material": "carbon", "angles_of_attack_deg": [45.0, -45.0] * 8,
                                      "volume_to_surface_mm": 25.0 * 2.048 / (2 * (25.0 + 2.048)), "uts_mpa": 147.9})
    jia2023 = FigureMetadata("Jia2023", 14, x_unit="cycles", y_is_absolute=False,
                             setup={"frequency_hz": 5.0, "fibre_material": "carbon",
                                    "angles_of_attack_deg": [0.0, 45.0, 90.0, -45.0, -45.0, 90.0, 45.0, 0.0],
                                    "volume_to_surface_mm": 25.0 * 1.009 / (2 * (25.0 + 1.009)), "uts_mpa": 607.0})
    # Najd2022 and Dolbachin2025 test the same 2/2 twill glass / polyester laminate of 22 plies, 5 mm thick. Every ply
    # has fibres along and across the load. Neither paper states a static strength.
    najd2022_setup = {"frequency_hz": 2.0, "fibre_material": "glass", "angles_of_attack_deg": [0.0, 90.0] * 22,
                      "volume_to_surface_mm": 40.0 * 5.0 / (2 * (40.0 + 5.0)), "uts_mpa": None}
    najd2022 = FigureMetadata("Najd2022", 13, x_unit="h", y_is_absolute=False, setup=najd2022_setup, stress_unit="MPa")
    # A specimen with embedded PVDF transducers, kept apart from the pristine specimen.
    najd2022_smart = FigureMetadata("Najd2022", 12, x_unit="h", y_is_absolute=False, setup=najd2022_setup,
                                    stress_unit="MPa", dataset_name="Najd2022_smart")
    # Four-point bending: the stresses are the outer fibre stresses.
    dolbachin2025 = FigureMetadata("Dolbachin2025", 7, x_unit="s", y_is_absolute=False,
                                   setup={"frequency_hz": 5.0, "fibre_material": "glass", "angles_of_attack_deg": [0.0, 90.0] * 22,
                                          "volume_to_surface_mm": 50.0 * 5.0 / (2 * (50.0 + 5.0)), "uts_mpa": None},
                                   stress_unit="MPa")

    figures = [(montesano2013, extract_figure(montesano2013, extract_montesano2013_fig6)),
               (huang2019, extract_figure(huang2019, extract_huang2019_fig10)),
               (huang2020, extract_figure(huang2020, extract_huang2020_fig7)),
               (jia2023, extract_figure(jia2023, extract_jia2023_fig14)),
               (najd2022, extract_figure(najd2022, extract_najd2022_fig13)),
               (najd2022_smart, extract_figure(najd2022_smart, extract_najd2022_fig12)),
               (dolbachin2025, extract_figure(dolbachin2025, extract_dolbachin2025_fig7))]

    for metadata, result in figures:
        save_figure(metadata, result)
    save_curves(figures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except ExtractionError as error:
        sys.exit(f"Extraction failed: {error}")
