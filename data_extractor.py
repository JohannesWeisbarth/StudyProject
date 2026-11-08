#!/usr/bin/env python3
"""Extract data from figures in PDFs: extract the data of every figure, then save the results.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import pymupdf

from evaluate_extraction import compute_statistics
from extract_raster_plots import extract_jia2023_fig14
from extract_vector_plots import extract_huang2019_fig10, extract_huang2020_fig7, extract_montesano2013_fig6
from extraction_common import ExtractionResult, FigureResult
from save_results import save_curves, save_figure, save_statistics
from normalize_data import normalize_curves, x_to_cycles


def open_document(pdf_file: Path) -> pymupdf.Document:
    """Open the paper, exiting with a readable message if that fails."""
    try:
        return pymupdf.open(pdf_file)
    except (FileNotFoundError, RuntimeError) as error:
        # PyMuPDF raises its own FileNotFoundError (a RuntimeError) or a plain RuntimeError, depending on version.
        sys.exit(f"Could not open PDF '{pdf_file}': {error}")


def find_caption(document: pymupdf.Document, fig: int) -> tuple[pymupdf.Page, pymupdf.Rect]:
    """Locate the caption of the to-be-digitized figure."""
    # Look for a text block that starts with "Fig. N" or "Figure N", followed by "." or ":".
    figure_pattern: re.Pattern[str] = re.compile(rf"^\s*(Fig\.|Figure)\s*{fig}\s*[.:]", re.I)
    for page in document.pages():
        for block in page.get_text("blocks"):
            if figure_pattern.match(block[4]):
                return page, pymupdf.Rect(block[:4])
    sys.exit(f"Caption for figure {fig} not found in PDF {Path(document.name or '').name}!")


def extract_figure(paper_name: str, fig: int,
                   # The extraction function for this plot.
                   extraction_function: Callable[[pymupdf.Document, pymupdf.Page, pymupdf.Rect], ExtractionResult],
                   # The axes might have to be normalized. All data should be in delta T per cycle,
                   # for the different stress_levels.
                   frequency_hz: float | None, y_is_absolute: bool) -> FigureResult:
    """Extract the data of one figure, convert the axes if required, and compute their statistics.
    frequency_hz is the loading frequency if the x axis of the figure is the time in s, None if it
    already shows the cycles. y_is_absolute is a flag to distinguish whether y axis of the figure
    is the absolute temperature, or only the temperature change, as desired. The ambient temperature that is subtracted from
    an absolute temperature is the lowest first temperature of all curves."""
    pdf_file = Path("papers") / f"{paper_name}.pdf"

    # Open the document and try to find a caption to localize this figure.
    document: pymupdf.Document = open_document(pdf_file)
    page, caption = find_caption(document, fig)
    print(f"Caption for Fig. {fig} on page {(page.number or 0) + 1}")

    print(f"Extracting figure {fig} of {pdf_file.name} with {extraction_function.__name__}.")
    result = extraction_function(document, page, caption)

    print(f"Computing the statistics of {pdf_file.name} figure {fig}.\n")
    normalized_data = normalize_curves(result.extracted_data, frequency_hz, y_is_absolute)
    statistics: pd.DataFrame = compute_statistics(normalized_data, pdf_file.name, x_to_cycles(result.x_accuracy, frequency_hz),
                                                  result.y_accuracy)
    return FigureResult(paper_name, fig, result, normalized_data, statistics)


def main() -> None:
    # The papers to digitize. Required parameters for each paper:
    # - the name of the PDF file
    # - the number of the figure
    # - the function for figure extraction
    # - the loading frequency in Hz if its x axis is the time (None: the x axis shows the cycles)
    # - whether its y axis is the absolute temperature or delta T. Montesano2013 plots one specimen through all stress levels
    #   in a row and gives no ambient temperature, so the lowest first temperature of its curves is used as the ambient one.
    #
    # The frequencies in the papers:
    # - 10 Hz in Montesano2013, however plot uses cycles.
    # - 5 Hz in Huang2019, Huang2020 and Jia2023.
    figures = [extract_figure("Montesano2013_CS", 6, extract_montesano2013_fig6, frequency_hz=None, y_is_absolute=True),
               extract_figure("Huang2019_IJF", 10, extract_huang2019_fig10, frequency_hz=None, y_is_absolute=False),
               extract_figure("Huang2020_EFM", 7, extract_huang2020_fig7, frequency_hz=5.0, y_is_absolute=False),
               extract_figure("Jia2023", 14, extract_jia2023_fig14, frequency_hz=None, y_is_absolute=False)]

    for figure in figures:
        save_figure(figure)
    save_curves(figures)
    save_statistics(figures)


if __name__ == "__main__":
    main()
