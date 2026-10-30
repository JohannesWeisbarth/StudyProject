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
from extract_vector_plots import extract_huang2020_fig7, extract_montesano2013_fig6
from extraction_common import ExtractionResult, FigureResult
from save_results import save_figure, save_statistics


def open_document(pdf_file: Path) -> pymupdf.Document:
    """Open the PDF, exiting with a readable message if that fails."""
    try:
        return pymupdf.open(pdf_file)
    except (FileNotFoundError, RuntimeError) as error:
        # PyMuPDF raises its own FileNotFoundError (a RuntimeError) or a plain RuntimeError, depending on version.
        sys.exit(f"Could not open PDF '{pdf_file}': {error}")


def find_caption(document: pymupdf.Document, fig: int) -> tuple[pymupdf.Page, pymupdf.Rect]:
    """Locate the caption block of the to-be-digitized figure."""
    figure_pattern: re.Pattern[str] = re.compile(rf"^\s*(Fig\.|Figure)\s*{fig}\s*[.:]", re.I)
    for page in document.pages():
        for block in page.get_text("blocks"):
            if figure_pattern.match(block[4]):
                return page, pymupdf.Rect(block[:4])
    sys.exit(f"Caption for figure {fig} not found in PDF {Path(document.name or '').name}!")


def extract_figure(paper: str, fig: int,
                   extract_plot: Callable[[pymupdf.Document, pymupdf.Page, pymupdf.Rect], ExtractionResult]) -> FigureResult:
    """Extract the data of one figure and compute the statistics of its curves. Nothing is written.
    The PDF file is read from the papers directory, in the directory the script is started in."""
    pdf_file = Path("papers") / f"{paper}.pdf"

    # Open the document and try to find a caption to localize this figure.
    document: pymupdf.Document = open_document(pdf_file)
    page, caption = find_caption(document, fig)
    print(f"Caption for Fig. {fig} on page {(page.number or 0) + 1}")

    print(f"Extracting figure {fig} of {pdf_file.name} with {extract_plot.__name__}.")
    extracted_data, original_plot, calibration_data, line_colors = extract_plot(document, page, caption)

    print(f"Computing the statistics of {pdf_file.name} figure {fig}.\n")
    statistics: pd.DataFrame = compute_statistics(extracted_data, pdf_file.name)
    return FigureResult(paper, fig, extracted_data, original_plot, calibration_data, line_colors, statistics)


def main() -> None:
    # The papers to digitize, in this order: the name of the PDF file without .pdf, the number of the figure in its caption and the
    # function that extracts this figure.
    figures = [extract_figure("Montesano2013_CS", 6, extract_montesano2013_fig6),
               extract_figure("Huang2020_EFM", 7, extract_huang2020_fig7),
               extract_figure("Jia2023", 14, extract_jia2023_fig14)]

    for figure in figures:
        save_figure(figure)
    save_statistics(figures)


if __name__ == "__main__":
    main()
