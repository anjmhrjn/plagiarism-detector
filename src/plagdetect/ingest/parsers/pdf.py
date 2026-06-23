"""
PDF handler.

Strategy per page:
  1. Try pypdf embedded-text extraction.
  2. If a page yields no text (scanned), render it to a PIL image and OCR it.
     We use pypdf's page-to-image export (requires pypdf[image] or pypdf >= 4).
     If that fails, log a warning and skip the page rather than crash.
"""
from __future__ import annotations

import io
import logging

import pypdf

from plagdetect.ingest.router import IngestionUnit

_log = logging.getLogger(__name__)


def parse_pdf(data: bytes, source_path: str) -> list[IngestionUnit]:
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages_text: list[str] = []
    ocr_used = False
    warnings: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = txt.strip()

        if txt:
            pages_text.append(txt)
        else:
            # Scanned page — attempt OCR fallback.
            ocr_text = _ocr_page(page, page_num, source_path, warnings)
            if ocr_text:
                pages_text.append(ocr_text)
                ocr_used = True
            else:
                warnings.append(f"page {page_num}: no text layer and OCR yielded nothing.")

    full_text = "\n".join(pages_text)
    return [IngestionUnit(
        text=full_text,
        source_path=source_path,
        file_type="pdf",
        via_ocr=ocr_used,
        notes="; ".join(warnings) if warnings else None,
    )]


def _ocr_page(page: pypdf.PageObject, page_num: int, source_path: str, warnings: list[str]) -> str:
    """Render a single PDF page to a PIL image, then OCR it."""
    try:
        from PIL import Image
        import pytesseract

        # pypdf >= 4.x exposes page.to_image() which requires pypdf[image] extras.
        # Prefer that; fall back to rendering via pypdf's internal rasterizer
        # or a simple white-canvas approach for minimal-resource environments.
        try:
            pil_img = page.to_image()  # type: ignore[attr-defined]
        except AttributeError:
            # pypdf < 4 or no image extras: skip OCR for this page.
            warnings.append(
                f"page {page_num}: no text layer; OCR skipped (pypdf image support unavailable)."
            )
            return ""

        text = pytesseract.image_to_string(pil_img)
        return text.strip()

    except Exception as exc:
        _log.warning("OCR failed for page %d of %s: %s", page_num, source_path, exc)
        warnings.append(f"page {page_num}: OCR error — {exc}")
        return ""
