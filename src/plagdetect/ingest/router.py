"""
Deterministic file-type router.

Accepts raw bytes + a filename, returns a list of IngestionUnit records.
Each unit carries pre-normalization text and provenance metadata.
Unknown / unreadable files produce an error unit instead of crashing.

The only model invoked here is Tesseract OCR (via pytesseract).
No LLM decides file types or orchestrates routing.
"""
from __future__ import annotations

import io
import logging
import mimetypes
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class IngestionUnit:
    text: str                  # raw extracted text, pre-normalization
    source_path: str           # original filename or archive-relative path
    file_type: str             # pdf | docx | txt | image | unknown
    via_ocr: bool = False
    notes: str | None = None   # extraction warnings / errors


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF",         "pdf"),
    (b"PK\x03\x04",  "zip"),
    # DOCX is a zip internally; check after zip so we can peek at content
]

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
_DOCX_EXT   = ".docx"
_PDF_EXT    = ".pdf"
_TXT_EXT    = ".txt"


def _detect_type(data: bytes, filename: str) -> str:
    """Return a coarse file type string from magic bytes and/or filename."""
    ext = Path(filename).suffix.lower()

    # Magic-byte detection first (beats extension spoofing for security).
    for magic, kind in _MAGIC:
        if data.startswith(magic):
            # A .docx is a zip — distinguish by extension.
            if kind == "zip" and ext == _DOCX_EXT:
                return "docx"
            return kind

    # Fallback: extension only.
    if ext == _PDF_EXT:
        return "pdf"
    if ext == _DOCX_EXT:
        return "docx"
    if ext == _TXT_EXT:
        return "txt"
    if ext in _IMAGE_EXTS:
        return "image"

    # Last resort: MIME
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if mime == "application/pdf":
            return "pdf"
        if mime.startswith("image/"):
            return "image"
        if mime == "text/plain":
            return "txt"

    return "unknown"


def ingest(
    data: bytes,
    filename: str,
    *,
    _archive_prefix: str = "",
    _depth: int = 0,
) -> list[IngestionUnit]:
    """
    Route bytes to the appropriate handler and return IngestionUnit list.

    archive_prefix: prepended to source_path for zip-contained files,
                    e.g. "submission.zip/" so nested paths read naturally.
    _depth: current zip nesting level; checked against ZIP_MAX_DEPTH.
    """
    from .parsers.pdf import parse_pdf
    from .parsers.docx import parse_docx
    from .parsers.image import parse_image
    from .parsers.zip_handler import parse_zip

    source_path = _archive_prefix + filename
    file_type = _detect_type(data, filename)

    _DISPATCH: dict[str, Callable[..., list[IngestionUnit]]] = {
        "pdf":   lambda: parse_pdf(data, source_path),
        "docx":  lambda: parse_docx(data, source_path),
        "txt":   lambda: _parse_txt(data, source_path),
        "image": lambda: parse_image(data, source_path),
        "zip":   lambda: parse_zip(
            data, source_path, parent_prefix=source_path + "/", depth=_depth
        ),
    }

    handler = _DISPATCH.get(file_type)
    if handler is None:
        return [IngestionUnit(
            text="",
            source_path=source_path,
            file_type="unknown",
            notes=f"Unsupported file type for '{filename}' — skipped.",
        )]

    try:
        return handler()
    except Exception as exc:
        _log.exception("ingest error for %s", source_path)
        return [IngestionUnit(
            text="",
            source_path=source_path,
            file_type=file_type,
            notes=f"Extraction error: {exc}",
        )]


# ---------------------------------------------------------------------------
# Plain-text handler (inline — too trivial for its own module)
# ---------------------------------------------------------------------------

def _parse_txt(data: bytes, source_path: str) -> list[IngestionUnit]:
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc)
            return [IngestionUnit(text=text, source_path=source_path, file_type="txt")]
        except UnicodeDecodeError:
            continue
    return [IngestionUnit(
        text=data.decode("utf-8", errors="replace"),
        source_path=source_path,
        file_type="txt",
        notes="Non-UTF-8 content; decoded with replacement characters.",
    )]
