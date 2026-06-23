"""DOCX handler: python-docx paragraph/run extraction."""
from __future__ import annotations

import io

from docx import Document

from plagdetect.ingest.router import IngestionUnit


def parse_docx(data: bytes, source_path: str) -> list[IngestionUnit]:
    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    text = "\n".join(parts)
    return [IngestionUnit(text=text, source_path=source_path, file_type="docx")]
