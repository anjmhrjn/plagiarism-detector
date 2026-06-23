"""OCR handler: image bytes → IngestionUnit via Tesseract."""
from __future__ import annotations

import io

from PIL import Image
import pytesseract

from plagdetect.ingest.router import IngestionUnit


def parse_image(data: bytes, source_path: str) -> list[IngestionUnit]:
    img = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(img)
    return [IngestionUnit(
        text=text,
        source_path=source_path,
        file_type="image",
        via_ocr=True,
    )]
