import re
import unicodedata

# Gutenberg boilerplate markers (case-insensitive, tolerate slight variations)
_GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    re.IGNORECASE,
)


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = _strip_gutenberg(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _strip_gutenberg(text: str) -> str:
    start_match = _GUTENBERG_START.search(text)
    end_match = _GUTENBERG_END.search(text)
    if start_match:
        text = text[start_match.end():]
    if end_match:
        end_pos = _GUTENBERG_END.search(text)
        if end_pos:
            text = text[: end_pos.start()]
    return text
