from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Token:
    text: str   # original characters from the source string
    start: int  # inclusive char offset in the original string
    end: int    # exclusive char offset in the original string


def tokenize(text: str) -> list[Token]:
    """Split text into word tokens, each carrying its char offsets in the original string."""
    return [Token(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+", text)]


def canonical(token: Token) -> str:
    """Canonical (NFC + lowercased) form of a token, used for shingle matching."""
    return unicodedata.normalize("NFC", token.text).lower()
