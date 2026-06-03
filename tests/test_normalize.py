import unicodedata

import pytest

from plagdetect.ingest.normalize import normalize_text


def test_nfc_equivalence():
    # é as precomposed (U+00E9) vs decomposed (e + combining acute, U+0065 U+0301)
    composed = "é"
    decomposed = "é"
    assert unicodedata.is_normalized("NFC", composed)
    assert not unicodedata.is_normalized("NFC", decomposed)
    assert normalize_text(composed) == normalize_text(decomposed)


def test_lowercase():
    assert normalize_text("Hello WORLD") == "hello world"


def test_whitespace_collapsing():
    assert normalize_text("foo   bar") == "foo bar"
    assert normalize_text("foo\t\tbar") == "foo bar"
    assert normalize_text("foo\nbar\nbaz") == "foo bar baz"
    assert normalize_text("  leading and trailing  ") == "leading and trailing"


def test_mixed_whitespace_and_newlines():
    assert normalize_text("line one\n\n  line two\r\n  line three") == "line one line two line three"


def test_idempotency():
    samples = [
        "Hello   World\n\nFoo",
        "  spaces  ",
        "é", # decomposed é
        "UPPER lower MiXeD",
    ]
    for s in samples:
        once = normalize_text(s)
        twice = normalize_text(once)
        assert once == twice, f"Not idempotent for: {s!r}"


def test_punctuation_preserved():
    # Conservative normalization: punctuation must not be stripped
    assert normalize_text("Hello, world!") == "hello, world!"
    assert normalize_text("It's a test.") == "it's a test."


def test_gutenberg_boilerplate_stripped():
    body = "The actual content of the book goes here."
    text = (
        "Some preamble text.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n\n"
        f"{body}\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n\n"
        "Some postamble text."
    )
    result = normalize_text(text)
    assert "the actual content of the book goes here." in result
    assert "preamble" not in result
    assert "postamble" not in result
    assert "gutenberg" not in result


def test_gutenberg_start_only():
    # If only start marker is present, content after it is kept
    text = "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\nBook content here."
    result = normalize_text(text)
    assert "book content here." in result
    assert "gutenberg" not in result


def test_gutenberg_case_insensitive():
    text = (
        "*** start of the project gutenberg ebook foo ***\n"
        "Content.\n"
        "*** end of the project gutenberg ebook foo ***"
    )
    result = normalize_text(text)
    assert "content." in result
    assert "gutenberg" not in result


def test_plain_text_unchanged_structurally():
    # No markers — content passes through (modulo case/whitespace)
    text = "A simple sentence without any markers."
    assert normalize_text(text) == "a simple sentence without any markers."
