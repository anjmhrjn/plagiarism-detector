import json
import tempfile
from pathlib import Path

from plagdetect.corpus import load_corpus
from plagdetect.ingest.normalize import normalize_text

FAKE_RECORDS = [
    {
        "id": "gutenberg:1",
        "source": "gutenberg",
        "title": "A Tale of Two Cities",
        "url": "https://example.com/1.txt",
        "text": "It was the best of times,\nit was the worst of times.",
        "canonical": normalize_text("It was the best of times,\nit was the worst of times."),
    },
    {
        "id": "gutenberg:2",
        "source": "gutenberg",
        "title": "Pride and Prejudice",
        "url": "https://example.com/2.txt",
        "text": "It is a truth universally acknowledged.",
        "canonical": normalize_text("It is a truth universally acknowledged."),
    },
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _write_jsonl(tmp_path, FAKE_RECORDS)
        loaded = load_corpus(tmp_path)
        assert loaded == FAKE_RECORDS
    finally:
        tmp_path.unlink(missing_ok=True)


def test_record_fields_present():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _write_jsonl(tmp_path, FAKE_RECORDS)
        loaded = load_corpus(tmp_path)
        for rec in loaded:
            for field in ("id", "source", "title", "url", "text", "canonical"):
                assert field in rec, f"Missing field {field!r}"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_canonical_is_normalized():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    try:
        _write_jsonl(tmp_path, FAKE_RECORDS)
        loaded = load_corpus(tmp_path)
        for rec in loaded:
            assert rec["canonical"] == normalize_text(rec["text"])
    finally:
        tmp_path.unlink(missing_ok=True)


def test_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)

    try:
        loaded = load_corpus(tmp_path)
        assert loaded == []
    finally:
        tmp_path.unlink(missing_ok=True)


def test_skips_blank_lines():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(json.dumps(FAKE_RECORDS[0]) + "\n")
        tmp.write("\n")
        tmp.write(json.dumps(FAKE_RECORDS[1]) + "\n")

    try:
        loaded = load_corpus(tmp_path)
        assert len(loaded) == 2
    finally:
        tmp_path.unlink(missing_ok=True)
