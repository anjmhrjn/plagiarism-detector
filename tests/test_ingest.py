"""
Tests for the ingestion layer.

Covers:
  - Plain-text round-trip through the router.
  - Security guards: zip-bomb, Zip Slip, over-depth nesting.
  - Benign nested zip: succeeds with correct archive-relative source_path.
  - DOCX extraction.
  - /check-file endpoint: TXT upload returns detection results with source_path.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from plagdetect.ingest import IngestionUnit, ingest
from plagdetect.ingest.normalize import normalize_text
from plagdetect.ingest.parsers.zip_handler import (
    ZIP_MAX_DEPTH,
    ZIP_MAX_RATIO,
    ZIP_MAX_TOTAL_BYTES,
    ZIP_MAX_ENTRY_BYTES,
)

_FIXTURE_DIR = Path(__file__).parent.parent / "eval" / "fixtures" / "ingest"

# ---------------------------------------------------------------------------
# Shared corpus for endpoint tests
# ---------------------------------------------------------------------------

_RAW = {
    "doc1": ("g:1", "Fox and Dog", "the quick brown fox jumps over the lazy dog near the river bank"),
}
_TEST_CORPUS = [
    {"id": id_, "title": title, "canonical": normalize_text(raw)}
    for id_, title, raw in _RAW.values()
]
_TEST_INDEX = {doc["id"]: doc["canonical"] for doc in _TEST_CORPUS}


@pytest.fixture(autouse=True)
def _inject_corpus(monkeypatch):
    monkeypatch.setattr(main_module, "_CORPUS", _TEST_CORPUS)
    monkeypatch.setattr(main_module, "_CORPUS_INDEX", _TEST_INDEX)
    monkeypatch.setattr(main_module, "_SEMANTIC_INDEX", None)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Router: plain text
# ---------------------------------------------------------------------------

class TestRouterTxt:
    def test_txt_produces_one_unit(self):
        data = b"Hello world"
        units = ingest(data, "hello.txt")
        assert len(units) == 1
        assert units[0].file_type == "txt"
        assert units[0].via_ocr is False
        assert units[0].text == "Hello world"
        assert units[0].source_path == "hello.txt"

    def test_unknown_type_produces_error_unit(self):
        units = ingest(b"\x00\x01\x02", "mystery.xyz")
        assert len(units) == 1
        assert units[0].file_type == "unknown"
        assert units[0].text == ""
        assert units[0].notes is not None


# ---------------------------------------------------------------------------
# Security: zip-bomb
# ---------------------------------------------------------------------------

class TestZipBomb:
    def test_bomb_zip_is_rejected(self):
        """bomb.zip has a single entry with ratio >> ZIP_MAX_RATIO — must be rejected."""
        data = (_FIXTURE_DIR / "bomb.zip").read_bytes()
        units = ingest(data, "bomb.zip")

        # At least one unit must be a rejection notice.
        error_units = [u for u in units if u.notes and (
            "zip-bomb" in u.notes.lower() or "rejected" in u.notes.lower()
        )]
        assert error_units, (
            f"Expected at least one rejection unit, got: {[u.notes for u in units]}"
        )
        # No unit may carry actual text payload from the bomb.
        for u in units:
            assert len(u.text) < 1024, (
                f"Bomb payload leaked into unit text ({len(u.text)} chars)"
            )


# ---------------------------------------------------------------------------
# Security: Zip Slip
# ---------------------------------------------------------------------------

class TestZipSlip:
    def test_slip_entry_is_rejected(self):
        """slip.zip has ../../evil.txt — must be rejected, nothing written outside temp dir."""
        data = (_FIXTURE_DIR / "slip.zip").read_bytes()
        units = ingest(data, "slip.zip")

        slip_units = [u for u in units if "slip" in (u.notes or "").lower() or "escape" in (u.notes or "").lower()]
        assert slip_units, (
            f"Expected Zip Slip rejection, got: {[u.notes for u in units]}"
        )
        # The evil.txt content must not appear in any unit.
        for u in units:
            assert "zip slip payload" not in u.text

    def test_slip_nothing_escapes(self, tmp_path):
        """Verify the guard at the implementation level rejects escaping paths."""
        from plagdetect.ingest.parsers.zip_handler import _safe_path
        sandbox = tmp_path.resolve()
        # Traversal escape — must be rejected.
        assert _safe_path(sandbox, "../../etc/passwd") is None
        # Absolute paths are neutralised (leading / stripped → placed inside sandbox).
        # They must NOT return a path outside the sandbox.
        result = _safe_path(sandbox, "/absolute/path")
        assert result is not None
        result.relative_to(sandbox)  # raises ValueError if it escapes — that would be the bug
        # Safe relative path — must succeed.
        assert _safe_path(sandbox, "safe/sub/file.txt") is not None


# ---------------------------------------------------------------------------
# Security: over-depth nesting
# ---------------------------------------------------------------------------

class TestDepthCap:
    def test_deep_zip_is_rejected_at_cap(self):
        """deep.zip nests 4 levels — should be rejected at or before depth ZIP_MAX_DEPTH."""
        data = (_FIXTURE_DIR / "deep.zip").read_bytes()
        units = ingest(data, "deep.zip")

        depth_units = [u for u in units if u.notes and "depth" in u.notes.lower()]
        assert depth_units, (
            f"Expected at least one depth-cap rejection, got: {[u.notes for u in units]}"
        )
        # The innermost content must not appear.
        for u in units:
            assert "deeply nested content" not in u.text


# ---------------------------------------------------------------------------
# Benign nested zip
# ---------------------------------------------------------------------------

class TestBenignNestedZip:
    def test_benign_nested_yields_units_with_archive_paths(self):
        """benign_nested.zip has readme.txt + nested/chapter.zip/inner_essay.txt.
        Both must succeed; source_path must contain the archive prefix."""
        data = (_FIXTURE_DIR / "benign_nested.zip").read_bytes()
        units = ingest(data, "benign_nested.zip")

        # Filter to units with actual text.
        text_units = [u for u in units if u.text.strip()]
        assert len(text_units) >= 2, (
            f"Expected ≥2 text units, got {len(text_units)}: {[(u.source_path, u.text[:40]) for u in units]}"
        )

        paths = [u.source_path for u in text_units]
        # Outer readme
        assert any("readme.txt" in p for p in paths), f"readme.txt not found in {paths}"
        # Inner essay — must carry the full archive-relative path
        inner_units = [u for u in text_units if "inner_essay.txt" in u.source_path]
        assert inner_units, f"inner_essay.txt not found in {paths}"
        # Path must reference the enclosing archive(s)
        assert "benign_nested.zip" in inner_units[0].source_path, (
            f"Archive prefix missing: {inner_units[0].source_path}"
        )
        # Text content must survive.
        assert any("quick brown fox" in u.text for u in inner_units)


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

class TestDocxParser:
    def _make_docx(self, text: str) -> bytes:
        from docx import Document
        doc = Document()
        doc.add_paragraph(text)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def test_docx_extraction(self):
        content = "This is a test paragraph from a DOCX file."
        data = self._make_docx(content)
        units = ingest(data, "test.docx")
        assert len(units) == 1
        assert units[0].file_type == "docx"
        assert content in units[0].text
        assert units[0].via_ocr is False


# ---------------------------------------------------------------------------
# /check-file endpoint: TXT upload → detection results
# ---------------------------------------------------------------------------

class TestCheckFileEndpoint:
    def test_txt_upload_returns_source_path_and_matches(self, client):
        """A TXT upload that matches the corpus should return detection results
        with the original filename as source_path."""
        content = b"the quick brown fox jumps over the lazy dog near the river bank"
        resp = client.post(
            "/check-file",
            files={"file": ("submission.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "units" in data
        assert len(data["units"]) == 1
        unit = data["units"][0]
        assert unit["source_path"] == "submission.txt"
        assert unit["file_type"] == "txt"
        assert unit["via_ocr"] is False

    def test_txt_upload_shape(self, client):
        """Every unit result carries the expected keys."""
        resp = client.post(
            "/check-file",
            files={"file": ("doc.txt", b"hello world nothing matches", "text/plain")},
        )
        assert resp.status_code == 200
        for unit in resp.json()["units"]:
            assert {"source_path", "file_type", "via_ocr", "notes", "matches"} == set(unit.keys())

    def test_existing_text_check_still_works(self, client):
        """Ingestion is additive — the text-paste /check path must still work."""
        resp = client.post(
            "/check",
            json={"text": "the quick brown fox jumps over the lazy dog near the river bank"},
        )
        assert resp.status_code == 200
        assert "matches" in resp.json()
