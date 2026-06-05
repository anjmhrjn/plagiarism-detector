import pytest
import app.main as main_module
from fastapi.testclient import TestClient

from app.main import app
from plagdetect.ingest.normalize import normalize_text

_TEST_CORPUS = [
    {
        "id": "g:1",
        "title": "Fox and Dog",
        "canonical": normalize_text(
            "the quick brown fox jumps over the lazy dog near the river bank"
        ),
    }
]
_TEST_INDEX = {doc["id"]: doc["canonical"] for doc in _TEST_CORPUS}


@pytest.fixture(autouse=True)
def _inject_corpus(monkeypatch):
    monkeypatch.setattr(main_module, "_CORPUS", _TEST_CORPUS)
    monkeypatch.setattr(main_module, "_CORPUS_INDEX", _TEST_INDEX)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestUI:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_textarea(self, client):
        resp = client.get("/")
        assert "<textarea" in resp.text

    def test_index_is_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_check_endpoint_still_works_from_ui_corpus(self, client):
        """The /check endpoint the UI calls remains functional with the same corpus."""
        query = "the quick brown fox jumps over the lazy dog near the river bank"
        resp = client.post("/check", json={"text": query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matches"][0]["source"]["id"] == "g:1"
        assert data["matches"][0]["score"] > 0.9
        assert len(data["matches"][0]["spans"]) >= 1
