import pytest
import app.main as main_module
from fastapi.testclient import TestClient

from app.main import app, _MAX_TEXT_CHARS
from plagdetect.ingest.normalize import normalize_text

# Same small corpus used across detector tests — three thematically distinct docs.
_RAW = {
    "doc1": ("g:1", "Fox and Dog", "the quick brown fox jumps over the lazy dog near the river bank"),
    "doc2": ("g:2", "Hamlet", "to be or not to be that is the question worth pondering every single day"),
    "doc3": ("g:3", "Fourscore", "four score and seven years ago our fathers brought forth on this continent"),
}
_TEST_CORPUS = [
    {"id": id_, "title": title, "canonical": normalize_text(raw)}
    for id_, title, raw in _RAW.values()
]
_TEST_INDEX = {doc["id"]: doc["canonical"] for doc in _TEST_CORPUS}


@pytest.fixture(autouse=True)
def _inject_corpus(monkeypatch):
    # Inject before the lifespan runs; the guard in _lifespan skips loading when
    # _CORPUS is already non-empty, so this value survives through the test.
    monkeypatch.setattr(main_module, "_CORPUS", _TEST_CORPUS)
    monkeypatch.setattr(main_module, "_CORPUS_INDEX", _TEST_INDEX)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestCheckEndpoint:
    def test_plagiarized_body_returns_correct_source_and_spans(self, client):
        """Verbatim lift from doc1 → top match is doc1, score high, spans non-empty."""
        query = "the quick brown fox jumps over the lazy dog near the river bank"
        resp = client.post("/check", json={"text": query})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matches"]) >= 1
        top = data["matches"][0]
        assert top["source"]["id"] == "g:1"
        assert top["score"] > 0.9
        assert len(top["spans"]) >= 1
        # Offsets must be valid into the original query string.
        for s in top["spans"]:
            assert 0 <= s["start"] < s["end"] <= len(query)
            assert s["text"] == query[s["start"] : s["end"]]

    def test_unrelated_body_returns_no_matches(self, client):
        """Content with no lexical overlap against any corpus doc → empty match list."""
        resp = client.post(
            "/check",
            json={
                "text": (
                    "machine learning algorithms optimize neural network weights "
                    "during gradient descent backpropagation training"
                )
            },
        )
        assert resp.status_code == 200
        assert resp.json()["matches"] == []

    def test_body_too_large_is_rejected(self, client):
        """Text longer than the size cap must be rejected with 422."""
        resp = client.post("/check", json={"text": "x" * (_MAX_TEXT_CHARS + 1)})
        assert resp.status_code == 422

    def test_empty_text_returns_no_matches(self, client):
        """Empty query produces no matches (all containment scores are 0)."""
        resp = client.post("/check", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["matches"] == []

    def test_response_shape(self, client):
        """Every match has the expected keys with correct types."""
        query = "the quick brown fox jumps over the lazy dog near the river bank"
        resp = client.post("/check", json={"text": query})
        assert resp.status_code == 200
        for m in resp.json()["matches"]:
            assert {"source", "score", "lexical_score", "semantic_score", "spans", "judge"} == set(m.keys())
            assert {"id", "title"} == set(m["source"].keys())
            assert isinstance(m["score"], float)
            assert isinstance(m["lexical_score"], float)
            # semantic_score is None when index not loaded; float otherwise
            assert m["semantic_score"] is None or isinstance(m["semantic_score"], float)
            for s in m["spans"]:
                assert {"start", "end", "text"} == set(s.keys())
            # judge is None only when judge module itself is absent; in tests it
            # returns an error verdict (no key) which is a dict, not None.
            assert m["judge"] is None or isinstance(m["judge"], dict)
            if isinstance(m["judge"], dict):
                assert "is_derived" in m["judge"]
                assert "error" in m["judge"]
