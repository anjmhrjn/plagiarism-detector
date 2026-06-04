from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from plagdetect.corpus import load_corpus
from plagdetect.detect import detect
from plagdetect.detect.spans import extract_spans

_CORPUS_PATH = Path("data/corpus.jsonl")
_CORPUS: list[dict] = []
_CORPUS_INDEX: dict[str, str] = {}

# ~100 KB — enough for any real document, blocks trivially large payloads
_MAX_TEXT_CHARS = 100_000


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _CORPUS, _CORPUS_INDEX
    # Guard lets tests pre-inject a corpus before the lifespan runs.
    if not _CORPUS and _CORPUS_PATH.exists():
        _CORPUS = load_corpus(_CORPUS_PATH)
        _CORPUS_INDEX = {d["id"]: d["canonical"] for d in _CORPUS}
    yield


app = FastAPI(title="Plagiarism Detector", version="0.0.1", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 while the service is up."""
    return {"status": "ok"}


class _CheckBody(BaseModel):
    text: str = Field(..., max_length=_MAX_TEXT_CHARS)


class _SpanOut(BaseModel):
    start: int
    end: int
    text: str


class _MatchOut(BaseModel):
    source: dict  # {id, title}
    score: float
    spans: list[_SpanOut]


class _CheckResponse(BaseModel):
    matches: list[_MatchOut]


@app.post("/check", response_model=_CheckResponse)
def check(body: _CheckBody) -> _CheckResponse:
    if not _CORPUS:
        return _CheckResponse(matches=[])

    raw = detect(body.text, _CORPUS, k=5, top_k=5)
    matches = []
    for m in raw:
        if m["containment"] == 0.0:
            continue
        canonical = _CORPUS_INDEX.get(m["id"], "")
        spans = extract_spans(body.text, canonical, k=5) if canonical else []
        matches.append(
            _MatchOut(
                source={"id": m["id"], "title": m["title"]},
                score=m["containment"],
                spans=[_SpanOut(**s) for s in spans],
            )
        )
    return _CheckResponse(matches=matches)
