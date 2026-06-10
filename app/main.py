from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from plagdetect.corpus import load_corpus
from plagdetect.detect import detect
from plagdetect.detect.spans import extract_spans

_CORPUS_PATH = Path("data/corpus.jsonl")
_SEMANTIC_INDEX_DIR = Path("data/semantic_index")
_CORPUS: list[dict] = []
_CORPUS_INDEX: dict[str, str] = {}
_SEMANTIC_INDEX = None  # SemanticIndex | None; None means lexical-only mode

# ~100 KB — enough for any real document, blocks trivially large payloads
_MAX_TEXT_CHARS = 100_000

# Minimum cosine similarity for a semantic-only hit (no lexical overlap) to
# appear in /check results.  Not a fusion weight — scores remain independent.
_SEM_INCLUDE_THRESHOLD = 0.40


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _CORPUS, _CORPUS_INDEX, _SEMANTIC_INDEX
    # Guard lets tests pre-inject a corpus before the lifespan runs.
    if not _CORPUS and _CORPUS_PATH.exists():
        _CORPUS = load_corpus(_CORPUS_PATH)
        _CORPUS_INDEX = {d["id"]: d["canonical"] for d in _CORPUS}

    if _SEMANTIC_INDEX is None and _SEMANTIC_INDEX_DIR.exists():
        try:
            from plagdetect.detect.semantic import load_index
            _SEMANTIC_INDEX = load_index(_SEMANTIC_INDEX_DIR)
        except Exception as exc:
            import sys
            print(f"WARNING: semantic index not loaded: {exc}", file=sys.stderr)
    yield


app = FastAPI(title="Plagiarism Detector", version="0.0.2", lifespan=_lifespan)


_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plagiarism Detector</title>
<style>
  body { font-family: sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; box-sizing: border-box; }
  mark { background: #ffe066; }
  pre { white-space: pre-wrap; word-break: break-word; border: 1px solid #ddd;
        padding: .75rem; border-radius: 4px; }
  #status { color: #c00; min-height: 1.4em; }
  h2 { margin-top: 1.5rem; font-size: 1rem; }
  .scores { font-size: .85rem; color: #555; margin-bottom: .25rem; }
</style>
</head>
<body>
<h1>Plagiarism Detector</h1>
<form id="form">
  <textarea id="text" rows="12" placeholder="Paste the text you want to check…"></textarea>
  <br><button type="submit" style="margin-top:.5rem">Check</button>
</form>
<p id="status"></p>
<div id="results"></div>

<script>
const MAX_CHARS = """ + str(_MAX_TEXT_CHARS) + """;

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = document.getElementById("text").value;
  const status = document.getElementById("status");
  const results = document.getElementById("results");

  if (!text.trim()) {
    status.textContent = "Please enter some text first.";
    results.innerHTML = "";
    return;
  }
  if (text.length > MAX_CHARS) {
    status.textContent = "Input is too long (limit: " + MAX_CHARS.toLocaleString() + " characters).";
    results.innerHTML = "";
    return;
  }

  status.textContent = "Checking…";
  results.innerHTML = "";

  let data;
  try {
    const resp = await fetch("/check", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    if (resp.status === 422) {
      status.textContent = "Input rejected by the server (too long or invalid).";
      return;
    }
    if (!resp.ok) {
      status.textContent = "Server error (" + resp.status + "). Please try again.";
      return;
    }
    data = await resp.json();
  } catch (err) {
    status.textContent = "Network error — is the server reachable?";
    return;
  }

  status.textContent = "";
  if (!data.matches.length) {
    results.textContent = "No matches found.";
    return;
  }

  for (const m of data.matches) {
    const section = document.createElement("section");

    const h2 = document.createElement("h2");
    h2.textContent =
      m.source.title + " — " + (m.score * 100).toFixed(1) + "% containment";
    section.appendChild(h2);

    const scores = document.createElement("p");
    scores.className = "scores";
    const lexStr = "lexical: " + (m.lexical_score * 100).toFixed(1) + "%";
    const semStr = m.semantic_score != null
      ? "  semantic: " + (m.semantic_score * 100).toFixed(1) + "%"
      : "";
    scores.textContent = lexStr + semStr;
    section.appendChild(scores);

    const pre = document.createElement("pre");
    // highlightSpans mirrors app/render.py:render_html — walk spans in order,
    // escape plain text, wrap matched regions in <mark>.
    pre.innerHTML = highlightSpans(text, m.spans);
    section.appendChild(pre);

    results.appendChild(section);
  }
});

function highlightSpans(text, spans) {
  let html = "";
  let cursor = 0;
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  for (const s of sorted) {
    if (s.start > cursor) html += esc(text.slice(cursor, s.start));
    html += "<mark>" + esc(text.slice(s.start, s.end)) + "</mark>";
    cursor = s.end;
  }
  if (cursor < text.length) html += esc(text.slice(cursor));
  return html;
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


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
    score: float  # = lexical_score (containment); kept for backward compatibility
    lexical_score: float
    semantic_score: float | None
    spans: list[_SpanOut]


class _CheckResponse(BaseModel):
    matches: list[_MatchOut]


@app.post("/check", response_model=_CheckResponse)
def check(body: _CheckBody) -> _CheckResponse:
    if not _CORPUS:
        return _CheckResponse(matches=[])

    # Run both channels through the shared detect() entry point.
    # top_k=len(_CORPUS) so semantic-only hits (containment=0) aren't cut off
    # before the union filter below.  Lexical computation is O(corpus) anyway.
    raw = detect(body.text, _CORPUS, k=5, top_k=len(_CORPUS), semantic_index=_SEMANTIC_INDEX)

    matches: list[_MatchOut] = []
    for m in raw:
        is_lex = m["containment"] > 0.0
        is_sem = _SEMANTIC_INDEX is not None and (m["semantic_score"] or 0.0) >= _SEM_INCLUDE_THRESHOLD
        if not is_lex and not is_sem:
            continue

        canonical = _CORPUS_INDEX.get(m["id"], "")
        # Spans come from the lexical channel only; semantic matches show no highlights.
        spans = extract_spans(body.text, canonical, k=5) if canonical and is_lex else []

        matches.append(
            _MatchOut(
                source={"id": m["id"], "title": m["title"]},
                score=m["containment"],
                lexical_score=m["containment"],
                semantic_score=m["semantic_score"],
                spans=[_SpanOut(**s) for s in spans],
            )
        )
        if len(matches) >= 5:
            break

    return _CheckResponse(matches=matches)
