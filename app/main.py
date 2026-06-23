from contextlib import asynccontextmanager
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from plagdetect.corpus import load_corpus
from plagdetect.detect import detect
from plagdetect.detect.spans import extract_spans
from plagdetect.judge import judge_pair

_log = logging.getLogger(__name__)

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

# --- LLM judge gating thresholds -------------------------------------------
# Below floor  → auto-exclude (noise / same-topic overlap; judge adds nothing).
# Above ceiling → auto-include (near-verbatim; judge confirmation is redundant).
# Middle band   → call judge (ambiguous paraphrase vs. same-topic).
_JUDGE_LEX_FLOOR  = 0.08   # lexical containment below this → skip judge, exclude
_JUDGE_SEM_FLOOR  = 0.50   # semantic score below this (when lex also below floor) → exclude
_JUDGE_LEX_CEIL   = 0.60   # lexical containment at or above this → skip judge, include


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


_SHARED_CSS = """\
  body { font-family: sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  nav { margin-bottom: 1.5rem; font-size: .9rem; }
  nav a { margin-right: 1rem; text-decoration: none; color: #0066cc; }
  nav a.active { font-weight: bold; color: #333; text-decoration: none; pointer-events: none; }
  mark { background: #ffe066; }
  pre { white-space: pre-wrap; word-break: break-word; border: 1px solid #ddd;
        padding: .75rem; border-radius: 4px; }
  .status { color: #c00; min-height: 1.4em; }
  h2 { margin-top: 1.5rem; font-size: 1rem; }
  .scores { font-size: .85rem; color: #555; margin-bottom: .25rem; }
"""

_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plagiarism Detector</title>
<style>
  textarea { width: 100%; box-sizing: border-box; }
""" + _SHARED_CSS + """
</style>
</head>
<body>
<h1>Plagiarism Detector</h1>
<nav>
  <a href="/" class="active">Paste text</a>
  <a href="/upload">Upload file</a>
</nav>
<form id="form">
  <textarea id="text" rows="12" placeholder="Paste the text you want to check…"></textarea>
  <br><button type="submit" style="margin-top:.5rem">Check</button>
</form>
<p id="status" class="status"></p>
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


_UPLOAD_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plagiarism Detector — File Upload</title>
<style>
""" + _SHARED_CSS + """
  #drop-zone {
    border: 2px dashed #aaa; border-radius: 6px; padding: 2rem 1rem;
    text-align: center; color: #666; cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  #drop-zone.drag-over { border-color: #0066cc; background: #f0f7ff; }
  #drop-zone.has-file  { border-color: #28a745; background: #f0fff4; color: #155724; }
  #file-input { display: none; }
  button { margin-top: .75rem; }
  .unit-card {
    border: 1px solid #ddd; border-radius: 6px;
    margin-top: 1.25rem; overflow: hidden;
  }
  .unit-header {
    display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap;
    padding: .6rem .75rem; background: #f8f8f8; border-bottom: 1px solid #ddd;
    font-size: .9rem;
  }
  .unit-path { font-weight: bold; word-break: break-all; }
  .badge {
    font-size: .7rem; font-weight: bold; padding: .15rem .45rem;
    border-radius: 3px; text-transform: uppercase; letter-spacing: .03em;
  }
  .badge-pdf   { background: #fde8e8; color: #9b1c1c; }
  .badge-docx  { background: #dbeafe; color: #1e3a8a; }
  .badge-txt   { background: #e5e7eb; color: #374151; }
  .badge-image { background: #fef3c7; color: #92400e; }
  .badge-zip   { background: #ede9fe; color: #4c1d95; }
  .badge-ocr   { background: #d1fae5; color: #065f46; }
  .badge-unknown { background: #e5e7eb; color: #6b7280; }
  .unit-body   { padding: .75rem; }
  .notes-box {
    font-size: .85rem; background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 4px; padding: .5rem .75rem; margin-bottom: .75rem;
    color: #78350f;
  }
  .no-matches  { color: #555; font-size: .9rem; }
  .match       { margin-top: 1rem; }
  .match-title { font-weight: bold; font-size: .95rem; }
  .match-scores { font-size: .82rem; color: #555; margin: .2rem 0 .4rem; }
  .judge-box {
    font-size: .82rem; border-radius: 4px; padding: .4rem .65rem;
    margin: .4rem 0; display: flex; flex-wrap: wrap; gap: .3rem .75rem;
  }
  .judge-auto   { background: #f0fff4; border: 1px solid #86efac; color: #14532d; }
  .judge-clean  { background: #f3f4f6; border: 1px solid #d1d5db; color: #374151; }
  .judge-flagged{ background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; }
  .rel-verbatim  { color: #dc2626; font-weight: bold; }
  .rel-edited    { color: #d97706; font-weight: bold; }
  .rel-paraphrase{ color: #ca8a04; font-weight: bold; }
  .rel-same_topic_only { color: #16a34a; }
  .rel-unrelated       { color: #6b7280; }
  .evidence-list { margin: .3rem 0 0 1rem; padding: 0; list-style: disc; }
  .evidence-list li { margin: .15rem 0; font-style: italic; color: #555; }
  .excerpts { margin-top: .5rem; }
  .excerpt  {
    border-left: 3px solid #ffe066; padding: .3rem .6rem;
    margin: .3rem 0; font-size: .85rem; background: #fffdf0;
    white-space: pre-wrap; word-break: break-word;
  }
</style>
</head>
<body>
<h1>Plagiarism Detector</h1>
<nav>
  <a href="/">Paste text</a>
  <a href="/upload" class="active">Upload file</a>
</nav>

<div id="drop-zone" role="button" tabindex="0" aria-label="File drop zone">
  <p>Drop a file here, or <strong>click to browse</strong></p>
  <p style="font-size:.8rem;margin:.25rem 0 0">PDF · DOCX · TXT · PNG / JPG / TIFF · ZIP</p>
  <input type="file" id="file-input"
         accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.tiff,.tif,.zip">
</div>
<button id="check-btn" disabled>Check file</button>
<p id="status" class="status"></p>
<div id="results"></div>

<script>
const zone   = document.getElementById("drop-zone");
const input  = document.getElementById("file-input");
const btn    = document.getElementById("check-btn");
const status = document.getElementById("status");
const results = document.getElementById("results");
let chosenFile = null;

// --- file selection ---------------------------------------------------------
zone.addEventListener("click", () => input.click());
zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") input.click(); });

input.addEventListener("change", () => setFile(input.files[0]));

zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
zone.addEventListener("dragleave", ()  => zone.classList.remove("drag-over"));
zone.addEventListener("drop", e => {
  e.preventDefault();
  zone.classList.remove("drag-over");
  setFile(e.dataTransfer.files[0]);
});

function setFile(f) {
  if (!f) return;
  chosenFile = f;
  zone.classList.add("has-file");
  zone.querySelector("p").textContent = "Selected: " + f.name + " (" + fmtBytes(f.size) + ")";
  btn.disabled = false;
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
  return (n/(1024*1024)).toFixed(1) + " MB";
}

// --- submit -----------------------------------------------------------------
btn.addEventListener("click", async () => {
  if (!chosenFile) return;
  status.textContent = "Uploading and checking…";
  results.innerHTML  = "";
  btn.disabled       = true;

  const fd = new FormData();
  fd.append("file", chosenFile);

  let data;
  try {
    const resp = await fetch("/check-file", { method: "POST", body: fd });
    if (!resp.ok) {
      status.textContent = "Server error (" + resp.status + "). See server logs.";
      btn.disabled = false;
      return;
    }
    data = await resp.json();
  } catch (err) {
    status.textContent = "Network error — is the server reachable?";
    btn.disabled = false;
    return;
  }

  status.textContent = "";
  btn.disabled = false;

  if (!data.units.length) {
    results.textContent = "No content extracted.";
    return;
  }
  for (const unit of data.units) results.appendChild(renderUnit(unit));
});

// --- rendering --------------------------------------------------------------
function renderUnit(unit) {
  const card = el("div", "unit-card");

  // header
  const hdr = el("div", "unit-header");
  const path = el("span", "unit-path");
  path.textContent = unit.source_path;
  hdr.appendChild(path);
  hdr.appendChild(badge(unit.file_type));
  if (unit.via_ocr) hdr.appendChild(badge("ocr"));
  card.appendChild(hdr);

  const body = el("div", "unit-body");

  // notes / errors
  if (unit.notes) {
    const nb = el("div", "notes-box");
    nb.textContent = "⚠ " + unit.notes;
    body.appendChild(nb);
  }

  if (!unit.matches.length) {
    const nm = el("p", "no-matches");
    nm.textContent = unit.notes && !unit.notes.includes("OCR") && !unit.notes.includes("Rejected")
      ? "Rejected — no content checked."
      : "No plagiarism detected.";
    body.appendChild(nm);
  } else {
    for (const m of unit.matches) body.appendChild(renderMatch(m));
  }

  card.appendChild(body);
  return card;
}

function renderMatch(m) {
  const div = el("div", "match");

  const title = el("p", "match-title");
  title.textContent = m.source.title + " — " + (m.score * 100).toFixed(1) + "% containment";
  div.appendChild(title);

  const scores = el("p", "match-scores");
  let s = "lexical: " + (m.lexical_score * 100).toFixed(1) + "%";
  if (m.semantic_score != null) s += "  ·  semantic: " + (m.semantic_score * 100).toFixed(1) + "%";
  scores.textContent = s;
  div.appendChild(scores);

  div.appendChild(renderJudge(m.judge, m.score));

  // matched excerpts (spans already carry .text — no client-side highlighting needed)
  if (m.spans.length) {
    const exc = el("div", "excerpts");
    for (const sp of m.spans) {
      const d = el("div", "excerpt");
      d.textContent = sp.text;
      exc.appendChild(d);
    }
    div.appendChild(exc);
  }

  return div;
}

function renderJudge(judge, lexScore) {
  const box = el("div", "judge-box");

  if (judge === null) {
    // auto-include gate fired (high containment, no LLM call)
    box.classList.add("judge-auto");
    const lbl = el("span");
    lbl.textContent = "Auto-flagged — containment " + (lexScore * 100).toFixed(1) + "% exceeds threshold; LLM call skipped.";
    box.appendChild(lbl);
    return box;
  }

  if (judge.error) {
    box.classList.add("judge-clean");
    const lbl = el("span");
    lbl.textContent = "Judge unavailable: " + judge.error;
    box.appendChild(lbl);
    return box;
  }

  box.classList.add(judge.is_derived ? "judge-flagged" : "judge-clean");

  const rel = el("span", "rel-" + judge.relationship);
  rel.textContent = judge.relationship.replace(/_/g, " ");
  box.appendChild(rel);

  const conf = el("span");
  conf.textContent = judge.confidence + " confidence";
  box.appendChild(conf);

  const rat = el("span");
  rat.style.fontStyle = "italic";
  rat.style.flexBasis = "100%";
  rat.textContent = judge.rationale;
  box.appendChild(rat);

  if (judge.evidence && judge.evidence.length) {
    const ul = el("ul", "evidence-list");
    ul.style.flexBasis = "100%";
    for (const e of judge.evidence) {
      const li = el("li");
      li.textContent = "\\"" + e + "\\"";
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }

  return box;
}

function badge(type) {
  const s = el("span", "badge badge-" + type);
  s.textContent = type;
  return s;
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


@app.get("/upload", response_class=HTMLResponse)
def upload_page() -> str:
    return _UPLOAD_PAGE


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
    judge: dict | None  # structured verdict from LLM judge; None when judge is unavailable


class _CheckResponse(BaseModel):
    matches: list[_MatchOut]


def _judge_gate(containment: float, semantic_score: float | None) -> str:
    """
    Returns one of three decisions before touching the LLM:
      "exclude"  — below floor on both channels; auto-reject, skip judge.
      "include"  — above ceiling on lexical; auto-accept, skip judge.
      "judge"    — ambiguous middle band; call judge_pair().
    """
    sem = semantic_score or 0.0
    if containment < _JUDGE_LEX_FLOOR and sem < _JUDGE_SEM_FLOOR:
        return "exclude"
    if containment >= _JUDGE_LEX_CEIL:
        return "include"
    return "judge"


def _judge_passage(query_text: str, canonical: str, m: dict, is_lex: bool) -> str:
    """
    Pick the source passage to send to the judge based on which channel flagged the candidate.

    Lexical hit  → reverse extract_spans to find the matching region in the source,
                   then widen by ±500 chars so the judge sees the full edited passage.
    Semantic-only → use the chunk offsets from the semantic index directly.
    """
    if is_lex and canonical:
        source_spans = extract_spans(canonical, query_text, k=5)
        if source_spans:
            span_start = min(s["start"] for s in source_spans)
            span_end = max(s["end"] for s in source_spans)
            ctx_start = max(0, span_start - 500)
            ctx_end = min(len(canonical), span_end + 500)
            return canonical[ctx_start:ctx_end]

    # Semantic chunk (precise offset from the index).
    char_start = m.get("chunk_char_start")
    char_end = m.get("chunk_char_end")
    if char_start is not None and char_end is not None and canonical:
        return canonical[char_start:char_end]

    return canonical[:2000]


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

        gate = _judge_gate(m["containment"], m["semantic_score"])
        if gate == "exclude":
            continue

        canonical = _CORPUS_INDEX.get(m["id"], "")
        verdict: dict | None = None

        if gate == "include":
            is_derived = True
        else:
            # "judge" band: ambiguous — let the LLM decide.
            candidate_chunk = _judge_passage(body.text, canonical, m, is_lex)
            verdict = judge_pair(
                body.text,
                candidate_chunk,
                lexical_score=m["containment"],
                semantic_score=m["semantic_score"],
            )
            if verdict["error"] is not None:
                _log.warning("judge unavailable for %s, falling back to provisional flag: %s", m["id"], verdict["error"])
                is_derived = is_lex or is_sem
            else:
                is_derived = verdict["is_derived"]

        if not is_derived:
            continue

        # Spans come from the lexical channel only; semantic-only matches show no highlights.
        spans = extract_spans(body.text, canonical, k=5) if canonical and is_lex else []

        matches.append(
            _MatchOut(
                source={"id": m["id"], "title": m["title"]},
                score=m["containment"],
                lexical_score=m["containment"],
                semantic_score=m["semantic_score"],
                spans=[_SpanOut(**s) for s in spans],
                judge=verdict,
            )
        )
        if len(matches) >= 5:
            break

    return _CheckResponse(matches=matches)


# ---------------------------------------------------------------------------
# File-upload variant — ingests arbitrary files then runs the same pipeline
# ---------------------------------------------------------------------------

class _FileUnitResult(BaseModel):
    source_path: str
    file_type: str
    via_ocr: bool
    notes: str | None
    matches: list[_MatchOut]


class _CheckFileResponse(BaseModel):
    units: list[_FileUnitResult]


@app.post("/check-file", response_model=_CheckFileResponse)
async def check_file(file: UploadFile = File(...)) -> _CheckFileResponse:
    """
    Accept an uploaded file (PDF, DOCX, TXT, image, or ZIP), ingest it into
    one or more canonical text units, then run each unit through the same
    detect → judge pipeline that /check uses.

    Provenance (source_path) is preserved per unit — zip-contained files
    carry archive-relative paths (e.g. submission.zip/essays/paper.pdf).
    """
    from plagdetect.ingest import ingest
    from plagdetect.ingest.normalize import normalize_text

    if not _CORPUS:
        return _CheckFileResponse(units=[])

    raw_bytes = await file.read()
    filename  = file.filename or "upload"

    # Treat uploaded document text as DATA, never as instructions (prompt-injection
    # defense).  The router and parsers never evaluate document content; the LLM
    # judge receives it only inside a structured, role-bounded prompt in judge.py.
    units = ingest(raw_bytes, filename)

    results: list[_FileUnitResult] = []
    for unit in units:
        # Skip error/empty units — still surface them with empty matches.
        if not unit.text.strip():
            results.append(_FileUnitResult(
                source_path=unit.source_path,
                file_type=unit.file_type,
                via_ocr=unit.via_ocr,
                notes=unit.notes,
                matches=[],
            ))
            continue

        normalized = normalize_text(unit.text)
        if len(normalized) > _MAX_TEXT_CHARS:
            results.append(_FileUnitResult(
                source_path=unit.source_path,
                file_type=unit.file_type,
                via_ocr=unit.via_ocr,
                notes="Skipped: normalized text exceeds size cap.",
                matches=[],
            ))
            continue

        raw = detect(normalized, _CORPUS, k=5, top_k=len(_CORPUS), semantic_index=_SEMANTIC_INDEX)

        matches: list[_MatchOut] = []
        for m in raw:
            is_lex = m["containment"] > 0.0
            is_sem = _SEMANTIC_INDEX is not None and (m["semantic_score"] or 0.0) >= _SEM_INCLUDE_THRESHOLD
            if not is_lex and not is_sem:
                continue

            gate = _judge_gate(m["containment"], m["semantic_score"])
            if gate == "exclude":
                continue

            canonical = _CORPUS_INDEX.get(m["id"], "")
            verdict: dict | None = None

            if gate == "include":
                is_derived = True
            else:
                candidate_chunk = _judge_passage(normalized, canonical, m, is_lex)
                verdict = judge_pair(
                    normalized,
                    candidate_chunk,
                    lexical_score=m["containment"],
                    semantic_score=m["semantic_score"],
                )
                if verdict["error"] is not None:
                    is_derived = is_lex or is_sem
                else:
                    is_derived = verdict["is_derived"]

            if not is_derived:
                continue

            spans = extract_spans(normalized, canonical, k=5) if canonical and is_lex else []

            matches.append(_MatchOut(
                source={"id": m["id"], "title": m["title"]},
                score=m["containment"],
                lexical_score=m["containment"],
                semantic_score=m["semantic_score"],
                spans=[_SpanOut(**s) for s in spans],
                judge=verdict,
            ))
            if len(matches) >= 5:
                break

        results.append(_FileUnitResult(
            source_path=unit.source_path,
            file_type=unit.file_type,
            via_ocr=unit.via_ocr,
            notes=unit.notes,
            matches=matches,
        ))

    return _CheckFileResponse(units=results)
