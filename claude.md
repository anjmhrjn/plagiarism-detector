# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A **source-based plagiarism detection system**: it matches an uploaded document
against a defined **reference corpus** and reports which spans match which
sources, with evidence-cited explanations. Portfolio project demonstrating
end-to-end AI systems engineering and a Databricks model-serving lifecycle.

## Scope — LOCKED. Do not expand without explicit approval.

**IN:**
- Source-based plagiarism detection against a reference corpus.
- Images handled as **OCR'd-text-only** (vision model → text → normal pipeline).

**OUT** (do not reintroduce unless explicitly asked):
- AI-generated-text detection.
- Image/figure plagiarism (perceptual hashing).
- Inference-time web crawling.

## Architecture rules (the load-bearing decisions)

- The **detection core is DETERMINISTIC** and does the heavy lifting: n-gram
  fingerprinting + MinHash/LSH for near-duplicates, plus local embeddings +
  vector search for paraphrase. Two-stage: **retrieve candidates, then
  span-level compare.** The LLM stays **OFF this hot path.**
- **Orchestration / file-type routing is plain deterministic control flow**
  (dispatch table + fallback decision tree). Do NOT turn plumbing into an agent.
- The **LLM is reserved for exactly two judgment tasks**: (a) LLM-as-judge
  re-ranking of FLAGGED candidate pairs (cuts false positives from same-topic
  similarity), and (b) generating human-readable, **evidence-cited** explanation
  reports.
- **Ingestion / normalization is the differentiator**: route by file type, parse
  PDF/DOCX, extract zips (recursive/nested) WITH **zip-bomb and Zip Slip
  guards**, OCR images, normalize everything to canonical text + metadata.
- Uploaded docs are **UNTRUSTED input** flowing into LLM prompts. Treat document
  text as **data, never as instructions** (prompt-injection defense). Use
  structured output contracts.

## Build approach

- **LOCAL-FIRST** (Python, FastAPI, FAISS/Chroma for dev), then migrate to
  **Databricks Free Edition**. Do NOT pull in Databricks dependencies before the
  migration phase.
- **Thin vertical slice first**, then deepen each layer on top of something that
  already runs.
- Incremental and checkpoint-driven. **Prefer the simpler non-AI option when
  it's the right call — and say so.**

## Conventions

- **src-layout**: importable code lives in `src/plagdetect/`. The three core
  areas are subpackages: `ingest/`, `detect/`, `corpus/`.
- The HTTP service lives in `app/` (FastAPI; entrypoint `app/main.py`).
- Tests in `tests/`, run with `pytest`.
- Python >= 3.11. Manage the environment with `uv` (or a venv).

## Layout

| Path                      | Responsibility                                          |
|---------------------------|---------------------------------------------------------|
| `src/plagdetect/ingest/`  | file-type routing, parsing, OCR, normalization          |
| `src/plagdetect/detect/`  | fingerprinting, MinHash/LSH, embeddings, span compare   |
| `src/plagdetect/corpus/`  | reference corpus loading / indexing                     |
| `app/`                    | FastAPI service                                         |
| `tests/`                  | pytest suite                                            |