# Checkpoint — Week 1

**Date:** 2026-06-05
**Phase:** End of Week 1 (Days 1–5), local-first MVP
**Status:** Working end-to-end lexical detector with a clickable UI. No Databricks yet (as planned).

## Goal for week 1

Ship a thin vertical slice: plain text in → n-gram containment match against a
small local corpus → top sources with highlighted overlapping spans, wrapped in
a FastAPI endpoint and a one-page UI. Local-first, deterministic core, LLM off
the hot path.

## What works

- **Repo skeleton & service.** src-layout package
  (`src/plagdetect/{ingest,detect,corpus}`), FastAPI app, `/health`, `CLAUDE.md`.
  Public repo, runs from a clean clone.
- **Corpus + normalization.** ~100 open-licensed Project Gutenberg documents,
  normalized via `normalize_text()` (Unicode NFC, lowercase, whitespace collapse,
  Gutenberg header/footer boilerplate stripping) and stored as canonical text +
  metadata in JSONL. Each record keeps both the original `text` and the
  `canonical` form.
- **Detector v0.** Word n-gram shingling (k=5), containment + Jaccard scoring,
  brute-force ranking over the corpus, top-k results. Ranks by **containment**
  (asymmetric — "how much of the query appears in the source"), reports Jaccard
  alongside as a second signal.
- **Span-level evidence.** Offset-aware tokenizer maps matched shingles back to
  character offsets in the user's *original* text; overlapping/adjacent matched
  positions merge into maximal spans. `/check` returns structured
  `{matches: [{source, score, spans}]}`.
- **UI.** One-page interface served directly from the FastAPI app: paste text →
  POST `/check` → matched passages highlighted inline. Usable by a non-engineer.

**Verified behavior:** a verbatim copy of a corpus passage ranks the correct
source #1 with containment = 1.0; unrelated text scores low; highlights land on
the correct characters in the original (un-normalized) text.

## Findings & failure modes

### 1. Containment < 1.0 on a known copy — RESOLVED (test artifact, not a pipeline bug)

While testing, a passage known to be in the corpus scored below 1.0. Inspecting
the query shingles showed corrupted tokens such as `r` and `nphilosophy`
(e.g. `while r nphilosophy has hitherto`).

**Cause.** The test input was typed into the textarea as *literal backslash
characters* (`\`, `r`, `\`, `n`), not real CR/LF control characters. The `\w+`
tokenizer correctly drops the backslashes (non-word characters) but keeps `r`
and `n` as word characters, fusing `n` onto the following word. Every k-word
shingle crossing that point was corrupted, which dragged containment down and
dropped the following word from the matched spans.

**Resolution.** Real browser submissions contain actual newlines, which the
tokenizer treats as separators and handles correctly. This was an artifact of
hand-typing escape sequences into the textarea, not a defect in the
ingest → tokenize → detect path. No code change required.

**Insight worth keeping (hardening, not urgent).** The near-miss surfaced a real
principle: normalization should be the explicit job of `normalize_text()`, not an
accidental side effect of the tokenizer's `\w+` regex. The regex "cleaned enough
to look right but not enough to be correct," which is what made the symptom
confusing. If a real client ever submits escaped or otherwise malformed input
(some JSON clients escape control characters), the correct fix is to
normalize/decode **at the request boundary** so the tokenizer only ever splits
already-clean text. Logged as a future hardening item; not needed for current
(browser) usage.

### 2. Paraphrase not detected — BY DESIGN (known ceiling, motivates week 2)

Paraphrased versions of corpus passages score low and are not flagged.

This is n-gram shingling behaving exactly as designed: shingles match *surface
word sequences*, so reworded text shares almost no k-word windows with its source
even when the meaning is identical. This is the expected ceiling of a purely
lexical detector — not a bug. It is the direct motivation for the
semantic/embeddings layer: embeddings compare meaning in vector space rather than
exact word runs, which is what catches paraphrase.

## Evidence quality (what we don't know yet)

Current assessment is **qualitative** — verbatim copies, unrelated text, and a
few paraphrases, eyeballed. There are no precision/recall/F1 numbers yet, and the
corpus is small (100 docs), so brute-force scoring is instant at this size. Real
metrics arrive with the eval harness against a labeled corpus (PAN-PC) in
week 2. Before then, a structured eyeball over a handful of varied pairs
(verbatim / lightly edited / paraphrased / unrelated) would sharpen the picture
cheaply.
