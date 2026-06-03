# Plagiarism Detector

Source-based plagiarism detection: match an uploaded document against a
reference corpus and report which spans match which sources, with
evidence-cited explanations.

## Scope

- **In:** source-based detection against a reference corpus; images handled as
  OCR'd text.
- **Out:** AI-generated-text detection, perceptual image hashing, inference-time
  web crawling.

## Quickstart

```bash
# 1. Create and activate an environment
python3 -m venv venv
source .venv/bin/activate

# 2. Install the package + dev tools
pip install -e ".[dev]"

# 3. Run the service
uvicorn app.main:app --reload

# 4. Check it's alive
curl http://127.0.0.1:8000/health        # -> {"status":"ok"}

# 5. Run the tests
pytest
```

## Project layout

```
src/plagdetect/   importable package
  ingest/         file-type routing, parsing, OCR, normalization
  detect/         fingerprinting, MinHash/LSH, embeddings, span compare
  corpus/         reference corpus loading / indexing
app/              FastAPI service (entrypoint app/main.py)
tests/            pytest suite
```

## Built with the help of Claude Code

This project is built with the help of Claude Code, and that workflow is kept visible. See
[`CLAUDE.md`](./CLAUDE.md) for the working contract — scope, architecture rules,
and conventions the assistant follows in this repo.