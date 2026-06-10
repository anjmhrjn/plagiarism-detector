"""
After-embeddings evaluation: lexical + semantic channels vs the fixture.

Extends the baseline harness logic (eval/run_baseline.py) to add the
semantic channel.  The committed baseline_lexical.json is NOT overwritten.
Output: eval/results/after_embeddings.json

Run from project root:
    python -m eval.run_after_embeddings

Entry point exercised: plagdetect.detect.detect(query_text, corpus, k=5,
    top_k=len(corpus), semantic_index=<SemanticIndex>)
  — the same function /check calls.

Lexical scores are expected to be IDENTICAL to baseline_lexical.json.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from plagdetect.corpus import load_corpus
from plagdetect.detect import detect
from plagdetect.detect.semantic import MODEL_NAME, NORM_VERSION, load_index

_FIXTURE = _ROOT / "eval" / "fixtures" / "paraphrase_seed.jsonl"
_CORPUS_PATH = _ROOT / "data" / "corpus.jsonl"
_INDEX_DIR = _ROOT / "data" / "semantic_index"
_RESULTS_DIR = _ROOT / "eval" / "results"
_BASELINE_PATH = _RESULTS_DIR / "baseline_lexical.json"

FLAG_THRESHOLD = 0.5
POSITIVE_LABELS = {"verbatim", "light_edit", "paraphrase"}
NEGATIVE_LABELS = {"negative_offtopic", "negative_same_topic"}
LABEL_ORDER = ["verbatim", "light_edit", "paraphrase", "negative_offtopic", "negative_same_topic"]


def _load_fixture(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find_rank(
    results: list[dict], true_id: str, sort_key: str
) -> tuple[int | None, float | None]:
    """Rank of true_id when results are sorted by sort_key descending."""
    sorted_results = sorted(
        results, key=lambda x: (x.get(sort_key) or 0.0), reverse=True
    )
    for rank, r in enumerate(sorted_results, start=1):
        if r["id"] == true_id:
            return rank, r.get(sort_key)
    return None, None


def main() -> None:
    print("=" * 72)
    print("AFTER-EMBEDDINGS EVALUATION  (lexical + semantic channels)")
    print("=" * 72)

    if not _INDEX_DIR.exists():
        print(f"\nERROR: semantic index not found at {_INDEX_DIR}")
        print("Run: python scripts/build_semantic_index.py")
        sys.exit(1)

    records = _load_fixture(_FIXTURE)
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"\nFixture : {_FIXTURE.relative_to(_ROOT)}")
    print(f"  Total : {len(records)} records")
    for lbl in LABEL_ORDER:
        if lbl in label_counts:
            print(f"    {lbl}: {label_counts[lbl]}")

    corpus = load_corpus(_CORPUS_PATH)
    print(f"\nCorpus  : {_CORPUS_PATH.relative_to(_ROOT)}  ({len(corpus)} docs)")

    print(f"\nLoading semantic index …")
    sem_index = load_index(_INDEX_DIR)
    print(f"  Model        : {MODEL_NAME}")
    print(f"  NORM_VERSION : {NORM_VERSION}")
    print(f"  Chunks       : {sem_index.n_chunks}")

    print(f"\nRunning {len(records)} fixture records …")

    rows: list[dict] = []
    for rec in records:
        all_results = detect(rec["query_text"], corpus, k=5, top_k=len(corpus), semantic_index=sem_index)

        true_id = rec.get("source_doc_id")

        # Lexical rank/score — same as baseline (containment sort)
        lex_rank, lex_score = _find_rank(all_results, true_id, "containment") if true_id else (None, None)

        # Semantic rank/score — sort by semantic_score
        sem_rank, sem_score = _find_rank(all_results, true_id, "semantic_score") if true_id else (None, None)

        # top1 by lexical (for P/R/F1 on lexical channel)
        lex_top1 = max(all_results, key=lambda x: x["containment"], default=None)
        lex_top1_score = lex_top1["containment"] if lex_top1 else 0.0
        lex_top1_id = lex_top1["id"] if lex_top1 else None

        # top1 by semantic (for reference)
        sem_top1 = max(all_results, key=lambda x: (x["semantic_score"] or 0.0), default=None)
        sem_top1_score = sem_top1["semantic_score"] or 0.0 if sem_top1 else 0.0
        sem_top1_id = sem_top1["id"] if sem_top1 else None

        rows.append(
            {
                "id": rec["id"],
                "label": rec["label"],
                "paraphrase_type": rec.get("paraphrase_type"),
                "source_doc_id": true_id,
                # Lexical channel (must match baseline)
                "lex_rank": lex_rank,
                "lex_score": round(lex_score, 6) if lex_score is not None else None,
                "lex_top1_score": round(lex_top1_score, 6),
                "lex_top1_doc_id": lex_top1_id,
                # Semantic channel (new)
                "sem_rank": sem_rank,
                "sem_score": round(sem_score, 6) if sem_score is not None else None,
                "sem_top1_score": round(sem_top1_score, 6),
                "sem_top1_doc_id": sem_top1_id,
            }
        )

    # ── Delta view: paraphrase rows ──────────────────────────────────────────
    print()
    print("=" * 72)
    print("DELTA: PARAPHRASE ROWS  (the win condition)")
    print("  Lexical vs semantic channel for rows that lexical struggled with")
    print("=" * 72)
    hdr = f"{'id':<18} {'para_type':<20} {'lex_rank':>8} {'lex_score':>9}  {'sem_rank':>8} {'sem_score':>9}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        if r["label"] == "paraphrase":
            lr = str(r["lex_rank"]) if r["lex_rank"] is not None else "—"
            ls = f"{r['lex_score']:.4f}" if r["lex_score"] is not None else "—"
            sr = str(r["sem_rank"]) if r["sem_rank"] is not None else "—"
            ss = f"{r['sem_score']:.4f}" if r["sem_score"] is not None else "—"
            para = r["paraphrase_type"] or ""
            print(f"{r['id']:<18} {para:<20} {lr:>8} {ls:>9}  {sr:>8} {ss:>9}")
    print()

    # ── Verbatim/light_edit: confirm lexical unchanged ───────────────────────
    print("=" * 72)
    print("LEXICAL UNCHANGED VERIFICATION  (verbatim + light_edit)")
    print("=" * 72)

    if _BASELINE_PATH.exists():
        baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_by_id = {r["id"]: r for r in baseline["per_query"]}
        all_match = True
        for r in rows:
            if r["label"] not in {"verbatim", "light_edit"}:
                continue
            b = baseline_by_id.get(r["id"])
            if b is None:
                continue
            expected = b["true_source_score"]
            got = r["lex_score"]
            match = (expected is None and got is None) or (
                expected is not None and got is not None and abs(expected - got) < 1e-5
            )
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            print(f"  {r['id']:<22}  baseline={expected}  now={got}  {status}")
        if all_match:
            print("\n  ✓ All lexical scores match the committed baseline.")
        else:
            print("\n  ✗ LEXICAL MISMATCH — investigate before committing.")
    else:
        print("  (baseline_lexical.json not found — skipping verification)")
    print()

    # ── Per-category rollup ──────────────────────────────────────────────────
    print("=" * 72)
    print("PER-CATEGORY ROLLUP")
    print("=" * 72)

    rollup: dict[str, dict] = {}

    for lbl in ["verbatim", "light_edit", "paraphrase"]:
        label_rows = [r for r in rows if r["label"] == lbl]
        lex_ranks = [r["lex_rank"] for r in label_rows if r["lex_rank"] is not None]
        lex_scores = [r["lex_score"] for r in label_rows if r["lex_score"] is not None]
        sem_ranks = [r["sem_rank"] for r in label_rows if r["sem_rank"] is not None]
        sem_scores = [r["sem_score"] for r in label_rows if r["sem_score"] is not None]

        lex_mr = f"{mean(lex_ranks):.2f}" if lex_ranks else "n/a"
        lex_ms = f"{mean(lex_scores):.4f}" if lex_scores else "n/a"
        sem_mr = f"{mean(sem_ranks):.2f}" if sem_ranks else "n/a"
        sem_ms = f"{mean(sem_scores):.4f}" if sem_scores else "n/a"

        print(
            f"  {lbl:<22}  "
            f"lex mean_rank={lex_mr}  lex mean_score={lex_ms}  |  "
            f"sem mean_rank={sem_mr}  sem mean_score={sem_ms}"
        )
        rollup[lbl] = {
            "n": len(label_rows),
            "lex_mean_rank": round(mean(lex_ranks), 2) if lex_ranks else None,
            "lex_mean_score": round(mean(lex_scores), 4) if lex_scores else None,
            "sem_mean_rank": round(mean(sem_ranks), 2) if sem_ranks else None,
            "sem_mean_score": round(mean(sem_scores), 4) if sem_scores else None,
        }

    print()
    for lbl in ["negative_offtopic", "negative_same_topic"]:
        label_rows = [r for r in rows if r["label"] == lbl]
        sem_top1s = [r["sem_top1_score"] for r in label_rows]
        lex_top1s = [r["lex_top1_score"] for r in label_rows]
        print(
            f"  {lbl:<22}  "
            f"lex mean_top1={mean(lex_top1s):.4f}  max_top1={max(lex_top1s):.4f}  |  "
            f"sem mean_top1={mean(sem_top1s):.4f}  max_top1={max(sem_top1s):.4f}"
        )
        rollup[lbl] = {
            "n": len(label_rows),
            "lex_mean_top1_score": round(mean(lex_top1s), 4) if lex_top1s else None,
            "lex_max_top1_score": round(max(lex_top1s), 4) if lex_top1s else None,
            "sem_mean_top1_score": round(mean(sem_top1s), 4) if sem_top1s else None,
            "sem_max_top1_score": round(max(sem_top1s), 4) if sem_top1s else None,
        }

    # ── Write artifact ───────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "AFTER — semantic embeddings layer added",
            "entry_point": "plagdetect.detect.detect(query_text, corpus, k=5, top_k=len(corpus), semantic_index=<SemanticIndex>)",
            "model_name": MODEL_NAME,
            "norm_version": NORM_VERSION,
            "n_chunks": sem_index.n_chunks,
            "corpus_path": str(_CORPUS_PATH.relative_to(_ROOT)),
            "corpus_size": len(corpus),
            "fixture_path": str(_FIXTURE.relative_to(_ROOT)),
            "fixture_size": len(records),
            "label_counts": label_counts,
        },
        "per_query": rows,
        "rollup": rollup,
    }

    out_path = _RESULTS_DIR / "after_embeddings.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\nResults written → eval/results/after_embeddings.json")


if __name__ == "__main__":
    main()
