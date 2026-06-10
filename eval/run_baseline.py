"""
Baseline evaluation harness for the lexical plagiarism detector.

Entry point exercised: plagdetect.detect.detect()
  query_text → detect() → shingle() → tokenize() + canonical() (NFC + lowercase per token)

normalize_text() is applied at corpus-build time and stored in the corpus 'canonical'
field. The /check route also does NOT call normalize_text() on the query — this harness
matches that behavior exactly.

Run from project root:
    python -m eval.run_baseline
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

_FIXTURE = _ROOT / "eval" / "fixtures" / "paraphrase_seed.jsonl"
_CORPUS_PATH = _ROOT / "data" / "corpus.jsonl"
_RESULTS_DIR = _ROOT / "eval" / "results"

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


def _run_query(query_text: str, corpus: list[dict]) -> list[tuple[str, float]]:
    raw = detect(query_text, corpus, k=5, top_k=5)
    return [(m["id"], m["containment"]) for m in raw]


def _find_rank(ranked: list[tuple[str, float]], true_id: str) -> tuple[int | None, float | None]:
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        if doc_id == true_id:
            return rank, score
    return None, None


def main() -> None:
    print("=" * 72)
    print("BASELINE LEXICAL DETECTOR EVALUATION")
    print("=" * 72)

    records = _load_fixture(_FIXTURE)
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"\nFixture : {_FIXTURE.relative_to(_ROOT)}")
    print(f"  Fields: {', '.join(records[0].keys())}")
    print(f"  Total : {len(records)} records")
    for lbl in LABEL_ORDER:
        if lbl in label_counts:
            print(f"    {lbl}: {label_counts[lbl]}")

    corpus = load_corpus(_CORPUS_PATH)
    print(f"\nCorpus  : {_CORPUS_PATH.relative_to(_ROOT)}")
    print(f"  Documents: {len(corpus)}")
    print(f"  Doc-id field: 'id'  (matches fixture 'source_doc_id')")

    print(f"\nEntry point : plagdetect.detect.detect(query_text, corpus, k=5, top_k=len(corpus))")
    print(f"  Pipeline  : query_text → shingle() → tokenize() + canonical() (NFC+lowercase/token)")
    print(f"  normalize_text() runs at corpus-build time only; /check does the same.")
    print(f"  Running all {len(records)} fixture records through full pipeline …")

    rows: list[dict] = []
    for rec in records:
        ranked = _run_query(rec["query_text"], corpus)
        top1_score = ranked[0][1] if ranked else 0.0
        top1_doc_id = ranked[0][0] if ranked else None

        true_id = rec.get("source_doc_id")
        if true_id is not None:
            rank, src_score = _find_rank(ranked, true_id)
        else:
            rank, src_score = None, None

        rows.append({
            "id": rec["id"],
            "label": rec["label"],
            "paraphrase_type": rec.get("paraphrase_type"),
            "source_doc_id": true_id,
            "true_source_rank": rank,
            "true_source_score": round(src_score, 6) if src_score is not None else None,
            "top1_score": round(top1_score, 6),
            "top1_doc_id": top1_doc_id,
        })

    # ── Per-query table ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("PER-QUERY RESULTS  (grouped by label)")
    print("=" * 72)
    hdr = f"{'id':<22} {'label':<22} {'rank':>5}  {'src_score':>9}  {'top1_score':>10}  paraphrase_type"
    print(hdr)
    print("-" * 80)

    for lbl in LABEL_ORDER:
        label_rows = [r for r in rows if r["label"] == lbl]
        if not label_rows:
            continue
        for r in label_rows:
            rank_str = str(r["true_source_rank"]) if r["true_source_rank"] is not None else "—"
            score_str = f"{r['true_source_score']:.4f}" if r["true_source_score"] is not None else "—"
            para = r["paraphrase_type"] or ""
            print(
                f"{r['id']:<22} {r['label']:<22} {rank_str:>5}  {score_str:>9}  "
                f"{r['top1_score']:>10.4f}  {para}"
            )
        print()

    # ── Per-category rollup ──────────────────────────────────────────────────
    print("=" * 72)
    print("PER-CATEGORY ROLLUP")
    print("=" * 72)

    rollup: dict[str, dict] = {}

    for lbl in ["verbatim", "light_edit", "paraphrase"]:
        label_rows = [r for r in rows if r["label"] == lbl]
        ranks = [r["true_source_rank"] for r in label_rows if r["true_source_rank"] is not None]
        scores = [r["true_source_score"] for r in label_rows if r["true_source_score"] is not None]
        not_found = sum(1 for r in label_rows if r["true_source_rank"] is None)
        nf_note = f"  ({not_found} not found)" if not_found else ""
        mean_rank_s = f"{mean(ranks):.2f}" if ranks else "n/a"
        worst_rank_s = str(max(ranks)) if ranks else "n/a"
        mean_score_s = f"{mean(scores):.4f}" if scores else "n/a"
        print(f"  {lbl:<22}  mean_rank={mean_rank_s}  worst_rank={worst_rank_s}  mean_score={mean_score_s}{nf_note}")
        rollup[lbl] = {
            "n": len(label_rows),
            "mean_rank": round(mean(ranks), 2) if ranks else None,
            "worst_rank": max(ranks) if ranks else None,
            "mean_score": round(mean(scores), 4) if scores else None,
            "not_found": not_found,
        }

    print()
    for lbl in ["negative_offtopic", "negative_same_topic"]:
        label_rows = [r for r in rows if r["label"] == lbl]
        top1s = [r["top1_score"] for r in label_rows]
        mean_top1_s = f"{mean(top1s):.4f}" if top1s else "n/a"
        max_top1_s = f"{max(top1s):.4f}" if top1s else "n/a"
        print(f"  {lbl:<22}  mean_top1={mean_top1_s}  max_top1={max_top1_s}")
        rollup[lbl] = {
            "n": len(label_rows),
            "mean_top1_score": round(mean(top1s), 4) if top1s else None,
            "max_top1_score": round(max(top1s), 4) if top1s else None,
        }

    # ── P/R/F1 ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"P/R/F1  (threshold: top1_score >= {FLAG_THRESHOLD})")
    print("CAVEAT: N=16 is too small to tune to. Rank is the primary signal.")
    print(f"        Threshold {FLAG_THRESHOLD} was chosen arbitrarily — NOT optimised.")
    print("=" * 72)

    tp = sum(1 for r in rows if r["label"] in POSITIVE_LABELS and r["top1_score"] >= FLAG_THRESHOLD)
    fp = sum(1 for r in rows if r["label"] in NEGATIVE_LABELS and r["top1_score"] >= FLAG_THRESHOLD)
    fn = sum(1 for r in rows if r["label"] in POSITIVE_LABELS and r["top1_score"] < FLAG_THRESHOLD)
    tn = sum(1 for r in rows if r["label"] in NEGATIVE_LABELS and r["top1_score"] < FLAG_THRESHOLD)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # ── Write artifact ───────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry_point": "plagdetect.detect.detect(query_text, corpus, k=5, top_k=len(corpus))",
            "normalize_text_called_on_query": False,
            "pipeline_note": (
                "normalize_text() runs at corpus-build time (stored in corpus 'canonical' field). "
                "Query path: shingle() → tokenize() + canonical() (NFC+lowercase per token). "
                "/check follows the same path."
            ),
            "flag_threshold": FLAG_THRESHOLD,
            "corpus_path": str(_CORPUS_PATH.relative_to(_ROOT)),
            "corpus_doc_id_field": "id",
            "fixture_path": str(_FIXTURE.relative_to(_ROOT)),
            "corpus_size": len(corpus),
            "fixture_size": len(records),
            "label_counts": label_counts,
            "phase": "BEFORE — lexical baseline; embeddings layer not yet built",
        },
        "per_query": rows,
        "rollup": rollup,
        "prf1": {
            "threshold": FLAG_THRESHOLD,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
    }

    out_path = _RESULTS_DIR / "baseline_lexical.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\nResults written → eval/results/baseline_lexical.json")
    print("This is the BEFORE baseline. Post-embeddings run produces the comparable AFTER.")


if __name__ == "__main__":
    main()
