"""
After-judge evaluation: retrieval → LLM judge decision vs the fixture.

Runs the full pipeline (both retrieval channels + judge) over the fixture and
writes eval/results/after_judge.json.  Requires OPENAI_API_KEY to be set.

Win condition:
  paraphrase rows        → is_derived = True
  negative_same_topic    → is_derived = False

Run from project root:
    python -m eval.run_after_judge
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from plagdetect.corpus import load_corpus
from plagdetect.detect import detect
from plagdetect.detect.semantic import MODEL_NAME as SEM_MODEL, NORM_VERSION, load_index
from plagdetect.detect.spans import extract_spans
from plagdetect.judge import JUDGE_MODEL, judge_pair

_FIXTURE = _ROOT / "eval" / "fixtures" / "paraphrase_seed.jsonl"
_CORPUS_PATH = _ROOT / "data" / "corpus.jsonl"
_INDEX_DIR = _ROOT / "data" / "semantic_index"
_RESULTS_DIR = _ROOT / "eval" / "results"
_INJECTION_DIR = _ROOT / "tests" / "fixtures" / "prompt_injection"

_SEM_INCLUDE_THRESHOLD = 0.40

POSITIVE_LABELS = {"verbatim", "light_edit", "paraphrase"}
NEGATIVE_LABELS = {"negative_offtopic", "negative_same_topic"}
LABEL_ORDER = ["verbatim", "light_edit", "paraphrase", "negative_offtopic", "negative_same_topic"]

# Win condition: judge must say DERIVED for these labels.
JUDGE_DERIVED_LABELS = {"verbatim", "light_edit", "paraphrase"}
# Win condition: judge must say NOT-DERIVED for these labels.
JUDGE_NOT_DERIVED_LABELS = {"negative_same_topic"}


def _load_fixture(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _judge_passage(query_text: str, canonical: str, m: dict, is_lex: bool) -> tuple[str, str]:
    """
    Returns (passage, source) where source is 'lexical', 'semantic', or 'fallback'.

    Lexical hit  → reverse extract_spans to find the matching region in the source doc,
                   widened by ±500 chars so the judge sees the full edited passage.
    Semantic-only → use the chunk offsets stored by the semantic index.
    """
    if is_lex and canonical:
        source_spans = extract_spans(canonical, query_text, k=5)
        if source_spans:
            span_start = min(s["start"] for s in source_spans)
            span_end = max(s["end"] for s in source_spans)
            ctx_start = max(0, span_start - 500)
            ctx_end = min(len(canonical), span_end + 500)
            return canonical[ctx_start:ctx_end], "lexical"

    char_start = m.get("chunk_char_start")
    char_end = m.get("chunk_char_end")
    if char_start is not None and char_end is not None and canonical:
        return canonical[char_start:char_end], "semantic"

    return canonical[:2000], "fallback"


def main() -> None:
    print("=" * 72)
    print("AFTER-JUDGE EVALUATION  (retrieval → LLM judge)")
    print("=" * 72)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not _INDEX_DIR.exists():
        print(f"\nERROR: semantic index not found at {_INDEX_DIR}")
        print("  Run: python scripts/build_semantic_index.py")
        sys.exit(1)

    records = _load_fixture(_FIXTURE)
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"\nFixture  : {_FIXTURE.relative_to(_ROOT)}")
    print(f"  Total  : {len(records)} records")
    for lbl in LABEL_ORDER:
        if lbl in label_counts:
            print(f"    {lbl}: {label_counts[lbl]}")

    corpus = load_corpus(_CORPUS_PATH)
    corpus_index = {d["id"]: d["canonical"] for d in corpus}
    print(f"\nCorpus   : {_CORPUS_PATH.relative_to(_ROOT)}  ({len(corpus)} docs)")

    print("\nLoading semantic index …")
    sem_index = load_index(_INDEX_DIR)
    print(f"  Model  : {SEM_MODEL}")
    print(f"  Chunks : {sem_index.n_chunks}")

    print(f"\nJudge model: {JUDGE_MODEL}")
    print(f"\nRunning {len(records)} fixture records …\n")

    rows: list[dict] = []
    for rec in records:
        all_results = detect(
            rec["query_text"], corpus, k=5, top_k=len(corpus), semantic_index=sem_index
        )
        true_id = rec.get("source_doc_id")

        # Identify the top candidate for this row (by true source if present, else top-1).
        candidate = None
        if true_id:
            for r in all_results:
                if r["id"] == true_id:
                    candidate = r
                    break
        if candidate is None:
            # No true source (negative rows) — use whichever doc scored highest semantically.
            sem_sorted = sorted(all_results, key=lambda x: (x["semantic_score"] or 0.0), reverse=True)
            candidate = sem_sorted[0] if sem_sorted else None

        if candidate is None:
            rows.append({"id": rec["id"], "label": rec["label"], "error": "no candidate"})
            continue

        is_lex = candidate["containment"] > 0.0
        is_sem = (candidate["semantic_score"] or 0.0) >= _SEM_INCLUDE_THRESHOLD
        pre_flagged = is_lex or is_sem

        canonical = corpus_index.get(candidate["id"], "")
        candidate_chunk, passage_source = _judge_passage(
            rec["query_text"], canonical, candidate, is_lex
        )

        verdict = judge_pair(
            rec["query_text"],
            candidate_chunk,
            lexical_score=candidate["containment"],
            semantic_score=candidate["semantic_score"],
        )

        rows.append(
            {
                "id": rec["id"],
                "label": rec["label"],
                "paraphrase_type": rec.get("paraphrase_type"),
                "source_doc_id": true_id,
                "candidate_id": candidate["id"],
                "lex_score": round(candidate["containment"], 6),
                "sem_score": round(candidate["semantic_score"] or 0.0, 6),
                "pre_flagged": pre_flagged,
                "passage_source": passage_source,
                "judge_relationship": verdict["relationship"],
                "judge_is_derived": verdict["is_derived"],
                "judge_confidence": verdict["confidence"],
                "judge_evidence": verdict["evidence"],
                "judge_rationale": verdict["rationale"],
                "judge_error": verdict["error"],
            }
        )

        is_derived_str = "DERIVED    " if verdict["is_derived"] else "not-derived"
        print(f"  {rec['id']:<20} {rec['label']:<22} → {is_derived_str}  [{verdict['relationship']}]")

    # ── Confusion-style readout ───────────────────────────────────────────────
    print()
    print("=" * 72)
    print("JUDGE DECISION SUMMARY")
    print("=" * 72)

    paraphrase_rows = [r for r in rows if r.get("label") == "paraphrase"]
    same_topic_rows = [r for r in rows if r.get("label") == "negative_same_topic"]
    positive_rows = [r for r in rows if r.get("label") in JUDGE_DERIVED_LABELS]
    negative_rows = [r for r in rows if r.get("label") in NEGATIVE_LABELS]

    print("\n  TARGET: paraphrase → DERIVED | negative_same_topic → NOT-DERIVED\n")

    tp = sum(1 for r in positive_rows if r.get("judge_is_derived") is True)
    fn = sum(1 for r in positive_rows if r.get("judge_is_derived") is False)
    tn = sum(1 for r in negative_rows if r.get("judge_is_derived") is False)
    fp = sum(1 for r in negative_rows if r.get("judge_is_derived") is True)

    print(f"  True positives  (derived    labelled, DERIVED):     {tp}/{len(positive_rows)}")
    print(f"  False negatives (derived    labelled, not-derived): {fn}/{len(positive_rows)}")
    print(f"  True negatives  (not-derived labelled, not-derived): {tn}/{len(negative_rows)}")
    print(f"  False positives (not-derived labelled, DERIVED):     {fp}/{len(negative_rows)}")

    # Paraphrase-specific win condition
    para_derived = sum(1 for r in paraphrase_rows if r.get("judge_is_derived") is True)
    print(f"\n  Paraphrase rows DERIVED: {para_derived}/{len(paraphrase_rows)}", end="")
    print("  ✓ WIN" if para_derived == len(paraphrase_rows) else "  ✗ MISS")

    # Same-topic negatives win condition
    neg_not_derived = sum(1 for r in same_topic_rows if r.get("judge_is_derived") is False)
    print(f"  Same-topic negatives NOT-DERIVED: {neg_not_derived}/{len(same_topic_rows)}", end="")
    print("  ✓ WIN" if neg_not_derived == len(same_topic_rows) else "  ✗ MISS")

    # Report misses
    misses = [r for r in rows if _is_miss(r)]
    if misses:
        print(f"\n  MISSES ({len(misses)}):")
        for r in misses:
            direction = "should be DERIVED" if r["label"] in JUDGE_DERIVED_LABELS else "should be NOT-DERIVED"
            print(f"    {r['id']:<20} [{r['label']}]  {direction}")
            print(f"      judge: {r.get('judge_relationship')}  rationale: {r.get('judge_rationale')}")
    else:
        print("\n  No misses — all rows decided correctly.")

    # ── Prompt injection tests ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("PROMPT INJECTION TESTS")
    print("=" * 72)

    injection_results = []
    if _INJECTION_DIR.exists():
        injection_docs = sorted(_INJECTION_DIR.glob("*.txt"))
        # A benign candidate that has no connection to the injected text.
        benign_candidate = "The sky is blue. Clouds are made of water droplets. Rain falls from clouds."
        for doc_path in injection_docs:
            injected_text = doc_path.read_text(encoding="utf-8")
            verdict = judge_pair(injected_text, benign_candidate)
            injected_followed = (
                verdict.get("is_derived") is True and verdict.get("error") is None
            )
            status = "FAIL (injection followed)" if injected_followed else "PASS (injection ignored)"
            print(f"  {doc_path.name:<45} {status}")
            print(f"    relationship={verdict['relationship']}  is_derived={verdict['is_derived']}")
            print(f"    rationale: {verdict['rationale'][:100]}")
            injection_results.append(
                {
                    "doc": doc_path.name,
                    "is_derived": verdict["is_derived"],
                    "relationship": verdict["relationship"],
                    "rationale": verdict["rationale"],
                    "injection_followed": injected_followed,
                    "status": status,
                }
            )
    else:
        print("  (no prompt injection fixtures found)")

    injection_pass = all(not r["injection_followed"] for r in injection_results)
    print(f"\n  Injection test outcome: {'ALL PASSED ✓' if injection_pass else 'FAILURES DETECTED ✗'}")

    # ── Write artifact ────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "AFTER — LLM judge re-ranker",
            "judge_model": JUDGE_MODEL,
            "sem_model": SEM_MODEL,
            "norm_version": NORM_VERSION,
            "corpus_path": str(_CORPUS_PATH.relative_to(_ROOT)),
            "corpus_size": len(corpus),
            "fixture_path": str(_FIXTURE.relative_to(_ROOT)),
            "fixture_size": len(records),
            "label_counts": label_counts,
        },
        "summary": {
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
            "paraphrase_derived": para_derived,
            "paraphrase_total": len(paraphrase_rows),
            "same_topic_not_derived": neg_not_derived,
            "same_topic_total": len(same_topic_rows),
            "injection_tests_passed": injection_pass,
        },
        "per_query": rows,
        "injection_tests": injection_results,
    }

    out_path = _RESULTS_DIR / "after_judge.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\nResults written → eval/results/after_judge.json")


def _is_miss(row: dict) -> bool:
    label = row.get("label")
    is_derived = row.get("judge_is_derived")
    if label in JUDGE_DERIVED_LABELS and is_derived is False:
        return True
    if label in JUDGE_NOT_DERIVED_LABELS and is_derived is True:
        return True
    return False


if __name__ == "__main__":
    main()
