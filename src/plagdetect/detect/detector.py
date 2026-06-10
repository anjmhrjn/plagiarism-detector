from __future__ import annotations

from .shingles import shingle
from .similarity import containment, jaccard


def detect(
    query_text: str,
    corpus: list[dict],
    k: int = 5,
    top_k: int = 5,
    semantic_index=None,
) -> list[dict]:
    """
    Two-channel plagiarism detector.  Entry point shared by /check and the
    eval harness.

    LEXICAL CHANNEL (always runs, unchanged):
        n-gram shingling → containment (primary score) + Jaccard.

    SEMANTIC CHANNEL (runs when semantic_index is not None):
        SemanticIndex.query() → cosine similarity per chunk → best score per
        source_doc_id.  Results are UNIONed with the lexical results by
        source_doc_id; both scores are attached to every candidate.
        No score fusion: scores remain independent.

    Sort: containment descending (lexical primary).  Returns top_k results.
    Every result carries 'semantic_score' (float | None).
    """
    query_shingles = shingle(query_text, k)
    by_id: dict[str, dict] = {}
    for doc in corpus:
        doc_shingles = shingle(doc["canonical"], k)
        c = containment(query_shingles, doc_shingles)
        j = jaccard(query_shingles, doc_shingles)
        by_id[doc["id"]] = {
            "id": doc["id"],
            "title": doc["title"],
            "containment": c,
            "jaccard": j,
            "semantic_score": None,
        }

    if semantic_index is not None:
        for r in semantic_index.query(raw_query=query_text, top_k=len(corpus)):
            if r["doc_id"] in by_id:
                by_id[r["doc_id"]]["semantic_score"] = r["semantic_score"]

    results = list(by_id.values())
    results.sort(key=lambda x: x["containment"], reverse=True)
    return results[:top_k]


def _cli() -> None:
    import argparse
    import sys
    from pathlib import Path

    from plagdetect.corpus import load_corpus

    parser = argparse.ArgumentParser(description="N-gram plagiarism matcher")
    parser.add_argument("file", nargs="?", help="Input text file (default: stdin)")
    parser.add_argument("--corpus", default="data/corpus.jsonl", help="Corpus JSONL path")
    parser.add_argument("-k", type=int, default=5, help="Shingle size")
    parser.add_argument("--top", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    query = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    corpus = load_corpus(args.corpus)
    matches = detect(query, corpus, k=args.k, top_k=args.top)

    if not matches:
        print("No corpus documents loaded.")
        sys.exit(1)

    print(f"{'containment':>11}  {'jaccard':>7}  {'id':<30}  title")
    print("-" * 80)
    for m in matches:
        print(f"{m['containment']:>11.3f}  {m['jaccard']:>7.3f}  {m['id']:<30}  {m['title']}")


if __name__ == "__main__":
    _cli()
