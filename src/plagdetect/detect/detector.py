from .shingles import shingle
from .similarity import containment, jaccard


def detect(
    query_text: str,
    corpus: list[dict],
    k: int = 5,
    top_k: int = 5,
) -> list[dict]:
    query_shingles = shingle(query_text, k)
    results = []
    for doc in corpus:
        doc_shingles = shingle(doc["canonical"], k)
        c = containment(query_shingles, doc_shingles)
        j = jaccard(query_shingles, doc_shingles)
        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "containment": c,
                "jaccard": j,
            }
        )
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
