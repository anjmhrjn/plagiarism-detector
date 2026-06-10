"""
Build (or rebuild) the semantic FAISS index from the corpus.

Run from the project root:
    python scripts/build_semantic_index.py

The index is written to data/semantic_index/ and loaded automatically
by the app at startup. Re-running overwrites the previous index.

WARNING: takes several minutes on CPU (~163 k chunks at 200 words/chunk,
stride 100, for the 100-doc Gutenberg corpus).
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from plagdetect.corpus import load_corpus
from plagdetect.detect.semantic import MODEL_NAME, NORM_VERSION, build_index

_CORPUS_PATH = _ROOT / "data" / "corpus.jsonl"
_INDEX_DIR = _ROOT / "data" / "semantic_index"


def main() -> None:
    print("=" * 60)
    print("SEMANTIC INDEX BUILD")
    print("=" * 60)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  NORM_VERSION: {NORM_VERSION}")
    print(f"  Corpus      : {_CORPUS_PATH.relative_to(_ROOT)}")
    print(f"  Output      : {_INDEX_DIR.relative_to(_ROOT)}")
    print()

    corpus = load_corpus(_CORPUS_PATH)
    print(f"  Loaded {len(corpus)} corpus documents")

    t0 = time.time()
    build_index(corpus, index_dir=_INDEX_DIR)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Index → {_INDEX_DIR.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
