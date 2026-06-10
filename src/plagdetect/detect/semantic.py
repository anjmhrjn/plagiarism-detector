"""Semantic retrieval channel: FAISS flat index over corpus canonical chunks."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from plagdetect.ingest.normalize import normalize_text

# Bump NORM_VERSION whenever normalize_text() changes; the index asserts this
# matches at load time so a stale index is caught immediately rather than
# silently producing wrong scores.
NORM_VERSION = "1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_DEFAULT_INDEX_DIR = Path("data/semantic_index")
_CHUNK_WORDS = 200
_CHUNK_STRIDE = 100


def _word_positions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for each whitespace-delimited token."""
    positions: list[tuple[int, int]] = []
    pos = 0
    for word in text.split():
        idx = text.index(word, pos)
        positions.append((idx, idx + len(word)))
        pos = idx + len(word)
    return positions


def _chunk_doc(doc_id: str, canonical: str) -> list[dict]:
    """
    Sliding-window chunks (~200 words, stride 100) of canonical text.

    Chunk offsets are char positions within the canonical string.
    The embedded text for each chunk IS the canonical slice (already
    normalized — do not re-run normalize_text()).
    """
    words = canonical.split()
    if not words:
        return []
    wpos = _word_positions(canonical)
    chunks: list[dict] = []
    start_w = 0
    while start_w < len(words):
        end_w = min(start_w + _CHUNK_WORDS, len(words))
        char_start = wpos[start_w][0]
        char_end = wpos[end_w - 1][1]
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_idx": len(chunks),
                "char_start": char_start,
                "char_end": char_end,
                "text": canonical[char_start:char_end],
            }
        )
        if end_w == len(words):
            break
        start_w += _CHUNK_STRIDE
    return chunks


def build_index(corpus: list[dict], index_dir: Path = _DEFAULT_INDEX_DIR) -> None:
    """
    Build a FAISS IndexFlatIP over all corpus canonical chunks and persist.

    Corpus chunks are embedded with normalize_embeddings=True so that
    IndexFlatIP inner-product == cosine similarity. The index, chunk
    metadata (JSONL), and a meta.json (model_name + norm_version) are
    written to index_dir. Calling this again overwrites the previous index.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    all_chunks: list[dict] = []
    for doc in corpus:
        all_chunks.extend(_chunk_doc(doc["id"], doc["canonical"]))

    texts = [c["text"] for c in all_chunks]
    print(f"  Encoding {len(texts)} chunks  model={MODEL_NAME}", flush=True)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "faiss.idx"))

    chunk_meta = [{k: v for k, v in c.items() if k != "text"} for c in all_chunks]
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(m) for m in chunk_meta),
        encoding="utf-8",
    )
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "norm_version": NORM_VERSION,
                "n_chunks": len(all_chunks),
                "dim": dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Index saved → {index_dir}  ({len(all_chunks)} chunks, dim={dim})")


class SemanticIndex:
    """Loaded index for query-time retrieval. Instantiate via load_index()."""

    def __init__(self, index_dir: Path = _DEFAULT_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["model_name"] != MODEL_NAME:
            raise RuntimeError(
                f"Model mismatch: index={meta['model_name']!r}  code={MODEL_NAME!r}"
            )
        if meta["norm_version"] != NORM_VERSION:
            raise RuntimeError(
                f"NORM_VERSION mismatch: index={meta['norm_version']!r}  "
                f"code={NORM_VERSION!r} — rebuild the index."
            )
        self._model = SentenceTransformer(MODEL_NAME)
        self._index: faiss.IndexFlatIP = faiss.read_index(str(index_dir / "faiss.idx"))
        self._chunks: list[dict] = []
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._chunks.append(json.loads(line))

    @property
    def n_chunks(self) -> int:
        return self._index.ntotal

    def query(self, raw_query: str, top_k: int = 10) -> list[dict]:
        """
        normalize_text(raw_query) → embed → FAISS exact search over ALL chunks →
        dedupe by doc_id keeping best score → return top_k docs sorted desc.

        Scans every vector (ntotal) so every corpus doc is guaranteed a score.
        Returns [{doc_id, semantic_score, chunk_idx, char_start, char_end}, ...].
        """
        normalized = normalize_text(raw_query)
        embedding = self._model.encode(
            [normalized],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Full exact scan — IndexFlatIP is O(n*d) regardless of k.
        k_chunks = self._index.ntotal
        scores, indices = self._index.search(embedding, k_chunks)

        best: dict[str, dict] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            doc_id = chunk["doc_id"]
            if doc_id not in best or float(score) > best[doc_id]["semantic_score"]:
                best[doc_id] = {
                    "doc_id": doc_id,
                    "semantic_score": float(score),
                    "chunk_idx": chunk["chunk_idx"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                }

        return sorted(best.values(), key=lambda x: x["semantic_score"], reverse=True)[
            :top_k
        ]


def load_index(index_dir: Path = _DEFAULT_INDEX_DIR) -> SemanticIndex:
    return SemanticIndex(index_dir)
