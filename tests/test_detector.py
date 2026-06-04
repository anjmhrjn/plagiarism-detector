import pytest
from plagdetect.detect import detect
from plagdetect.detect.shingles import shingle
from plagdetect.detect.similarity import containment, jaccard
from plagdetect.ingest.normalize import normalize_text

# ---------------------------------------------------------------------------
# Small deterministic corpus — three short, thematically distinct passages
# ---------------------------------------------------------------------------

_RAW = {
    "doc1": (
        "gutenberg:1",
        "Fox and Dog",
        "the quick brown fox jumps over the lazy dog near the river bank",
    ),
    "doc2": (
        "gutenberg:2",
        "To Be",
        "to be or not to be that is the question worth pondering every single day",
    ),
    "doc3": (
        "gutenberg:3",
        "Fourscore",
        "four score and seven years ago our fathers brought forth on this continent a new nation",
    ),
}

CORPUS = [
    {"id": id_, "title": title, "canonical": normalize_text(raw)}
    for id_, title, raw in _RAW.values()
]

# Convenience: canonical text for doc1
_DOC1_CANONICAL = normalize_text(_RAW["doc1"][2])


# ---------------------------------------------------------------------------
# Unit tests for shingle()
# ---------------------------------------------------------------------------

class TestShingle:
    def test_basic_five_gram(self):
        s = shingle("a b c d e f", k=5)
        assert "a b c d e" in s
        assert "b c d e f" in s
        assert len(s) == 2

    def test_punctuation_as_separator(self):
        s = shingle("hello, world. foo bar baz", k=3)
        assert "hello world foo" in s

    def test_too_short_returns_empty(self):
        assert shingle("one two three", k=5) == set()

    def test_exact_length(self):
        s = shingle("a b c d e", k=5)
        assert s == {"a b c d e"}

    def test_returns_set(self):
        # duplicate sub-sequences collapsed
        s = shingle("a a a a a a", k=5)
        assert isinstance(s, set)
        assert s == {"a a a a a"}


# ---------------------------------------------------------------------------
# Unit tests for containment() and jaccard()
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_sets(self):
        s = {"a b c", "b c d"}
        assert containment(s, s) == 1.0
        assert jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert containment({"x y z"}, {"a b c"}) == 0.0
        assert jaccard({"x y z"}, {"a b c"}) == 0.0

    def test_empty_a_returns_zero(self):
        assert containment(set(), {"a b c"}) == 0.0
        assert jaccard(set(), set()) == 0.0

    def test_containment_subset(self):
        a = {"a b c"}
        b = {"a b c", "b c d", "c d e"}
        # a fully contained in b
        assert containment(a, b) == 1.0
        # jaccard = 1/3
        assert jaccard(a, b) == pytest.approx(1 / 3)

    def test_containment_superset(self):
        a = {"a b c", "b c d", "c d e"}
        b = {"a b c"}
        # only 1 of 3 shingles in b
        assert containment(a, b) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Integration tests for detect()
# ---------------------------------------------------------------------------

class TestDetect:
    def test_verbatim_chunk_ranks_first_with_high_containment(self):
        """A verbatim lift from doc1 should rank doc1 #1 with containment near 1.0."""
        # Take the full canonical text of doc1 — containment must be 1.0.
        results = detect(_DOC1_CANONICAL, CORPUS, k=5, top_k=3)
        assert results[0]["id"] == "gutenberg:1"
        assert results[0]["containment"] == pytest.approx(1.0)

    def test_partial_verbatim_chunk_ranks_first(self):
        """A 10-word verbatim chunk still ranks doc1 first."""
        # "the quick brown fox jumps over the lazy dog near" — 10 words, 6 shingles
        chunk = "the quick brown fox jumps over the lazy dog near"
        results = detect(chunk, CORPUS, k=5, top_k=3)
        assert results[0]["id"] == "gutenberg:1"
        assert results[0]["containment"] > 0.9

    def test_clearly_unrelated_query_scores_low(self):
        """Completely unrelated content produces near-zero containment for all docs."""
        unrelated = (
            "machine learning algorithms optimize neural network weights "
            "during gradient descent backpropagation training epochs"
        )
        results = detect(unrelated, CORPUS, k=5, top_k=3)
        for r in results:
            assert r["containment"] < 0.1, f"{r['id']} scored {r['containment']:.3f}"

    def test_paraphrase_scores_low(self):
        """A paraphrase of doc1 using different words scores low.

        This is a known limitation: n-gram matching requires lexical overlap.
        The embedding layer (Day 4) will catch paraphrases that slip through here.
        """
        paraphrase = (
            "a speedy auburn canine leaps above the sluggish hound beside the stream"
        )
        results = detect(paraphrase, CORPUS, k=5, top_k=3)
        top = results[0]
        assert top["containment"] < 0.2, (
            f"Paraphrase unexpectedly scored {top['containment']:.3f} — "
            "if this is intentional, update the embedding layer motivation comment."
        )

    def test_top_k_limits_results(self):
        results = detect(_DOC1_CANONICAL, CORPUS, k=5, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_corpus(self):
        results = detect(_DOC1_CANONICAL, CORPUS, k=5, top_k=100)
        assert len(results) == len(CORPUS)

    def test_results_sorted_by_containment_descending(self):
        results = detect(_DOC1_CANONICAL, CORPUS, k=5, top_k=3)
        scores = [r["containment"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_shape(self):
        results = detect(_DOC1_CANONICAL, CORPUS, k=5, top_k=1)
        assert len(results) == 1
        r = results[0]
        assert set(r.keys()) == {"id", "title", "containment", "jaccard"}
        assert 0.0 <= r["containment"] <= 1.0
        assert 0.0 <= r["jaccard"] <= 1.0

    def test_empty_query_returns_zero_scores(self):
        results = detect("", CORPUS, k=5, top_k=3)
        for r in results:
            assert r["containment"] == 0.0
            assert r["jaccard"] == 0.0
