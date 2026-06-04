from plagdetect.detect.spans import extract_spans
from plagdetect.ingest.normalize import normalize_text

# Same doc1 used in test_detector.py — canonical is what the corpus stores.
_DOC1_RAW = "the quick brown fox jumps over the lazy dog near the river bank"
_DOC1_CANONICAL = normalize_text(_DOC1_RAW)


class TestExtractSpans:
    def test_verbatim_chunk_offsets_reproduce_original_text(self):
        """A verbatim lifted chunk produces one span whose offsets, sliced from the
        original query, equal the copied text exactly."""
        query = "the quick brown fox jumps over the lazy dog"
        spans = extract_spans(query, _DOC1_CANONICAL, k=5)
        assert len(spans) == 1
        s = spans[0]
        # Slicing the ORIGINAL query at the returned offsets must give back the chunk.
        assert query[s["start"] : s["end"]] == query
        assert s["text"] == query

    def test_mixed_case_matches_canonically_but_highlights_original(self):
        """Query in mixed case: canonical matching finds the overlap but the returned
        offsets point at the ORIGINAL (mixed-case) characters — not a lowercased copy."""
        query = "The QUICK Brown FOX Jumps over the lazy dog near"
        spans = extract_spans(query, _DOC1_CANONICAL, k=5)
        assert len(spans) >= 1
        for s in spans:
            # text field must equal the slice of the original string
            assert s["text"] == query[s["start"] : s["end"]]
        # At least some characters should be uppercase — confirming originals are preserved.
        combined = "".join(s["text"] for s in spans)
        assert combined != combined.lower(), (
            "Span text must reflect original casing, not the canonicalized form"
        )

    def test_extra_whitespace_matches_and_offsets_preserve_spaces(self):
        """Query with extra internal spaces: canonical tokens still match the corpus;
        the returned span's text includes the original extra whitespace."""
        query = "the  quick   brown fox jumps over the lazy dog"
        spans = extract_spans(query, _DOC1_CANONICAL, k=5)
        assert len(spans) >= 1
        s = spans[0]
        # text must be the exact original slice — extra spaces and all
        assert s["text"] == query[s["start"] : s["end"]]
        # The span should cover the whole query (all tokens match doc1)
        assert s["start"] == 0
        assert s["end"] == len(query)
        # Confirm the preserved text actually contains the original extra whitespace
        assert "  " in s["text"]

    def test_unrelated_query_returns_no_spans(self):
        """A query with no lexical overlap against the corpus doc returns an empty list."""
        query = (
            "machine learning algorithms optimize neural network weights "
            "during gradient descent backpropagation"
        )
        spans = extract_spans(query, _DOC1_CANONICAL, k=5)
        assert spans == []
