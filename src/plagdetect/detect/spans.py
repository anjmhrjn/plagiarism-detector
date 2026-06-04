from .tokens import Token, tokenize, canonical


def extract_spans(query_text: str, source_canonical: str, k: int = 5) -> list[dict]:
    """
    Find every span in query_text whose canonical shingles appear in source_canonical.

    Returns a list of {start, end, text} dicts where start/end are char offsets
    into query_text's ORIGINAL string (preserving original casing, spacing, etc.).
    Overlapping/adjacent matched token positions are merged into maximal spans.
    """
    query_tokens = tokenize(query_text)
    if len(query_tokens) < k:
        return []

    source_tokens = tokenize(source_canonical)
    if len(source_tokens) < k:
        return []

    source_shingles: set[str] = {
        " ".join(canonical(source_tokens[i + j]) for j in range(k))
        for i in range(len(source_tokens) - k + 1)
    }

    matched: set[int] = set()
    for i in range(len(query_tokens) - k + 1):
        gram = " ".join(canonical(query_tokens[i + j]) for j in range(k))
        if gram in source_shingles:
            matched.update(range(i, i + k))

    if not matched:
        return []

    sorted_pos = sorted(matched)
    spans: list[dict] = []
    run = [sorted_pos[0]]
    for pos in sorted_pos[1:]:
        if pos == run[-1] + 1:
            run.append(pos)
        else:
            spans.append(_span_from_run(query_text, query_tokens, run))
            run = [pos]
    spans.append(_span_from_run(query_text, query_tokens, run))
    return spans


def _span_from_run(text: str, tokens: list[Token], run: list[int]) -> dict:
    start = tokens[run[0]].start
    end = tokens[run[-1]].end
    return {"start": start, "end": end, "text": text[start:end]}
