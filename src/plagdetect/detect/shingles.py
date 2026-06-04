from .tokens import tokenize, canonical


def shingle(text: str, k: int = 5) -> set[str]:
    tokens = tokenize(text)
    if len(tokens) < k:
        return set()
    return {
        " ".join(canonical(tokens[i + j]) for j in range(k))
        for i in range(len(tokens) - k + 1)
    }
