import re

def shingle(text: str, k: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", text)
    if len(tokens) < k:
        return set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}
