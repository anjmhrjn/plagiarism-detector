def containment(a: set, b: set) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
