import json
from pathlib import Path


def load_corpus(path: str | Path) -> list[dict]:
    """Load a JSONL corpus file and return all records as a list of dicts."""
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
