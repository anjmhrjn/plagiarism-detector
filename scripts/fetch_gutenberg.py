import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import urllib.request
import urllib.error

from plagdetect.ingest.normalize import normalize_text

GUTENDEX_SEARCH = "https://gutendex.com/books/?languages=en&mime_type=text/plain"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "plagdetect-corpus-fetcher/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _plain_text_url(book: dict) -> str | None:
    formats = book.get("formats", {})
    for mime in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if mime in formats:
            return formats[mime]
    # Fallback: any key containing text/plain
    for key, url in formats.items():
        if "text/plain" in key:
            return url
    return None


def _fetch_text(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "plagdetect-corpus-fetcher/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        # Gutenberg files are UTF-8 or Latin-1; try UTF-8 first
        for enc in ("utf-8", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    except (urllib.error.URLError, OSError) as exc:
        print(f"  skip ({exc})", file=sys.stderr)
    return None


def fetch(out_path: Path, limit: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    next_url: str | None = GUTENDEX_SEARCH

    with out_path.open("w", encoding="utf-8") as fh:
        while next_url and written < limit:
            print(f"[{written}/{limit}] fetching page: {next_url}", file=sys.stderr)
            page = _get_json(next_url)
            for book in page.get("results", []):
                if written >= limit:
                    break
                txt_url = _plain_text_url(book)
                if not txt_url:
                    continue
                print(f"  [{written+1}] {book['id']} {book['title'][:60]!r} …", file=sys.stderr)
                text = _fetch_text(txt_url)
                if not text:
                    continue
                record = {
                    "id": f"gutenberg:{book['id']}",
                    "source": "gutenberg",
                    "title": book.get("title", ""),
                    "url": txt_url,
                    "text": text,
                    "canonical": normalize_text(text),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                time.sleep(0.5)  # be polite to the Gutenberg servers

            next_url = page.get("next")
            if next_url:
                time.sleep(1)

    print(f"Done. {written} records written to {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Gutenberg corpus")
    parser.add_argument("--out", default="data/corpus.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    fetch(Path(args.out), args.limit)


if __name__ == "__main__":
    main()
