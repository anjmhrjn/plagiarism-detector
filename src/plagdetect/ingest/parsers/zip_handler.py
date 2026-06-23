"""
ZIP handler with security guards:

  - Zip-bomb: cap total uncompressed bytes (50 MB) and per-entry
    compression ratio (100×). Checked BEFORE extracting entry data.
  - Zip Slip: every entry path is resolved inside a temp dir; paths that
    escape the sandbox are rejected immediately.
  - Depth: max nesting level is ZIP_MAX_DEPTH (3). Deeper zips are rejected
    with a clear error unit — we never recurse past the cap.

Extracted entries re-enter ingest.router.ingest(), which re-dispatches each
by its own type. The zip handler knows nothing about other formats.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path

from plagdetect.ingest.router import IngestionUnit

_log = logging.getLogger(__name__)

ZIP_MAX_DEPTH        = 3        # max recursive zip nesting
ZIP_MAX_TOTAL_BYTES  = 50 * 1024 * 1024   # 50 MB total uncompressed
ZIP_MAX_RATIO        = 100      # max compression ratio per entry (compressed → uncompressed)
ZIP_MAX_ENTRY_BYTES  = 20 * 1024 * 1024   # 20 MB per single entry


def parse_zip(
    data: bytes,
    source_path: str,
    parent_prefix: str,
    depth: int,
) -> list[IngestionUnit]:
    if depth >= ZIP_MAX_DEPTH:
        return [IngestionUnit(
            text="",
            source_path=source_path,
            file_type="zip",
            notes=f"Rejected: zip nesting depth {depth} exceeds cap ({ZIP_MAX_DEPTH}).",
        )]

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        return [IngestionUnit(
            text="",
            source_path=source_path,
            file_type="zip",
            notes=f"Rejected: corrupt zip — {exc}",
        )]

    units: list[IngestionUnit] = []
    total_uncompressed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir).resolve()

        for info in zf.infolist():
            # Skip directory entries.
            if info.filename.endswith("/"):
                continue

            # --- Zip Slip guard -------------------------------------------
            safe = _safe_path(tmp_root, info.filename)
            if safe is None:
                units.append(IngestionUnit(
                    text="",
                    source_path=parent_prefix + info.filename,
                    file_type="zip",
                    notes=f"Rejected (Zip Slip): entry '{info.filename}' escapes archive sandbox.",
                ))
                continue

            # --- Bomb guard: ratio check (compress_size may be 0 for stored) ---
            compress_size = info.compress_size or 1  # avoid div-by-zero
            uncompressed  = info.file_size
            if uncompressed > ZIP_MAX_ENTRY_BYTES:
                units.append(IngestionUnit(
                    text="",
                    source_path=parent_prefix + info.filename,
                    file_type="zip",
                    notes=(
                        f"Rejected (zip-bomb): entry uncompressed size "
                        f"{uncompressed:,} B exceeds {ZIP_MAX_ENTRY_BYTES:,} B cap."
                    ),
                ))
                continue

            ratio = uncompressed / compress_size
            if ratio > ZIP_MAX_RATIO:
                units.append(IngestionUnit(
                    text="",
                    source_path=parent_prefix + info.filename,
                    file_type="zip",
                    notes=(
                        f"Rejected (zip-bomb): entry '{info.filename}' compression "
                        f"ratio {ratio:.0f}× exceeds {ZIP_MAX_RATIO}× cap."
                    ),
                ))
                continue

            total_uncompressed += uncompressed
            if total_uncompressed > ZIP_MAX_TOTAL_BYTES:
                units.append(IngestionUnit(
                    text="",
                    source_path=parent_prefix + info.filename,
                    file_type="zip",
                    notes=(
                        f"Rejected (zip-bomb): total uncompressed content "
                        f"exceeds {ZIP_MAX_TOTAL_BYTES // (1024*1024)} MB cap."
                    ),
                ))
                # Stop processing further entries.
                break

            # --- Extract to sandboxed path -----------------------------------
            safe.parent.mkdir(parents=True, exist_ok=True)
            entry_data = zf.read(info.filename)

            # Re-enter the router for each extracted entry.
            from plagdetect.ingest.router import ingest as _ingest
            child_units = _ingest(
                entry_data,
                info.filename,
                _archive_prefix=parent_prefix,
                _depth=depth + 1,
            )
            units.extend(child_units)

    return units


def _safe_path(sandbox: Path, entry_name: str) -> Path | None:
    """
    Resolve the entry path inside sandbox. Return None if it escapes.

    Rejects:
      - Absolute paths (e.g. /etc/passwd)
      - Paths with .. components that escape the sandbox
      - Any OS-level symlink traversal (resolved after join)
    """
    # Normalise to POSIX separators and strip leading slashes/dots
    # before joining so Path(sandbox) / Path(entry_name) doesn't treat
    # a leading "/" as filesystem root.
    clean = entry_name.replace("\\", "/").lstrip("/")
    candidate = (sandbox / clean).resolve()
    try:
        candidate.relative_to(sandbox)
        return candidate
    except ValueError:
        return None
