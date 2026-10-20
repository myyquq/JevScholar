"""Small shared helpers: hashing, verified downloads, JSON reads and atomic writes, sqlite,
vendored imports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jevscholar.paths import VENDOR

_MISSING = object()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url: str, dest: Path, sha256: str | None, *, timeout: float = 120) -> str:
    """Fetch `url` to `dest` unless a verified copy exists; returns the sha256.

    A blank `sha256` is trusted on first use (the caller decides whether to persist it)."""
    import requests

    if sha256 and dest.exists() and sha256_file(dest) == sha256:
        return sha256
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp = Path(f.name)
    digest = sha256_file(tmp)
    if sha256 and digest != sha256:
        tmp.unlink()
        raise ValueError(f"sha256 mismatch for {url}: got {digest}, pinned {sha256}")
    tmp.replace(dest)
    return digest


def read_json(path: Path, default: Any = _MISSING) -> Any:
    """A JSON file's content; `default`, if given, when there is no such file."""
    if default is not _MISSING and not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    """A JSONL file's records (blank lines skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """Write records as JSONL atomically."""
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file + rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def connect(path: Path, timeout: float = 120) -> sqlite3.Connection:
    """A WAL-mode sqlite connection that tolerates concurrent writers (other processes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=timeout, isolation_level=None, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    return db


def squash(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def vendored(*parts: str) -> Path:
    """Put a vendored (or fetched) source tree on sys.path once and return it.

    `vendored("deepscholar")` -> third_party/deepscholar; absolute paths are used as given."""
    path = Path(*parts) if Path(parts[0]).is_absolute() else VENDOR.joinpath(*parts)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
