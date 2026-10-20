"""Local arXiv backend: BM25 over a pinned metadata snapshot, same contract as `arxiv`.

The default backend of runs (experiment `search.arxiv: local`, passed to workers as
JEV_ARXIV_BACKEND): the live API rate-limits this host after ~10 requests, whereas this is one
frozen corpus at zero latency, identical for every system and exactly reproducible. The
snapshot is HF librarian-bots/arxiv-metadata-snapshot at a pinned revision, a weekly mirror of the
Kaggle arXiv metadata (11 parquet shards, ~2.7 GB). `build_index()` downloads the shards into
ROOT/data/raw/arxiv_snapshot/ (sha256-checked against the revision's LFS ids) and writes
ROOT/data/derived/arxiv_index/: a bm25s index over title + abstract (`bm25/`), the v1 submission
times aligned with index rows (`published.npy`, the date mask) and `meta.sqlite` (row and
versionless id -> paper dict in the API backend's shape). Ids carry the latest version at snapshot
time, just as the API returns the latest version at query time.
"""

from __future__ import annotations

import functools
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from jevscholar.paths import DATA_DERIVED, DATA_RAW
from jevscholar.search import FatalToolError
from jevscholar.search.policy import arxiv_id
from jevscholar.util import download, squash

if TYPE_CHECKING:
    import numpy as np

HF = "https://huggingface.co"
REPO = "librarian-bots/arxiv-metadata-snapshot"
REVISION = "5733e66fde7e840bf1dcb2d9cffa0c64f4c5654d"
RAW = DATA_RAW / "arxiv_snapshot"
INDEX = DATA_DERIVED / "arxiv_index"
COLUMNS = ["id", "title", "abstract", "authors_parsed", "categories", "versions"]


def search(query: str, k: int, before: datetime | None = None) -> tuple[list[dict], bool]:
    """Top-k papers by BM25 over title + abstract, published (v1) strictly before `before`."""
    import bm25s
    import numpy as np

    bm25, published = _load(INDEX)
    tokens = bm25s.tokenize(query, stopwords="en", return_ids=False, show_progress=False)[0]
    tokens = [t for t in tokens if t in bm25.vocab_dict]
    if not tokens:
        return [], False
    mask = None if before is None else (published < before.timestamp()).astype(np.float32)
    docs, scores = bm25.retrieve(
        [tokens], k=min(k, len(published)), weight_mask=mask, show_progress=False
    )
    rows = [int(d) for d, s in zip(docs[0], scores[0], strict=True) if s > 0]
    found = _papers(INDEX, "row", rows)
    return [found[r] for r in rows], False


def lookup(ids: list[str]) -> tuple[dict[str, dict], bool]:
    """Snapshot metadata keyed by the requested ids (versions are ignored)."""
    _load(INDEX)  # fails loudly if there is no index
    wanted = {i: base for i in ids if (base := arxiv_id(i))}
    found = _papers(INDEX, "id", sorted(set(wanted.values())))
    return {i: found[b] for i, b in wanted.items() if b in found}, False


# --------------------------------------------------------------------------- building
def build_index() -> Path:
    """Download the pinned snapshot (reusing verified shards) and rebuild the index."""
    import pyarrow.parquet as pq

    def records() -> Iterator[dict[str, Any]]:
        for shard in _download():
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=65536, columns=COLUMNS):
                yield from batch.to_pylist()

    write_index(records(), INDEX)
    return INDEX


def write_index(records: Iterable[dict[str, Any]], out: Path) -> None:
    """Index Kaggle-format records (id, title, abstract, authors_parsed, categories, versions)."""
    import bm25s
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.sqlite").unlink(missing_ok=True)
    texts, published = [], []
    with closing(sqlite3.connect(out / "meta.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE papers (row INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,"
            " paper TEXT NOT NULL)"
        )
        seen: set[str] = set()
        for rec in records:
            if (key := arxiv_id(rec["id"]) or rec["id"]) in seen:
                continue  # the Kaggle dump repeats ~80 cross-listed old-style ids; keep the first
            seen.add(key)
            paper = _paper(rec)
            db.execute("INSERT INTO papers VALUES (?, ?, ?)", (len(texts), key, json.dumps(paper)))
            texts.append(f"{paper['title']}\n{paper['abstract']}")
            published.append(datetime.fromisoformat(paper["published"]).timestamp())
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en"))
    bm25.save(str(out / "bm25"))
    np.save(out / "published.npy", np.asarray(published))  # written last: marks a complete build


def _paper(rec: dict[str, Any]) -> dict[str, Any]:
    versions = sorted(rec["versions"], key=lambda v: int(v["version"].lstrip("v")))
    vid = rec["id"] + versions[-1]["version"]
    return {
        "id": vid,
        "title": squash(rec["title"]),
        "abstract": squash(rec["abstract"]),
        "published": _rfc2822(versions[0]["created"]),
        "updated": _rfc2822(versions[-1]["created"]),
        "authors": [
            " ".join(filter(None, (first, last, *rest)))
            for last, first, *rest in rec["authors_parsed"]
        ],
        "categories": rec["categories"].split(),
        "url": f"http://arxiv.org/abs/{vid}",
    }


def _rfc2822(s: str) -> str:
    return parsedate_to_datetime(s).astimezone(UTC).isoformat()


def _download() -> list[Path]:
    """The revision's parquet shards, verified against their LFS sha256 (good copies are kept)."""
    r = requests.get(f"{HF}/api/datasets/{REPO}/tree/{REVISION}/data", timeout=60)
    r.raise_for_status()
    shards = []
    for f in sorted(r.json(), key=lambda f: f["path"]):
        if f["path"].endswith(".parquet"):
            dest = RAW / Path(f["path"]).name
            download(f"{HF}/datasets/{REPO}/resolve/{REVISION}/{f['path']}", dest, f["lfs"]["oid"])
            shards.append(dest)
    return shards


# --------------------------------------------------------------------------- reading
@functools.cache
def _load(index: Path) -> tuple[Any, np.ndarray]:
    import bm25s
    import numpy as np

    if not (index / "published.npy").exists():  # without it every search would fail silently
        raise FatalToolError(f"no local arXiv index at {index}; run build_index() first")
    bm25 = bm25s.BM25.load(str(index / "bm25"), mmap=True, show_progress=False)
    return bm25, np.load(index / "published.npy", mmap_mode="r")


def _papers(index: Path, column: str, keys: list[Any]) -> dict[Any, dict]:
    marks = ",".join("?" * len(keys))
    with closing(sqlite3.connect(f"file:{index / 'meta.sqlite'}?mode=ro", uri=True)) as db:
        rows = db.execute(f"SELECT {column}, paper FROM papers WHERE {column} IN ({marks})", keys)
        return {key: json.loads(paper) for key, paper in rows}
