"""Frozen cache of raw upstream search responses, shared by all worker processes (sqlite, WAL).

Keys are the exact upstream request, never a task: the AccessPolicy is applied on every read, so
two tasks with different cutoffs never share filtered results. Only successful responses are
stored, and the first one stored for a key wins (INSERT OR IGNORE, then re-read), so concurrent
misses in different processes settle on one value and every later read replays it.
"""

from __future__ import annotations

import functools
import json
import threading
from pathlib import Path
from typing import Any

from jevscholar.paths import CACHE
from jevscholar.util import connect


class Store:
    """One connection, shared by a process's threads under a lock (sqlite3 objects are not)."""

    def __init__(self, path: Path) -> None:
        self.db, self._lock = connect(path), threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: dict[str, Any]) -> Any | None:
        with self._lock:
            row = self.db.execute("SELECT value FROM cache WHERE key = ?", (_key(key),)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: dict[str, Any], value: Any) -> Any:
        """Store `value` unless the key is already set; return the value that is stored."""
        row = (_key(key), json.dumps(value))
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO cache VALUES (?, ?)", row)
        return self.get(key)


def _key(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


@functools.cache
def default_store() -> Store:
    return Store(CACHE / "search.sqlite")
