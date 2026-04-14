"""Pluggable key-value backend for the Layer-3 token registry.

Two implementations are provided:

* :class:`RocksDictBackend` — thin wrapper around ``rocksdict`` (the Python
  binding for Facebook's RocksDB). Chosen per the HMUTTMS PDF for
  ~8 µs/lookup and billions-of-keys scalability. Install with
  ``pip install rocksdict`` on the research PC.

* :class:`JsonDirBackend` — directory-of-JSON-files fallback. Zero native
  dependencies; lets the rest of the pipeline be developed / unit-tested
  on a laptop that can't build RocksDB. Same semantics, much slower.

Both backends store UTF-8 JSON payloads keyed by a **string** key
(``str(token_id)``). Binary serialization is intentionally avoided so the
same on-disk file can be inspected with ``cat`` during development.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional


# ========================================================================== base
class BaseKVBackend(ABC):
    """Minimal KV interface used by :class:`TokenRegistry`."""

    @abstractmethod
    def get(self, key: str) -> Optional[dict]: ...

    @abstractmethod
    def put(self, key: str, value: dict) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def iter_items(self) -> Iterator[tuple[str, dict]]: ...

    @abstractmethod
    def close(self) -> None: ...

    # ------------------------------------------------------------------- misc
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_items())

    def put_many(self, items: Any) -> int:
        n = 0
        for k, v in items:
            self.put(str(k), v)
            n += 1
        return n


# ==================================================================== rocksdict
class RocksDictBackend(BaseKVBackend):
    """RocksDB-backed store via ``rocksdict``.

    Each sub-db (blacklist / substitution / metadata / fallback) is a
    separate RocksDB directory. This avoids column-family setup and makes
    it trivial to back up individual slices.
    """

    def __init__(self, path: str | Path):
        try:
            import rocksdict  # type: ignore
        except ImportError as e:  # pragma: no cover - import guarded at runtime
            raise ImportError(
                "rocksdict is required for RocksDictBackend. "
                "Install with: pip install rocksdict"
            ) from e

        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        opts = rocksdict.Options()
        opts.create_if_missing(True)
        self._db = rocksdict.Rdict(str(self._path), options=opts)

    # ------------------------------------------------------------------ CRUD
    def get(self, key: str) -> Optional[dict]:
        raw = self._db.get(key)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw) if isinstance(raw, str) else raw

    def put(self, key: str, value: dict) -> None:
        self._db[key] = json.dumps(value, ensure_ascii=False)

    def delete(self, key: str) -> None:
        try:
            del self._db[key]
        except KeyError:
            pass

    def iter_items(self) -> Iterator[tuple[str, dict]]:
        for k, v in self._db.items():
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            yield (str(k), json.loads(v) if isinstance(v, str) else v)

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


# ===================================================================== json-dir
class JsonDirBackend(BaseKVBackend):
    """Dependency-free fallback: one JSON file per key under a directory.

    Keys are sanitised (``/`` → ``_``) before being used as filenames so
    arbitrary token strings / ids are safe.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ path helper
    def _file(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self._path / f"{safe}.json"

    # ------------------------------------------------------------------ CRUD
    def get(self, key: str) -> Optional[dict]:
        f = self._file(key)
        if not f.exists():
            return None
        return json.loads(f.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self._file(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )

    def delete(self, key: str) -> None:
        f = self._file(key)
        if f.exists():
            f.unlink()

    def iter_items(self) -> Iterator[tuple[str, dict]]:
        for f in sorted(self._path.glob("*.json")):
            try:
                yield f.stem, json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

    def close(self) -> None:
        return None


# ==================================================================== factory
def open_backend(path: str | Path, kind: str = "auto") -> BaseKVBackend:
    """Open a KV backend.

    ``kind`` can be ``"rocksdict"``, ``"jsondir"``, or ``"auto"``. ``auto``
    tries RocksDict first and silently falls back to ``JsonDirBackend`` if
    the native library is not installed.
    """
    kind = kind.lower()
    if kind == "rocksdict":
        return RocksDictBackend(path)
    if kind == "jsondir":
        return JsonDirBackend(path)
    if kind == "auto":
        try:
            return RocksDictBackend(path)
        except ImportError:
            return JsonDirBackend(path)
    raise ValueError(f"Unknown backend kind: {kind!r}")
