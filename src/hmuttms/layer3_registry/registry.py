"""``TokenRegistry`` — unified accessor over all Layer-3 sub-DBs.

Wraps the four RocksDB-backed stores described in the HMUTTMS flowchart:

* ``blacklist_tokens``   — :class:`BlacklistEntry`
* ``substitution_map``   — :class:`SubstitutionEntry`
* ``metadata``           — :class:`MetadataEntry`
* ``fallback_rules``     — :class:`FallbackRule`

Keys are always the **string form of ``token_id``** so the same registry
can be used from Python, shell inspection, and downstream C++ runtime
code without key-encoding drift.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterator, Optional

from .kv_store import BaseKVBackend, open_backend
from .schemas import (
    BlacklistEntry,
    FallbackRule,
    MetadataEntry,
    SubstitutionEntry,
)

DEFAULT_ROOT = Path("data/mitigation_system/rocksdb_storage")
SUBDIRS = ("blacklist_tokens", "substitution_map", "metadata", "fallback_rules")


class TokenRegistry(AbstractContextManager):
    """Typed facade over the four Layer-3 KV stores."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        backend_kind: str = "auto",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_kind = backend_kind
        self._open: dict[str, BaseKVBackend] = {
            name: open_backend(self.root / name, kind=backend_kind)
            for name in SUBDIRS
        }

    # ---------------------------------------------------------------- dunder
    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    def close(self) -> None:
        for b in self._open.values():
            b.close()

    # ------------------------------------------------------ key/value helper
    @staticmethod
    def _k(token_id: int | str) -> str:
        return str(int(token_id))

    # ============================================================ blacklist
    def put_blacklist(self, entry: BlacklistEntry) -> None:
        self._open["blacklist_tokens"].put(self._k(entry.token_id), entry.to_dict())

    def get_blacklist(self, token_id: int) -> Optional[BlacklistEntry]:
        raw = self._open["blacklist_tokens"].get(self._k(token_id))
        return BlacklistEntry.from_dict(raw) if raw else None

    def iter_blacklist(self) -> Iterator[BlacklistEntry]:
        for _, raw in self._open["blacklist_tokens"].iter_items():
            yield BlacklistEntry.from_dict(raw)

    def delete_blacklist(self, token_id: int) -> None:
        self._open["blacklist_tokens"].delete(self._k(token_id))

    # ========================================================= substitution
    def put_substitution(self, entry: SubstitutionEntry) -> None:
        self._open["substitution_map"].put(self._k(entry.token_id), entry.to_dict())

    def get_substitution(self, token_id: int) -> Optional[SubstitutionEntry]:
        raw = self._open["substitution_map"].get(self._k(token_id))
        return SubstitutionEntry.from_dict(raw) if raw else None

    def iter_substitution(self) -> Iterator[SubstitutionEntry]:
        for _, raw in self._open["substitution_map"].iter_items():
            yield SubstitutionEntry.from_dict(raw)

    def delete_substitution(self, token_id: int) -> None:
        self._open["substitution_map"].delete(self._k(token_id))

    # ============================================================= metadata
    def put_metadata(self, entry: MetadataEntry) -> None:
        self._open["metadata"].put(self._k(entry.token_id), entry.to_dict())

    def get_metadata(self, token_id: int) -> Optional[MetadataEntry]:
        raw = self._open["metadata"].get(self._k(token_id))
        return MetadataEntry.from_dict(raw) if raw else None

    def upsert_metadata(self, entry: MetadataEntry) -> MetadataEntry:
        """Insert if missing; merge counters if already present."""
        existing = self.get_metadata(entry.token_id)
        if existing is None:
            self.put_metadata(entry)
            return entry
        merged = MetadataEntry(
            token_id=entry.token_id,
            token_string=entry.token_string or existing.token_string,
            severity=entry.severity or existing.severity,
            frequency=max(existing.frequency, entry.frequency),
            observed_in_models=sorted(
                set(existing.observed_in_models) | set(entry.observed_in_models)
            ),
            embedding=entry.embedding or existing.embedding,
            mitigation_attempts=existing.mitigation_attempts + entry.mitigation_attempts,
            mitigation_successes=existing.mitigation_successes + entry.mitigation_successes,
            last_strategy=entry.last_strategy or existing.last_strategy,
            last_updated=entry.last_updated,
        )
        self.put_metadata(merged)
        return merged

    def iter_metadata(self) -> Iterator[MetadataEntry]:
        for _, raw in self._open["metadata"].iter_items():
            yield MetadataEntry.from_dict(raw)

    # ========================================================== fallback_rules
    def put_fallback(self, rule: FallbackRule) -> None:
        self._open["fallback_rules"].put(self._k(rule.token_id), rule.to_dict())

    def get_fallback(self, token_id: int) -> Optional[FallbackRule]:
        raw = self._open["fallback_rules"].get(self._k(token_id))
        return FallbackRule.from_dict(raw) if raw else None

    def iter_fallback(self) -> Iterator[FallbackRule]:
        for _, raw in self._open["fallback_rules"].iter_items():
            yield FallbackRule.from_dict(raw)

    # =============================================================== summary
    def counts(self) -> dict[str, int]:
        return {name: len(db) for name, db in self._open.items()}

    def lookup(self, token_id: int) -> dict[str, Optional[dict]]:
        """Return the token's rows across every sub-db (for debugging)."""
        k = self._k(token_id)
        return {name: db.get(k) for name, db in self._open.items()}
