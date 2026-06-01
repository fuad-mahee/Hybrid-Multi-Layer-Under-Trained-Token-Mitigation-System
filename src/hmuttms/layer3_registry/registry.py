"""``TokenRegistry`` — simple JSON-dictionary token registry.

Replaces the RocksDB backend with a single, human-readable JSON file. For
a thesis verification that processes a fixed token set once, a plain
in-memory dict persisted to JSON does everything RocksDB did — and the
committee can open the file and read it directly.

On-disk layout::

    <registry_root>/registry.json
    {
      "blacklist_tokens": { "12345": {...BlacklistEntry...}, ... },
      "substitution_map": { "67890": {...SubstitutionEntry...}, ... },
      "metadata":         { "12345": {...MetadataEntry...}, ... },
      "fallback_rules":   { "12345": {...FallbackRule...}, ... }
    }

Keys are always the **string form of ``token_id``**. The whole file is
loaded into memory on open and written back on ``save()`` / context exit.

The public method names match the previous RocksDB-backed registry, so
the rest of the pipeline (populate, Track 3, session, feedback) is
unchanged.
"""
from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterator, Optional

from .schemas import (
    BlacklistEntry,
    FallbackRule,
    MetadataEntry,
    SubstitutionEntry,
)

DEFAULT_ROOT = Path("data/mitigation_system/registry")
SECTIONS = ("blacklist_tokens", "substitution_map", "metadata", "fallback_rules")


class TokenRegistry(AbstractContextManager):
    """Typed facade over a single JSON dictionary file."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        backend_kind: str | None = None,  # accepted & ignored (CLI compat)
    ):
        root = Path(root)
        # Accept either a directory (→ registry.json inside) or a .json path.
        if root.suffix == ".json":
            self.path = root
        else:
            self.path = root / "registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.path.parent

        self._data: dict[str, dict[str, dict]] = {s: {} for s in SECTIONS}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            for s in SECTIONS:
                self._data[s] = {
                    str(k): v for k, v in (loaded.get(s) or {}).items()
                }
        self._dirty = False

    # ---------------------------------------------------------------- dunder
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._dirty:
            self.save()

    def save(self) -> Path:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False
        return self.path

    # ------------------------------------------------------ key/value helper
    @staticmethod
    def _k(token_id: int | str) -> str:
        return str(int(token_id))

    # ============================================================ blacklist
    def put_blacklist(self, entry: BlacklistEntry) -> None:
        self._data["blacklist_tokens"][self._k(entry.token_id)] = entry.to_dict()
        self._dirty = True

    def get_blacklist(self, token_id: int) -> Optional[BlacklistEntry]:
        raw = self._data["blacklist_tokens"].get(self._k(token_id))
        return BlacklistEntry.from_dict(raw) if raw else None

    def iter_blacklist(self) -> Iterator[BlacklistEntry]:
        for raw in self._data["blacklist_tokens"].values():
            yield BlacklistEntry.from_dict(raw)

    def delete_blacklist(self, token_id: int) -> None:
        if self._data["blacklist_tokens"].pop(self._k(token_id), None) is not None:
            self._dirty = True

    # ========================================================= substitution
    def put_substitution(self, entry: SubstitutionEntry) -> None:
        self._data["substitution_map"][self._k(entry.token_id)] = entry.to_dict()
        self._dirty = True

    def get_substitution(self, token_id: int) -> Optional[SubstitutionEntry]:
        raw = self._data["substitution_map"].get(self._k(token_id))
        return SubstitutionEntry.from_dict(raw) if raw else None

    def iter_substitution(self) -> Iterator[SubstitutionEntry]:
        for raw in self._data["substitution_map"].values():
            yield SubstitutionEntry.from_dict(raw)

    def delete_substitution(self, token_id: int) -> None:
        if self._data["substitution_map"].pop(self._k(token_id), None) is not None:
            self._dirty = True

    # ============================================================= metadata
    def put_metadata(self, entry: MetadataEntry) -> None:
        self._data["metadata"][self._k(entry.token_id)] = entry.to_dict()
        self._dirty = True

    def get_metadata(self, token_id: int) -> Optional[MetadataEntry]:
        raw = self._data["metadata"].get(self._k(token_id))
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
        for raw in self._data["metadata"].values():
            yield MetadataEntry.from_dict(raw)

    # ========================================================== fallback_rules
    def put_fallback(self, rule: FallbackRule) -> None:
        self._data["fallback_rules"][self._k(rule.token_id)] = rule.to_dict()
        self._dirty = True

    def get_fallback(self, token_id: int) -> Optional[FallbackRule]:
        raw = self._data["fallback_rules"].get(self._k(token_id))
        return FallbackRule.from_dict(raw) if raw else None

    def iter_fallback(self) -> Iterator[FallbackRule]:
        for raw in self._data["fallback_rules"].values():
            yield FallbackRule.from_dict(raw)

    def delete_fallback(self, token_id: int) -> None:
        if self._data["fallback_rules"].pop(self._k(token_id), None) is not None:
            self._dirty = True

    # =============================================================== summary
    def counts(self) -> dict[str, int]:
        return {s: len(self._data[s]) for s in SECTIONS}

    def lookup(self, token_id: int) -> dict[str, Optional[dict]]:
        """Return the token's rows across every section (for runtime + debug)."""
        k = self._k(token_id)
        return {s: self._data[s].get(k) for s in SECTIONS}
