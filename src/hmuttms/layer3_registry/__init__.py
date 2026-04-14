"""Layer 3 — Persistent Lookup Table System (RocksDB-based token registry).

On-disk layout (per the HMUTTMS flowchart)::

    data/mitigation_system/
      ├── bloom_filter.bin           # built by Layer 1
      └── rocksdb_storage/
          ├── blacklist_tokens/      # confirmed under-trained (both models fail)
          ├── substitution_map/      # model-specific weaknesses + replacements
          ├── metadata/              # severity, frequency, embeddings, stats
          └── fallback_rules/        # context-aware handling strategies

The storage layer is pluggable (see :mod:`.kv_store`):

* ``RocksDictBackend`` — production backend using ``rocksdict`` (install with
  ``pip install rocksdict`` on the research PC).
* ``JsonDirBackend``   — dependency-free fallback that writes one JSON file
  per key. Slow, but lets you develop and unit-test end-to-end on a laptop
  without building the native RocksDB wheel.
"""
from .kv_store import BaseKVBackend, JsonDirBackend, RocksDictBackend, open_backend
from .registry import TokenRegistry
from .schemas import (
    BlacklistEntry,
    FallbackRule,
    MetadataEntry,
    SubstitutionEntry,
)

__all__ = [
    "BaseKVBackend",
    "BlacklistEntry",
    "FallbackRule",
    "JsonDirBackend",
    "MetadataEntry",
    "RocksDictBackend",
    "SubstitutionEntry",
    "TokenRegistry",
    "open_backend",
]
