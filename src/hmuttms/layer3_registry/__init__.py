"""Layer 3 — Simple JSON-dictionary token registry.

On-disk layout::

    data/mitigation_system/
      ├── bloom_filter.bin           # built by Layer 1 (optional fast gate)
      └── registry/
          └── registry.json          # the whole registry, four sections:
                                      #   blacklist_tokens, substitution_map,
                                      #   metadata, fallback_rules

The original HMUTTMS flowchart specified RocksDB for ~8 µs/lookup at
production scale. For a thesis verification that processes a fixed token
set once, a plain Python dict persisted to a single human-readable JSON
file is equivalent and far easier to inspect/defend — so that is what
:class:`TokenRegistry` now uses.
"""
from .registry import DEFAULT_ROOT, SECTIONS, TokenRegistry
from .schemas import (
    BlacklistEntry,
    FallbackRule,
    MetadataEntry,
    SubstitutionEntry,
)

__all__ = [
    "BlacklistEntry",
    "DEFAULT_ROOT",
    "FallbackRule",
    "MetadataEntry",
    "SECTIONS",
    "SubstitutionEntry",
    "TokenRegistry",
]
