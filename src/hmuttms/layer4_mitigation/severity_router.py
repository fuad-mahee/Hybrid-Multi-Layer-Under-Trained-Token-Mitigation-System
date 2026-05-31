"""Severity router for the Layer-4 pipeline.

Only Track 3 (Context-Aware Embedding Injection) is active. The router
always returns ``["track3_embedding_injection"]`` regardless of
classification or severity, since this is the only mitigation strategy
that produced results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterConfig:
    pass  # reserved for future configuration


class SeverityRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.cfg = config or RouterConfig()

    def order(self, registry_view: dict) -> list[str]:
        """Always return Track 3 — the only active mitigation track."""
        return ["track3_embedding_injection"]
