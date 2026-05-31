"""Layer 4 — Runtime Mitigation (embedding injection).

Pipeline::

    token_ids ──► Bloom gate ──► Registry lookup ──► Track 3: Embedding Injection
                    │                 │
                    └──► skip (99% of traffic)
"""
from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)
from .pipeline import MitigationPipeline
from .severity_router import RouterConfig, SeverityRouter
from .track3_embedding_injection import Track3EmbeddingInjection

__all__ = [
    "MitigationAction",
    "MitigationContext",
    "MitigationPipeline",
    "MitigationResult",
    "MitigationTrack",
    "RouterConfig",
    "SeverityRouter",
    "Track3EmbeddingInjection",
]
