"""Layer 4 — Runtime Mitigation (four-track hybrid dispatcher).

Pipeline::

    token_ids ──► Bloom gate ──► Registry lookup ──► Severity router ──► Track {1..4}
                    │                 │                                  ├─ 1: substitution
                    │                 │                                  ├─ 2: activation repair
                    │                 └──► metadata/fallback_rules       ├─ 3: embedding injection
                    └──► skip (99% of traffic)                           └─ 4: hard block
"""
from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)
from .pipeline import MitigationPipeline
from .severity_router import RouterConfig, SeverityRouter
from .track1_substitution import Track1Substitution
from .track2_activation_repair import Track2ActivationRepair
from .track3_embedding_injection import Track3EmbeddingInjection
from .track4_hard_block import Track4HardBlock

__all__ = [
    "MitigationAction",
    "MitigationContext",
    "MitigationPipeline",
    "MitigationResult",
    "MitigationTrack",
    "RouterConfig",
    "SeverityRouter",
    "Track1Substitution",
    "Track2ActivationRepair",
    "Track3EmbeddingInjection",
    "Track4HardBlock",
]
