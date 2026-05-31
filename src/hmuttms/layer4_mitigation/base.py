"""Base types for the Layer-4 four-track mitigation system.

A *track* transforms a flagged token (or its activations/embeddings) into
something the downstream model can safely consume. Every track returns a
:class:`MitigationResult` so the pipeline can audit success, update
Layer-5 feedback counters, and escalate to the next track on failure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


# ============================================================== action enums
class MitigationAction(str, Enum):
    """What the pipeline did with a flagged token id."""

    PASS_THROUGH = "pass_through"            # never flagged
    SUBSTITUTE = "substitute"                # Track 1
    REPAIR_ACTIVATION = "repair_activation"  # Track 2
    INJECT_EMBEDDING = "inject_embedding"    # Track 3
    BLOCK = "block"                          # Track 4
    ESCALATED = "escalated"                  # track failed, next one tried


# =================================================================== contexts
@dataclass
class MitigationContext:
    """Per-call state carried through the pipeline.

    ``token_ids`` is the full prompt (mutable); ``position`` marks the
    flagged token currently being processed. ``neighbor_window`` tells
    Track 3 how many tokens on each side to use for the contextual
    average.
    """

    token_ids: list[int]
    position: int
    neighbor_window: int = 3
    tokenizer: Any = None             # HF PreTrainedTokenizer (optional)
    model: Any = None                 # HF PreTrainedModel (optional)
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ utils
    @property
    def current_id(self) -> int:
        return int(self.token_ids[self.position])

    def neighbors(self) -> list[int]:
        lo = max(self.position - self.neighbor_window, 0)
        hi = min(self.position + self.neighbor_window + 1, len(self.token_ids))
        return [
            int(t)
            for i, t in enumerate(self.token_ids[lo:hi], start=lo)
            if i != self.position
        ]


# ==================================================================== result
@dataclass
class MitigationResult:
    """Outcome of applying a track (or the whole pipeline) to one token."""

    original_token_id: int
    action: MitigationAction
    track: Optional[str] = None
    replacement_embedding: Optional[Sequence[float]] = None
    success: bool = True
    reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_token_id": int(self.original_token_id),
            "action": self.action.value,
            "track": self.track,
            "replacement_embedding": (
                list(self.replacement_embedding)
                if self.replacement_embedding is not None
                else None
            ),
            "success": bool(self.success),
            "reason": self.reason,
            "details": self.details,
        }


# ===================================================================== track
class MitigationTrack(ABC):
    """Base class for the four mitigation tracks."""

    name: str = "base"

    @abstractmethod
    def applies(self, context: MitigationContext, registry_view: dict) -> bool:
        """Return True if this track has something to try for the token."""

    @abstractmethod
    def apply(self, context: MitigationContext, registry_view: dict) -> MitigationResult:
        """Attempt to mitigate the flagged token at ``context.position``."""
