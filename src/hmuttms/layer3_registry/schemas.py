"""Typed schemas for the Layer-3 registry entries.

All schemas use plain dataclasses with ``to_dict`` / ``from_dict`` helpers
so they can round-trip cleanly through the KV backend (which stores
JSON).

Kept deliberately close to the exact JSON shapes shown in the HMUTTMS
flowchart PDF (Layer 3) so downstream code can copy-paste from the doc
without surprises.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ==================================================================== helpers
def _shared_base(data: dict) -> dict:
    """Pick common identity / score / timestamp fields from a record."""
    return {
        "token_id": int(data["token_id"]),
        "token_string": data.get("token_string") or data.get("token_str"),
        "model_a_score": float(data["model_a_score"]),
        "model_b_score": (
            None if data.get("model_b_score") is None else float(data["model_b_score"])
        ),
        "detection_date": data.get("detection_date") or _utc_today(),
    }


# ================================================================== blacklist
@dataclass
class BlacklistEntry:
    """Confirmed under-trained — both Model A and Model B fail.

    Matches the PDF example::

        {
          "token_id": 12345,
          "token_string": "solidgoldmagikarp",
          "status": "BLOCKED",
          "model_a_score": 0.12,
          "model_b_score": 0.38,
          "detection_date": "2025-12-27",
          "action": "reject_and_alert"
        }
    """

    token_id: int
    token_string: Optional[str]
    model_a_score: float
    model_b_score: Optional[float]
    detection_date: str = field(default_factory=_utc_today)
    status: str = "BLOCKED"
    action: str = "reject_and_alert"
    severity: str = "high"                              # high | medium | low
    sources: list[str] = field(default_factory=list)    # ["magikarp","glitchminer"]
    backends: list[str] = field(default_factory=list)   # Model-B backends used
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BlacklistEntry":
        base = _shared_base(data)
        return cls(
            **base,
            status=data.get("status", "BLOCKED"),
            action=data.get("action", "reject_and_alert"),
            severity=data.get("severity", "high"),
            sources=list(data.get("sources") or []),
            backends=list(data.get("backends") or []),
            notes=data.get("notes"),
        )


# ============================================================== substitution
@dataclass
class ReplacementStrategy:
    """How to replace a SUBSTITUTABLE token at inference time."""

    method: str = "semantic_decomposition"
    substitute_tokens: list[str] = field(default_factory=list)
    substitute_token_ids: list[int] = field(default_factory=list)
    fallback_embedding: Optional[list[float]] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReplacementStrategy":
        return cls(
            method=data.get("method", "semantic_decomposition"),
            substitute_tokens=list(data.get("substitute_tokens") or []),
            substitute_token_ids=list(data.get("substitute_token_ids") or []),
            fallback_embedding=(
                list(data["fallback_embedding"])
                if data.get("fallback_embedding") is not None
                else None
            ),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
        )


@dataclass
class SubstitutionEntry:
    """Model-A-specific weakness — Model B handles it, so swap it in.

    Matches the PDF example::

        {
          "token_id": 67890,
          "token_string": " thethe",
          "status": "SUBSTITUTABLE",
          "model_a_score": 0.25,
          "model_b_score": 0.85,
          "replacement_strategy": {
            "method": "semantic_decomposition",
            "substitute_tokens": [" the", " the"],
            "fallback_embedding": [0.234, -0.456, ...]
          }
        }
    """

    token_id: int
    token_string: Optional[str]
    model_a_score: float
    model_b_score: Optional[float]
    detection_date: str = field(default_factory=_utc_today)
    status: str = "SUBSTITUTABLE"
    replacement_strategy: ReplacementStrategy = field(default_factory=ReplacementStrategy)
    sources: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["replacement_strategy"] = self.replacement_strategy.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "SubstitutionEntry":
        base = _shared_base(data)
        return cls(
            **base,
            status=data.get("status", "SUBSTITUTABLE"),
            replacement_strategy=ReplacementStrategy.from_dict(
                data.get("replacement_strategy") or {}
            ),
            sources=list(data.get("sources") or []),
            backends=list(data.get("backends") or []),
        )


# ==================================================================== metadata
@dataclass
class MetadataEntry:
    """Severity, frequency, embedding — used by Layer-4 severity router.

    Keeps counters that Layer 5's feedback loop updates over time.
    """

    token_id: int
    token_string: Optional[str] = None
    severity: str = "medium"
    frequency: int = 0                                  # times seen across models
    observed_in_models: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    mitigation_attempts: int = 0
    mitigation_successes: int = 0
    last_strategy: Optional[str] = None                 # "substitution"|"activation"|...
    last_updated: str = field(default_factory=_utc_today)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MetadataEntry":
        return cls(
            token_id=int(data["token_id"]),
            token_string=data.get("token_string"),
            severity=data.get("severity", "medium"),
            frequency=int(data.get("frequency", 0)),
            observed_in_models=list(data.get("observed_in_models") or []),
            embedding=(
                list(data["embedding"]) if data.get("embedding") is not None else None
            ),
            mitigation_attempts=int(data.get("mitigation_attempts", 0)),
            mitigation_successes=int(data.get("mitigation_successes", 0)),
            last_strategy=data.get("last_strategy"),
            last_updated=data.get("last_updated", _utc_today()),
        )

    # --------------------------------------------------- convenience mutators
    def record_attempt(self, strategy: str, success: bool) -> None:
        self.mitigation_attempts += 1
        if success:
            self.mitigation_successes += 1
        self.last_strategy = strategy
        self.last_updated = _utc_today()


# ============================================================= fallback rules
@dataclass
class FallbackRule:
    """Context-aware handling strategy for a token.

    Consumed by Layer-4's four-track dispatcher. ``context_pattern`` is an
    optional regex/substring hint that, when matched against the
    surrounding input, selects this rule over the default.
    """

    token_id: int
    priority: int = 0                                   # higher = checked first
    track: str = "substitution"                         # substitution|activation|embedding|block
    context_pattern: Optional[str] = None
    description: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=_utc_today)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FallbackRule":
        return cls(
            token_id=int(data["token_id"]),
            priority=int(data.get("priority", 0)),
            track=data.get("track", "substitution"),
            context_pattern=data.get("context_pattern"),
            description=data.get("description"),
            params=dict(data.get("params") or {}),
            last_updated=data.get("last_updated", _utc_today()),
        )
