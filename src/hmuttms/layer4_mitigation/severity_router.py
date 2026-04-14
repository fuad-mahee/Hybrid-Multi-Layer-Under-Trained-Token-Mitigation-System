"""Severity router for the Layer-4 pipeline.

Given a registry view for a flagged token, produce an **ordered list** of
track names to try. The pipeline walks the list and applies each track
until one returns ``success=True`` (or the list is exhausted).

Rules (default; override via :class:`RouterConfig`):

* If a matching :class:`FallbackRule` exists for this token id, use its
  ``track`` first (highest-priority rule wins).
* Otherwise, derive from classification + severity:
    - ``SUBSTITUTABLE``             -> ``[1, 3, 4]``
    - ``BLACKLIST`` + severity high -> ``[3, 4]``         (embed inject -> block)
    - ``BLACKLIST`` + severity med  -> ``[2, 3, 4]``      (activation -> embed -> block)
    - ``BLACKLIST`` + severity low  -> ``[1, 2, 3, 4]``
* A track the registry has no data for (e.g. no substitution map entry)
  is still listed; it will cheaply ``return success=False`` via its
  ``applies()`` check and fall through.

Feedback-driven overrides: if ``metadata.mitigation_attempts`` > 0 and
``success_rate < escalate_below``, the router demotes the last-used
track (``metadata.last_strategy``) by appending it to the end so the
pipeline prefers a different track next time. This is the runtime half
of Layer 5's feedback loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TRACK_ORDER = {
    "track1_substitution": 1,
    "track2_activation_repair": 2,
    "track3_embedding_injection": 3,
    "track4_hard_block": 4,
}
NAME_BY_NUMBER = {v: k for k, v in TRACK_ORDER.items()}


@dataclass
class RouterConfig:
    escalate_below: float = 0.4         # success-rate floor before demoting a track
    min_attempts_to_escalate: int = 5
    # Policies per (classification, severity); values are track numbers.
    policies: dict[tuple[str, str], list[int]] = None

    def __post_init__(self):
        if self.policies is None:
            self.policies = {
                ("SUBSTITUTABLE", "high"):   [1, 3, 4],
                ("SUBSTITUTABLE", "medium"): [1, 3, 4],
                ("SUBSTITUTABLE", "low"):    [1, 3, 4],
                ("BLACKLIST", "high"):       [3, 4],
                ("BLACKLIST", "medium"):     [2, 3, 4],
                ("BLACKLIST", "low"):        [1, 2, 3, 4],
            }


class SeverityRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.cfg = config or RouterConfig()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _classify(registry_view: dict) -> str:
        if registry_view.get("blacklist_tokens"):
            return "BLACKLIST"
        if registry_view.get("substitution_map"):
            return "SUBSTITUTABLE"
        return "REVIEW"

    @staticmethod
    def _severity(registry_view: dict) -> str:
        meta = registry_view.get("metadata") or {}
        return (meta.get("severity") or "medium").lower()

    def _success_rate(self, registry_view: dict) -> Optional[float]:
        meta = registry_view.get("metadata") or {}
        attempts = int(meta.get("mitigation_attempts", 0))
        if attempts < self.cfg.min_attempts_to_escalate:
            return None
        successes = int(meta.get("mitigation_successes", 0))
        return successes / max(attempts, 1)

    # --------------------------------------------------------------- choose
    def order(self, registry_view: dict) -> list[str]:
        """Return an ordered list of track *names* to try."""
        # 1) Explicit fallback rule wins.
        rule = registry_view.get("fallback_rules") or {}
        if rule.get("track") in TRACK_ORDER:
            return self._expand_with_backstop([rule["track"]])

        # 2) Policy from classification + severity.
        cls = self._classify(registry_view)
        sev = self._severity(registry_view)
        numbers = self.cfg.policies.get((cls, sev)) or [1, 3, 4]
        names = [NAME_BY_NUMBER[n] for n in numbers if n in NAME_BY_NUMBER]

        # 3) Feedback-driven demotion of a chronically-failing last strategy.
        rate = self._success_rate(registry_view)
        meta = registry_view.get("metadata") or {}
        last = meta.get("last_strategy")
        if (
            rate is not None
            and rate < self.cfg.escalate_below
            and last in names
            and len(names) > 1
        ):
            names = [n for n in names if n != last] + [last]

        return self._expand_with_backstop(names)

    # ------------------------------------------------------ backstop helper
    @staticmethod
    def _expand_with_backstop(names: list[str]) -> list[str]:
        if "track4_hard_block" not in names:
            names = list(names) + ["track4_hard_block"]
        return names
