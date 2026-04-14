"""Periodic re-test against newer Model-B snapshots.

Over time, superior models change (Kimi K2 -> K3, Claude 3.5 -> 4, …).
The PDF calls for periodic re-testing so the registry reflects the best
current classification.

This scheduler:

1. Loads the union of ``blacklist_tokens`` + ``substitution_map`` ids
   from the Layer-3 registry.
2. Builds synthetic Layer-1 candidate rows (reusing the existing
   ``model_a_score``).
3. Runs each row through a user-supplied :class:`BaseModelBTester`
   (typically a fresh ``CachedFileTester`` or an API-backed subclass).
4. Reclassifies via the Layer-2 decision matrix and records the diff —
   promoting or demoting entries accordingly.

Designed to be called from a cron / scheduler (e.g. once a week).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..layer2_validation.decision_matrix import (
    Classification,
    DecisionConfig,
    classify,
)
from ..layer2_validation.ensemble import ensemble_vote
from ..layer2_validation.model_b_tester import BaseModelBTester
from ..layer3_registry.populate import rebuild_bloom
from ..layer3_registry.registry import TokenRegistry
from ..layer3_registry.schemas import (
    BlacklistEntry,
    ReplacementStrategy,
    SubstitutionEntry,
)


@dataclass
class RetestReport:
    total: int = 0
    reclassified: int = 0
    promoted_to_blacklist: list[int] = field(default_factory=list)
    demoted_to_substitutable: list[int] = field(default_factory=list)
    cleared_false_positives: list[int] = field(default_factory=list)
    unchanged: int = 0
    bloom_refresh: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "reclassified": self.reclassified,
            "unchanged": self.unchanged,
            "promoted_to_blacklist": self.promoted_to_blacklist,
            "demoted_to_substitutable": self.demoted_to_substitutable,
            "cleared_false_positives": self.cleared_false_positives,
            "bloom_refresh": self.bloom_refresh,
        }


# ================================================================ scheduler
class RetestScheduler:
    def __init__(
        self,
        registry: TokenRegistry,
        testers: list[BaseModelBTester],
        decision: Optional[DecisionConfig] = None,
        ensemble_weights: Optional[dict[str, float]] = None,
    ):
        if not testers:
            raise ValueError("RetestScheduler needs at least one Model-B tester")
        self.registry = registry
        self.testers = testers
        self.decision = decision or DecisionConfig()
        self.weights = ensemble_weights or {}

    # ---------------------------------------------------------------- iter
    def _iter_candidates(self):
        for entry in self.registry.iter_blacklist():
            yield ("BLACKLIST", entry)
        for entry in self.registry.iter_substitution():
            yield ("SUBSTITUTABLE", entry)

    # ---------------------------------------------------------------- main
    def run(
        self,
        *,
        refresh_bloom: bool = True,
        bloom_path: str | Path = "data/mitigation_system/bloom_filter.bin",
    ) -> RetestReport:
        report = RetestReport()

        for prev_cls, entry in self._iter_candidates():
            report.total += 1

            probes = [t.probe(entry.token_id, entry.token_string) for t in self.testers]
            ens = ensemble_vote(probes, weights=self.weights)
            new_cls = classify(entry.model_a_score, ens.score, self.decision)

            if new_cls.value == prev_cls:
                report.unchanged += 1
                continue

            # ---- Apply the diff to the registry --------------------
            report.reclassified += 1
            if prev_cls == "SUBSTITUTABLE" and new_cls == Classification.BLACKLIST:
                # Substitutable -> Blacklist (Model B now also fails)
                self.registry.delete_substitution(entry.token_id)
                self.registry.put_blacklist(
                    BlacklistEntry(
                        token_id=entry.token_id,
                        token_string=entry.token_string,
                        model_a_score=entry.model_a_score,
                        model_b_score=ens.score,
                        severity="high",
                        sources=list(getattr(entry, "sources", []) or []),
                        backends=ens.backends,
                        notes="promoted_by_retest",
                    )
                )
                report.promoted_to_blacklist.append(entry.token_id)

            elif prev_cls == "BLACKLIST" and new_cls == Classification.SUBSTITUTABLE:
                # Blacklist -> Substitutable (newer Model B handles it)
                self.registry.delete_blacklist(entry.token_id)
                self.registry.put_substitution(
                    SubstitutionEntry(
                        token_id=entry.token_id,
                        token_string=entry.token_string,
                        model_a_score=entry.model_a_score,
                        model_b_score=ens.score,
                        replacement_strategy=ReplacementStrategy(),
                        sources=list(getattr(entry, "sources", []) or []),
                        backends=ens.backends,
                    )
                )
                report.demoted_to_substitutable.append(entry.token_id)

            elif new_cls == Classification.FALSE_POSITIVE:
                # Drop completely — newer Model B says Model A handles it now too.
                self.registry.delete_blacklist(entry.token_id)
                self.registry.delete_substitution(entry.token_id)
                report.cleared_false_positives.append(entry.token_id)

            # REVIEW: keep current row; the next scheduled run will try again.

        if refresh_bloom:
            report.bloom_refresh = rebuild_bloom(self.registry, bloom_path)

        return report
