"""Escalator — update registry counters for Track 3 mitigation results.

Reads an event-log JSONL (written by :class:`RuntimeMonitor`) and:

1. Increments ``metadata.mitigation_attempts`` / ``mitigation_successes``
   for every audit record, tagging ``last_strategy`` with the track used.
2. If Track 3's success rate falls below ``EscalationConfig.promote_below``
   (after ``min_attempts``), logs a warning rule in the fallback_rules DB.
3. Optionally prunes rules once a token's success rate recovers above
   ``restore_above``.

The escalator is idempotent w.r.t. the ``_cursor_path`` file; re-running
over the same log picks up where it left off.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..layer3_registry.registry import TokenRegistry
from ..layer3_registry.schemas import FallbackRule, MetadataEntry


# ================================================================= config
# Only Track 3 is active; no escalation ladder needed.
_ACTIVE_TRACK = "track3_embedding_injection"


@dataclass
class EscalationConfig:
    promote_below: float = 0.40      # success rate under which we promote
    restore_above: float = 0.85      # success rate over which we drop the rule
    min_attempts: int = 5            # ignore tokens we haven't seen enough
    include_escalated_as_failure: bool = True


# ============================================================== entry point
class Escalator:
    def __init__(
        self,
        registry: TokenRegistry,
        config: Optional[EscalationConfig] = None,
        cursor_path: str | Path = "data/mitigation_system/feedback/.cursor",
    ):
        self.registry = registry
        self.cfg = config or EscalationConfig()
        self.cursor_path = Path(cursor_path)
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- cursor
    def _read_cursor(self) -> int:
        if not self.cursor_path.exists():
            return 0
        try:
            return int(self.cursor_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return 0

    def _write_cursor(self, offset: int) -> None:
        self.cursor_path.write_text(str(int(offset)), encoding="utf-8")

    # --------------------------------------------------------- log reader
    def _iter_new_events(self, log_path: Path) -> Iterator[tuple[int, dict]]:
        cursor = self._read_cursor()
        with log_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < cursor:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError:
                    continue

    # ---------------------------------------------------------------- core
    def process_log(self, log_path: str | Path) -> dict:
        """Fold the event log into registry updates; return a summary."""
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(log_path)

        # (token_id, track) -> (attempts, successes)
        deltas: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
        last_strategy: dict[int, str] = {}
        quality_success: dict[int, bool] = {}

        last_offset = self._read_cursor()
        for offset, event in self._iter_new_events(log_path):
            last_offset = offset + 1
            q = event.get("quality") or {}
            # If there was no quality judgement, treat track-level success
            # as the fallback signal.
            call_quality_pass = bool(q.get("success", True))

            for row in event.get("audit") or []:
                tid = int(row["original_token_id"])
                track = row.get("track") or ""
                action = row.get("action")
                if not track:
                    continue
                is_escalated = action == "escalated"
                ok = bool(row.get("success"))

                # Base attempt always counts.
                deltas[(tid, track)][0] += 1
                if ok and call_quality_pass:
                    deltas[(tid, track)][1] += 1
                elif is_escalated and self.cfg.include_escalated_as_failure:
                    # already counted as attempt; no success increment
                    pass

                if not is_escalated:
                    last_strategy[tid] = track
                    quality_success[tid] = call_quality_pass and ok

        # ----- apply deltas to registry.metadata --------------------
        touched_tokens = {tid for (tid, _track) in deltas.keys()}
        for tid in touched_tokens:
            meta = self.registry.get_metadata(tid) or MetadataEntry(token_id=tid)
            for (tok, track), (att, succ) in deltas.items():
                if tok != tid:
                    continue
                meta.mitigation_attempts += att
                meta.mitigation_successes += succ
            if tid in last_strategy:
                meta.last_strategy = last_strategy[tid]
            self.registry.upsert_metadata(meta)

        # ----- decide promotions / rule pruning ---------------------
        promotions: list[dict] = []
        restorations: list[dict] = []
        for tid in touched_tokens:
            meta = self.registry.get_metadata(tid)
            if meta is None or meta.mitigation_attempts < self.cfg.min_attempts:
                continue
            rate = meta.mitigation_successes / max(meta.mitigation_attempts, 1)
            rule = self.registry.get_fallback(tid)

            if rate < self.cfg.promote_below and meta.last_strategy == _ACTIVE_TRACK:
                # Track 3 performing poorly — record a warning rule.
                if rule and rule.track == _ACTIVE_TRACK:
                    continue
                new_rule = FallbackRule(
                    token_id=tid,
                    priority=100,
                    track=_ACTIVE_TRACK,
                    description=(
                        f"track3 low success_rate={rate:.2f} — flagged for review"
                    ),
                    params={"success_rate": rate},
                )
                self.registry.put_fallback(new_rule)
                promotions.append(
                    {
                        "token_id": tid,
                        "track": _ACTIVE_TRACK,
                        "success_rate": rate,
                        "note": "low_success_flagged",
                    }
                )
            elif rate > self.cfg.restore_above and rule is not None:
                # Recovered: drop the override so the router returns to policy.
                self.registry._open["fallback_rules"].delete(str(tid))  # noqa: SLF001
                restorations.append({"token_id": tid, "success_rate": rate})

        self._write_cursor(last_offset)

        return {
            "log": str(log_path),
            "events_processed": last_offset - self._read_cursor() if False else None,
            "new_cursor": last_offset,
            "tokens_updated": len(touched_tokens),
            "promotions": promotions,
            "restorations": restorations,
        }

    # --------------------------------------------- programmatic shortcut
    def apply_deltas(self, records: Iterable[dict]) -> None:
        """Apply a pre-loaded iterable of audit rows (tests / one-offs)."""
        for row in records:
            tid = int(row["original_token_id"])
            track = row.get("track")
            if not track:
                continue
            meta = self.registry.get_metadata(tid) or MetadataEntry(token_id=tid)
            meta.mitigation_attempts += 1
            if row.get("success"):
                meta.mitigation_successes += 1
            meta.last_strategy = track
            self.registry.upsert_metadata(meta)
