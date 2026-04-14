"""Mitigation evaluation — per-track success rates + quality scoring.

Two evaluation modes:

``from_event_log``
    Replays the RuntimeMonitor event log written during real inference.
    Rich: includes quality scores, generation latency, and track-level
    breakdown per token.

``from_audit_list``
    Accepts any list of :class:`~hmuttms.layer4_mitigation.base.MitigationResult`
    dicts (e.g. produced by running the pipeline in a test loop). Lighter
    weight; no quality scoring.

Both return a :class:`MitigationEvalResult` whose ``to_dict()`` is consumed
by the report module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from .metrics import (
    MitigationMetrics,
    compute_mitigation_metrics,
)


@dataclass
class MitigationEvalResult:
    metrics: MitigationMetrics
    events_analysed: int = 0
    tokens_with_quality_score: int = 0
    quality_pass_rate: Optional[float] = None
    quality_scores: list[float] = field(default_factory=list)
    per_token_overhead_us: Optional[float] = None  # mean pipeline_ms / token * 1000

    def to_dict(self) -> dict:
        d = self.metrics.to_dict()
        d.update(
            {
                "events_analysed": self.events_analysed,
                "quality_pass_rate": (
                    round(self.quality_pass_rate, 4)
                    if self.quality_pass_rate is not None
                    else None
                ),
                "per_token_overhead_us": (
                    round(self.per_token_overhead_us, 3)
                    if self.per_token_overhead_us is not None
                    else None
                ),
            }
        )
        return d


# ---------------------------------------------------------- event-log reader
def _iter_events(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def from_event_log(event_log_path: str | Path) -> MitigationEvalResult:
    """Compute mitigation metrics from a RuntimeMonitor event log."""
    path = Path(event_log_path)
    if not path.exists():
        raise FileNotFoundError(path)

    all_audit: list[dict] = []
    quality_scores: list[float] = []
    pipeline_ms_list: list[float] = []
    prompt_len_list: list[int] = []

    for event in _iter_events(path):
        for rec in event.get("audit") or []:
            all_audit.append(rec)
        q = event.get("quality") or {}
        if "overall" in q:
            quality_scores.append(float(q["overall"]))
        if "pipeline_ms" in event:
            pipeline_ms_list.append(float(event["pipeline_ms"]))
        if "prompt_len" in event:
            prompt_len_list.append(int(event["prompt_len"]))

    metrics = compute_mitigation_metrics(all_audit)

    quality_pass_rate: Optional[float] = None
    if quality_scores:
        pass_threshold = 0.5
        quality_pass_rate = sum(1 for s in quality_scores if s >= pass_threshold) / len(
            quality_scores
        )
        metrics.quality_pass = sum(1 for s in quality_scores if s >= pass_threshold)
        metrics.quality_fail = len(quality_scores) - metrics.quality_pass

    # Mean pipeline_ms converted to µs per prompt token.
    overhead_us: Optional[float] = None
    if pipeline_ms_list and prompt_len_list and len(pipeline_ms_list) == len(prompt_len_list):
        per_event = [
            (ms * 1000) / max(n, 1)
            for ms, n in zip(pipeline_ms_list, prompt_len_list)
        ]
        overhead_us = sum(per_event) / len(per_event)

    return MitigationEvalResult(
        metrics=metrics,
        events_analysed=len(pipeline_ms_list),
        tokens_with_quality_score=len(quality_scores),
        quality_pass_rate=quality_pass_rate,
        quality_scores=quality_scores,
        per_token_overhead_us=overhead_us,
    )


def from_audit_list(
    audit_records: Sequence[dict],
    pipeline_ms: Optional[float] = None,
    total_tokens: Optional[int] = None,
) -> MitigationEvalResult:
    """Compute mitigation metrics from a flat list of audit dicts."""
    metrics = compute_mitigation_metrics(audit_records)
    overhead_us: Optional[float] = None
    if pipeline_ms is not None and total_tokens:
        overhead_us = (pipeline_ms * 1000) / max(total_tokens, 1)
    return MitigationEvalResult(
        metrics=metrics,
        events_analysed=1 if audit_records else 0,
        per_token_overhead_us=overhead_us,
    )
