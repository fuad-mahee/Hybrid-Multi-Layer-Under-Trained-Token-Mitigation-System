"""Runtime monitor for Layer-4 mitigation.

Wraps a :class:`~hmuttms.layer4_mitigation.MitigationPipeline` and records
per-call telemetry to a JSONL event log. Each event contains:

    {
      "ts": "2026-04-14T12:34:56Z",
      "prompt_len": 128,
      "pipeline_ms": 0.42,              # wall time of pipeline.mitigate()
      "generation_ms": 180.3,           # only if generation is timed too
      "pipeline_stats": {...},          # PipelineStats.to_dict()
      "audit": [...],                   # per-token MitigationResult dicts
      "quality": {...} | None,          # optional QualityChecker output
      "session_id": "..."               # optional, set via constructor
    }

The log is the primary input to :class:`Escalator` — so keep every row
self-contained (no references to in-memory state).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class MonitoredGeneration:
    """Return value of :meth:`RuntimeMonitor.run`."""

    new_token_ids: list[int]
    audit: list[dict]
    extras: dict
    pipeline_ms: float
    generation_ms: Optional[float]
    quality: Optional[dict]
    event: dict


class RuntimeMonitor:
    def __init__(
        self,
        pipeline,
        event_log_path: str | Path = "data/mitigation_system/feedback/events.jsonl",
        quality_checker: Optional[Any] = None,
        session_id: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.path = Path(event_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.quality = quality_checker
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._fh = self.path.open("a", encoding="utf-8")

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "RuntimeMonitor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ run
    def run(
        self,
        token_ids: Sequence[int],
        *,
        generated_text: Optional[str] = None,
        generation_ms: Optional[float] = None,
        extra_context: Optional[dict] = None,
    ) -> MonitoredGeneration:
        """Run the pipeline once and emit an event.

        The caller is responsible for the actual ``model.generate`` call
        and for timing it; pass the output text back here (with
        ``generated_text`` / ``generation_ms``) so Layer 5 can score it.
        """
        t0 = time.perf_counter()
        new_ids, audit, extras = self.pipeline.mitigate(token_ids)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        quality_dict: Optional[dict] = None
        if self.quality is not None and generated_text is not None:
            quality_dict = self.quality.score(generated_text, context=extra_context).to_dict()

        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "prompt_len": len(token_ids),
            "mitigated_len": len(new_ids),
            "pipeline_ms": round(pipeline_ms, 4),
            "generation_ms": (
                round(float(generation_ms), 4) if generation_ms is not None else None
            ),
            "pipeline_stats": self.pipeline.stats.to_dict(),
            "audit": audit,
            "quality": quality_dict,
        }
        if extra_context:
            event["context"] = extra_context

        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

        return MonitoredGeneration(
            new_token_ids=new_ids,
            audit=audit,
            extras=extras,
            pipeline_ms=pipeline_ms,
            generation_ms=generation_ms,
            quality=quality_dict,
            event=event,
        )
