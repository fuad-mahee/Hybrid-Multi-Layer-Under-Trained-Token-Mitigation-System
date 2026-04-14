"""Layer 5 — Continuous Learning & Feedback Loop.

Flow (PDF)::

    User Input -> Bloom -> Lookup -> Mitigation -> Monitor Output
                                                       |
                                                       v
                                                Quality Check
                                                       |
                                                       v
                                         Feedback Loop <-> Update Strategies

This package wires four concerns together:

* :class:`RuntimeMonitor`    — latency + audit event log around a
  :class:`~hmuttms.layer4_mitigation.MitigationPipeline`.
* :class:`QualityChecker`    — pluggable scoring of generated output
  (length / repetition / user-supplied judge).
* :class:`Escalator`         — consumes the event log, updates
  ``metadata.mitigation_attempts / _successes`` in the Layer-3 registry,
  and writes / promotes ``FallbackRule`` entries (Track 1 -> 2 -> 4 per
  PDF).
* :class:`RetestScheduler`   — periodically re-runs Layer 2 on stored
  tokens against newer Model-B backends and diffs the classifications.
"""
from .escalator import EscalationConfig, Escalator
from .monitor import MonitoredGeneration, RuntimeMonitor
from .quality import QualityChecker, QualityScore, default_scorers
from .retest_scheduler import RetestReport, RetestScheduler

__all__ = [
    "EscalationConfig",
    "Escalator",
    "MonitoredGeneration",
    "QualityChecker",
    "QualityScore",
    "RetestReport",
    "RetestScheduler",
    "RuntimeMonitor",
    "default_scorers",
]
