"""Core metric computations for the HMUTTMS evaluation harness.

All functions accept plain Python lists / dicts so they work without
any ML framework dependency. NumPy is used optionally for percentile
latency stats (falls back to a pure-Python implementation).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ============================================================== detection
@dataclass
class DetectionMetrics:
    """Binary classification metrics for under-trained token detection."""

    tp: int = 0        # predicted under-trained  AND  truly under-trained
    fp: int = 0        # predicted under-trained  BUT  normal token
    fn: int = 0        # missed under-trained token
    tn: int = 0        # correctly identified normal token

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / max(total, 1)

    @property
    def false_positive_rate(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @property
    def false_negative_rate(self) -> float:
        return self.fn / max(self.fn + self.tp, 1)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


def compute_detection_metrics(
    predictions: Sequence[bool],
    ground_truth: Sequence[bool],
) -> DetectionMetrics:
    """Compute detection metrics from parallel boolean sequences.

    ``predictions[i]`` = True means token i was flagged as under-trained.
    ``ground_truth[i]`` = True means token i is genuinely under-trained.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must be the same length")
    m = DetectionMetrics()
    for pred, true in zip(predictions, ground_truth):
        if pred and true:
            m.tp += 1
        elif pred and not true:
            m.fp += 1
        elif not pred and true:
            m.fn += 1
        else:
            m.tn += 1
    return m


def detection_improvement(
    baseline: DetectionMetrics,
    improved: DetectionMetrics,
) -> dict:
    """Express ``improved`` relative to ``baseline`` in percentage points."""
    return {
        "accuracy_pp":   round((improved.accuracy  - baseline.accuracy)  * 100, 2),
        "precision_pp":  round((improved.precision - baseline.precision) * 100, 2),
        "recall_pp":     round((improved.recall    - baseline.recall)    * 100, 2),
        "f1_pp":         round((improved.f1        - baseline.f1)        * 100, 2),
        "fp_rate_change_pp": round(
            (improved.false_positive_rate - baseline.false_positive_rate) * 100, 2
        ),
        "fp_reduction_pct": round(
            (1 - improved.fp / max(baseline.fp, 1)) * 100, 2
        ) if baseline.fp else None,
    }


# ============================================================= mitigation
@dataclass
class MitigationMetrics:
    """Success/failure counts for Track 3 (Embedding Injection)."""

    total_flagged: int = 0
    embedding_attempts: int = 0
    embedding_success: int = 0
    quality_pass: int = 0
    quality_fail: int = 0
    quality_na: int = 0

    @property
    def overall_success_rate(self) -> float:
        return self.embedding_success / max(self.total_flagged, 1)

    @property
    def quality_pass_rate(self) -> Optional[float]:
        checked = self.quality_pass + self.quality_fail
        if checked == 0:
            return None
        return self.quality_pass / checked

    def per_track(self) -> dict:
        rate = round(self.embedding_success / max(self.embedding_attempts, 1), 4)
        return {
            "embedding_injection": {
                "attempts": self.embedding_attempts,
                "successes": self.embedding_success,
                "rate": rate,
            },
        }

    def to_dict(self) -> dict:
        return {
            "total_flagged": self.total_flagged,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "quality_pass_rate": (
                round(self.quality_pass_rate, 4)
                if self.quality_pass_rate is not None
                else None
            ),
            "per_track": self.per_track(),
        }


def compute_mitigation_metrics(audit_records: Sequence[dict]) -> MitigationMetrics:
    """Build :class:`MitigationMetrics` from a flat list of audit dicts.

    Each dict is a :class:`~hmuttms.layer4_mitigation.base.MitigationResult`
    serialised via ``to_dict()``.
    """
    m = MitigationMetrics()
    seen_flagged: set = set()
    for rec in audit_records:
        action = rec.get("action", "")
        if action == "pass_through":
            continue
        tid = rec.get("original_token_id")
        if tid not in seen_flagged:
            seen_flagged.add(tid)
            m.total_flagged += 1
        track = rec.get("track") or ""
        if track == "track3_embedding_injection":
            m.embedding_attempts += 1
            if rec.get("success"):
                m.embedding_success += 1
    return m


# ================================================================ latency
@dataclass
class LatencyStats:
    """Descriptive statistics for a sequence of latency measurements (µs)."""

    samples: list[float] = field(default_factory=list)

    def add(self, µs: float) -> None:
        self.samples.append(float(µs))

    def _sorted(self) -> list[float]:
        return sorted(self.samples)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = self._sorted()
        idx = (p / 100) * (len(s) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_us": round(self.mean, 3),
            "median_us": round(self.median, 3),
            "stdev_us": round(self.stdev, 3),
            "p50_us": round(self.percentile(50), 3),
            "p95_us": round(self.percentile(95), 3),
            "p99_us": round(self.percentile(99), 3),
            "min_us": round(min(self.samples), 3) if self.samples else 0,
            "max_us": round(max(self.samples), 3) if self.samples else 0,
        }
