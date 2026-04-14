"""Collate all evaluation metrics into a structured report.

The report is the single artefact submitted with the thesis. It records:

    * Experiment metadata (model ids, date, dataset sizes)
    * Detection section   — baseline vs. hybrid metrics + improvement
    * Mitigation section  — per-track rates + quality pass rate
    * Latency section     — bloom / RocksDB / pipeline overhead
    * Verdict table       — PASS / FAIL / N/A per PDF target

PDF targets (stored in :data:`PDF_TARGETS`)::

    {
        "detection_accuracy_improvement_pp":  [25.0, 35.0],   # percentage points
        "fp_reduction_pct":                   [50.0, 70.0],   # percent reduction
        "mitigation_success_rate":            [0.60, 0.75],   # fraction
        "inference_overhead_per_token_us":    [0.0,  20.0],   # microseconds
        "quality_pass_rate":                  [0.80,  1.0],   # fraction (our own target)
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PDF_TARGETS: dict[str, list[float]] = {
    "detection_accuracy_improvement_pp": [25.0, 35.0],
    "fp_reduction_pct":                  [50.0, 70.0],
    "mitigation_success_rate":           [0.60, 0.75],
    "inference_overhead_per_token_us":   [0.0,  20.0],
    "quality_pass_rate":                 [0.80,  1.0],
}

_VERDICT_LABELS = {True: "PASS", False: "FAIL", None: "N/A"}


def _in_range(value: Optional[float], lo: float, hi: float) -> Optional[bool]:
    if value is None:
        return None
    return bool(lo <= value <= hi)


def _meets_min(value: Optional[float], lo: float) -> Optional[bool]:
    if value is None:
        return None
    return bool(value >= lo)


# ==================================================================== report
@dataclass
class EvalReport:
    model_a_id: str = ""
    model_b_ids: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    dataset_info: dict[str, Any] = field(default_factory=dict)
    detection: Optional[dict] = None
    mitigation: Optional[dict] = None
    latency: Optional[dict] = None

    # ---------------------------------------------------------------- attach
    def attach_detection(self, result) -> None:
        """Attach a :class:`~hmuttms.evaluation.detection_eval.DetectionEvalResult`."""
        imp = result.improvement
        self.detection = {
            "baseline": result.baseline_metrics.to_dict(),
            "hybrid":   result.hybrid_metrics.to_dict(),
            "improvement": imp,
            "ground_truth_size": result.ground_truth_size,
            "baseline_flags":    result.baseline_flags,
            "hybrid_flags":      result.hybrid_flags,
            "universe":          result.universe,
        }

    def attach_mitigation(self, result) -> None:
        """Attach a :class:`~hmuttms.evaluation.mitigation_eval.MitigationEvalResult`."""
        self.mitigation = result.to_dict()

    def attach_latency(self, bench_dict: dict) -> None:
        self.latency = bench_dict

    # --------------------------------------------------------------- verdict
    def verdicts(self) -> dict[str, str]:
        v: dict[str, str] = {}

        # Detection accuracy improvement.
        acc_pp = (
            self.detection["improvement"].get("accuracy_pp")
            if self.detection
            else None
        )
        lo, hi = PDF_TARGETS["detection_accuracy_improvement_pp"]
        v["detection_accuracy_improvement_pp"] = _VERDICT_LABELS[
            _meets_min(acc_pp, lo)
        ]

        # FP reduction %.
        fp_pct = (
            self.detection["improvement"].get("fp_reduction_pct")
            if self.detection
            else None
        )
        lo, _ = PDF_TARGETS["fp_reduction_pct"]
        v["fp_reduction_pct"] = _VERDICT_LABELS[_meets_min(fp_pct, lo)]

        # Mitigation success rate.
        mit_rate = (
            self.mitigation.get("overall_success_rate")
            if self.mitigation
            else None
        )
        lo, _ = PDF_TARGETS["mitigation_success_rate"]
        v["mitigation_success_rate"] = _VERDICT_LABELS[_meets_min(mit_rate, lo)]

        # Inference overhead.
        overhead_us: Optional[float] = None
        if self.latency:
            overhead_us = (
                self.latency.get("pipeline_per_token_us") or {}
            ).get("mean_us")
        if overhead_us is None and self.mitigation:
            overhead_us = self.mitigation.get("per_token_overhead_us")
        _, hi = PDF_TARGETS["inference_overhead_per_token_us"]
        v["inference_overhead_per_token_us"] = _VERDICT_LABELS[
            (None if overhead_us is None else bool(overhead_us <= hi))
        ]

        # Quality pass rate.
        qpr = self.mitigation.get("quality_pass_rate") if self.mitigation else None
        lo, _ = PDF_TARGETS["quality_pass_rate"]
        v["quality_pass_rate"] = _VERDICT_LABELS[_meets_min(qpr, lo)]

        return v

    # ---------------------------------------------------------------- render
    def to_dict(self) -> dict:
        return {
            "model_a_id":    self.model_a_id,
            "model_b_ids":   self.model_b_ids,
            "timestamp":     self.timestamp,
            "dataset_info":  self.dataset_info,
            "detection":     self.detection,
            "mitigation":    self.mitigation,
            "latency":       self.latency,
            "pdf_targets":   PDF_TARGETS,
            "verdicts":      self.verdicts(),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    # ------------------------------------------------------- pretty-print
    def print_summary(self) -> None:
        vd = self.verdicts()
        w = 44
        print(f"\n{'─'*w}")
        print(f"  HMUTTMS Evaluation Report  —  {self.timestamp}")
        print(f"  Model A : {self.model_a_id or '(not set)'}")
        if self.model_b_ids:
            print(f"  Model B : {', '.join(self.model_b_ids)}")
        print(f"{'─'*w}")
        print(f"  {'Metric':<38} {'Verdict'}")
        print(f"{'─'*w}")
        labels = {
            "detection_accuracy_improvement_pp": "Detection accuracy improvement (pp)",
            "fp_reduction_pct":                  "False-positive reduction (%)",
            "mitigation_success_rate":           "Mitigation success rate",
            "inference_overhead_per_token_us":   "Inference overhead (µs / token)",
            "quality_pass_rate":                 "Quality pass rate",
        }
        for key, label in labels.items():
            print(f"  {label:<38} {vd.get(key, 'N/A')}")
        print(f"{'─'*w}\n")

        if self.detection:
            imp = self.detection["improvement"]
            print("  Detection detail:")
            print(f"    Baseline  F1={self.detection['baseline']['f1']}  "
                  f"FPR={self.detection['baseline']['false_positive_rate']}")
            print(f"    Hybrid    F1={self.detection['hybrid']['f1']}  "
                  f"FPR={self.detection['hybrid']['false_positive_rate']}")
            print(f"    Δ accuracy = {imp.get('accuracy_pp', 'N/A')} pp  "
                  f"FP reduction = {imp.get('fp_reduction_pct', 'N/A')} %")
        if self.mitigation:
            print("\n  Mitigation detail:")
            for track, td in (self.mitigation.get("per_track") or {}).items():
                print(f"    {track:<28}  "
                      f"attempts={td['attempts']}  "
                      f"success_rate={td['rate']:.2%}")
        print()
