"""Step 7 — Evaluation Harness.

Measures the HMUTTMS system against the PDF's target metrics:

    ┌─────────────────────────────────┬───────────────────┬──────────────┐
    │ Metric                          │ PDF target        │ Baseline     │
    ├─────────────────────────────────┼───────────────────┼──────────────┤
    │ Detection accuracy improvement  │ +25–35 %          │ Magikarp only│
    │ False-positive reduction        │ 50–70 % fewer     │ Layer-1 only │
    │ Mitigation success rate         │ 60–75 %           │ no mitigation│
    │ Inference overhead (per token)  │ < 20 µs           │ raw model    │
    │ Vocabulary scalability          │ 100K–500K tokens  │ —            │
    └─────────────────────────────────┴───────────────────┴──────────────┘

Sub-modules
-----------
metrics          — core computation functions (precision, recall, F1, …)
detection_eval   — compare single-detector vs. hybrid Layer-1 detector
mitigation_eval  — per-track success rates + quality scoring
latency_eval     — micro-benchmark Bloom, RocksDB, full pipeline
report           — collate every metric into a structured report dict
run_eval         — CLI entry-point (sub-commands: detect, mitigate, latency, full)
"""
from .metrics import (
    DetectionMetrics,
    MitigationMetrics,
    compute_detection_metrics,
    compute_mitigation_metrics,
)
from .report import EvalReport, PDF_TARGETS

__all__ = [
    "DetectionMetrics",
    "EvalReport",
    "MitigationMetrics",
    "PDF_TARGETS",
    "compute_detection_metrics",
    "compute_mitigation_metrics",
]
