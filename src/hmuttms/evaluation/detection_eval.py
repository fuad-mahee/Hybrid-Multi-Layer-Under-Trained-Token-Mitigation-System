"""Detection evaluation — compare Magikarp-only baseline vs HMUTTMS Layer-1.

Ground-truth can come from three sources (in priority order):

1. A JSONL file where each line has ``{"token_id": N, "is_undertrained": bool}``.
2. The Layer-2 classification output (``data/layer2_classified.jsonl``):
   ``BLACKLIST`` + ``SUBSTITUTABLE`` rows count as positive ground truth;
   ``FALSE_POSITIVE`` rows count as negative.
3. The pre-collected Kimi K2 verification file
   (``data/kimi_k2_magikarp_results.json``): ``is_undertrained`` field.

Baseline: tokens that appear in ``data/magikarp_comparison.csv``
(GPT-2 Model-A results), which contains Magikarp-only detections.

Hybrid: tokens that appear in ``data/layer1_candidates.jsonl``
(Magikarp + GlitchMiner aggregated, built by Layer 1).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .metrics import (
    DetectionMetrics,
    compute_detection_metrics,
    detection_improvement,
)


# ============================================================ data loaders
def _load_ground_truth_jsonl(path: Path) -> dict[int, bool]:
    gt: dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt[int(row["token_id"])] = bool(row["is_undertrained"])
    return gt


def _load_ground_truth_from_layer2(path: Path) -> dict[int, bool]:
    """Layer-2 JSONL as ground truth: BLACKLIST/SUBSTITUTABLE=True, FP=False."""
    gt: dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cls = row.get("classification", "REVIEW")
            if cls in ("BLACKLIST", "SUBSTITUTABLE"):
                gt[int(row["token_id"])] = True
            elif cls == "FALSE_POSITIVE":
                gt[int(row["token_id"])] = False
    return gt


def _load_ground_truth_from_kimi(path: Path) -> dict[int, bool]:
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    return {
        int(r["token_id"]): bool(r.get("is_undertrained", False))
        for r in blob.get("results", [])
    }


def load_ground_truth(path: str | Path) -> dict[int, bool]:
    """Auto-detect file type and return ``{token_id: is_undertrained}``."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return _load_ground_truth_jsonl(p)
    if suffix == ".json":
        blob = json.loads(p.read_text(encoding="utf-8"))
        if "results" in blob:
            return _load_ground_truth_from_kimi(p)
        return {int(k): bool(v) for k, v in blob.items()}
    raise ValueError(f"Unsupported ground-truth file type: {suffix}")


def _load_magikarp_csv(path: Path) -> set[int]:
    """Return token_ids from the Magikarp CSV baseline."""
    flagged: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("token_id") or row.get("id")
            if tid is not None:
                try:
                    flagged.add(int(tid))
                except ValueError:
                    pass
    return flagged


def _load_layer1_jsonl(path: Path, min_confidence: float = 0.0) -> set[int]:
    """Return token_ids from the Layer-1 aggregated candidates JSONL."""
    flagged: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if float(row.get("confidence", 0.0)) >= min_confidence:
                flagged.add(int(row["token_id"]))
    return flagged


# ============================================================ eval runner
@dataclass
class DetectionEvalResult:
    baseline_metrics: DetectionMetrics
    hybrid_metrics: DetectionMetrics
    improvement: dict
    ground_truth_size: int
    baseline_flags: int
    hybrid_flags: int
    universe: int


def run_detection_eval(
    ground_truth: dict[int, bool],
    baseline_flagged: set[int],
    hybrid_flagged: set[int],
    token_universe: Optional[set[int]] = None,
) -> DetectionEvalResult:
    """Compare baseline vs. hybrid on a shared ground-truth set.

    ``token_universe`` — the full vocabulary to use as the negative pool
    (i.e. tokens NOT in ground_truth that should be True). If None,
    uses the union of ground_truth keys + both flagged sets.
    """
    if token_universe is None:
        token_universe = (
            set(ground_truth.keys()) | baseline_flagged | hybrid_flagged
        )

    all_ids = sorted(token_universe)
    gt_list  = [bool(ground_truth.get(t, False)) for t in all_ids]
    base_list = [t in baseline_flagged for t in all_ids]
    hyb_list  = [t in hybrid_flagged   for t in all_ids]

    base_m = compute_detection_metrics(base_list, gt_list)
    hyb_m  = compute_detection_metrics(hyb_list,  gt_list)

    return DetectionEvalResult(
        baseline_metrics=base_m,
        hybrid_metrics=hyb_m,
        improvement=detection_improvement(base_m, hyb_m),
        ground_truth_size=sum(gt_list),
        baseline_flags=len(baseline_flagged),
        hybrid_flags=len(hybrid_flagged),
        universe=len(all_ids),
    )


def run_detection_eval_from_files(
    ground_truth_path: str | Path,
    *,
    magikarp_csv: str | Path = "data/magikarp_comparison.csv",
    layer1_jsonl: str | Path = "data/layer1_candidates.jsonl",
    min_confidence: float = 0.0,
) -> DetectionEvalResult:
    """Convenience wrapper: load all three files and return the result."""
    gt = load_ground_truth(ground_truth_path)
    baseline = _load_magikarp_csv(Path(magikarp_csv))
    hybrid = _load_layer1_jsonl(Path(layer1_jsonl), min_confidence)
    return run_detection_eval(gt, baseline, hybrid)
