"""Aggregate Magikarp + GlitchMiner candidate lists into a single candidate set.

Unified output schema (one dict per unique token_id):

    {
        "token_id": int,
        "token_str": str,
        "sources": ["magikarp", "glitchminer"],   # subset
        "magikarp_score": float | None,           # max_prob from verify, low=undertrained
        "magikarp_label": str | None,
        "magikarp_indicator": float | None,
        "glitchminer_score": float | None,        # 0 / 0.5 / 1.0
        "glitchminer_strict_verified": bool | None,
        "confidence": float,                      # [0,1], higher = more likely undertrained
    }

Confidence model (simple, tune later):
    - magikarp contributes `1 - magikarp_score` when present
    - glitchminer contributes `glitchminer_score` when present
    - confidence = weighted mean of whichever detectors fired
      (default weights: magikarp=0.45, glitchminer=0.55 per PDF: GlitchMiner
      reports +10.7% over Magikarp alone).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_WEIGHTS = {"magikarp": 0.45, "glitchminer": 0.55}


def _confidence(entry: dict, weights: dict[str, float]) -> float:
    parts: list[tuple[float, float]] = []
    if entry.get("magikarp_score") is not None:
        parts.append((weights["magikarp"], 1.0 - float(entry["magikarp_score"])))
    if entry.get("glitchminer_score") is not None:
        parts.append((weights["glitchminer"], float(entry["glitchminer_score"])))
    if not parts:
        return 0.0
    total_w = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_w


def aggregate_candidates(
    magikarp_results: list[dict],
    glitchminer_results: list[dict],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    merged: dict[int, dict] = {}

    for m in magikarp_results:
        tid = int(m["token_id"])
        merged[tid] = {
            "token_id": tid,
            "token_str": m.get("token_str"),
            "sources": ["magikarp"],
            "magikarp_score": m.get("magikarp_score"),
            "magikarp_label": m.get("magikarp_label"),
            "magikarp_indicator": m.get("magikarp_indicator"),
            "glitchminer_score": None,
            "glitchminer_strict_verified": None,
        }

    for g in glitchminer_results:
        tid = int(g["token_id"])
        if tid in merged:
            merged[tid]["sources"].append("glitchminer")
        else:
            merged[tid] = {
                "token_id": tid,
                "token_str": g.get("token_str_readable") or g.get("token_str"),
                "sources": ["glitchminer"],
                "magikarp_score": None,
                "magikarp_label": None,
                "magikarp_indicator": None,
            }
        merged[tid]["glitchminer_score"] = g.get("glitchminer_score")
        merged[tid]["glitchminer_strict_verified"] = g.get("glitchminer_strict_verified")

    for entry in merged.values():
        entry["confidence"] = _confidence(entry, weights)

    return sorted(merged.values(), key=lambda e: e["confidence"], reverse=True)


def write_jsonl(path: str | Path, candidates: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path
