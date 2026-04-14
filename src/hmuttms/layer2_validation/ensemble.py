"""Optional multi-Model-B ensemble (PDF Layer-2 enhancement).

Combines probes from 2-3 superior backends (e.g. GPT-4, Claude 3.5,
Gemma 2) into a single weighted score + majority verdict. Expected to
cut Layer-2 false positives by 50-70% per the flowchart.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .model_b_tester import TokenProbe


@dataclass
class EnsembleResult:
    token_id: int
    token_str: Optional[str]
    score: float                          # weighted mean in [0, 1]
    verdict: str                          # majority verdict
    backends: list[str] = field(default_factory=list)
    per_backend: list[dict] = field(default_factory=list)


def ensemble_vote(
    probes: Sequence[TokenProbe],
    weights: Optional[dict[str, float]] = None,
) -> EnsembleResult:
    """Combine probes (assumed to be for the same token) into one verdict.

    * ``score`` is the weighted mean of per-backend scores.
    * ``verdict`` is the weighted majority among
      ``{"handled", "undertrained", "inconclusive"}``; ties break toward
      ``"inconclusive"`` to stay conservative.
    """
    if not probes:
        raise ValueError("ensemble_vote requires at least one probe")

    token_id = probes[0].token_id
    token_str = probes[0].token_str
    weights = weights or {}

    weighted_score_num = 0.0
    weighted_score_den = 0.0
    verdict_weight: Counter[str] = Counter()
    per_backend: list[dict] = []

    for p in probes:
        w = float(weights.get(p.backend, 1.0))
        weighted_score_num += w * float(p.score)
        weighted_score_den += w
        verdict_weight[p.verdict] += w
        per_backend.append(
            {
                "backend": p.backend,
                "score": float(p.score),
                "verdict": p.verdict,
                "weight": w,
            }
        )

    score = weighted_score_num / weighted_score_den if weighted_score_den else 0.5

    # Weighted majority verdict.
    if verdict_weight:
        top = verdict_weight.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            verdict = "inconclusive"
        else:
            verdict = top[0][0]
    else:
        verdict = "inconclusive"

    return EnsembleResult(
        token_id=token_id,
        token_str=token_str,
        score=score,
        verdict=verdict,
        backends=[p.backend for p in probes],
        per_backend=per_backend,
    )
