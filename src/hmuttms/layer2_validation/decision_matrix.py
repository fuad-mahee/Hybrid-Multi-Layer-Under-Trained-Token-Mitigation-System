"""Decision matrix from the HMUTTMS flowchart (Layer 2).

Inputs per token:
    * ``model_a_score``  — how well Model A (target) handles it, in [0, 1].
                           Derived from Layer-1 Magikarp ``max_prob``-style
                           signal (``1 - magikarp_score``) or the Layer-1
                           aggregated confidence inverted.
    * ``model_b_score``  — how well Model B (superior) handles it, in [0, 1].
                           Produced by a :class:`TokenProbe` in Layer 2.

Rules with current default thresholds:
    * A < 0.3  AND  B < 0.5  -> ``BLACKLIST``         (fails both score thresholds)
    * A < 0.3  AND  B > 0.6  -> ``SUBSTITUTABLE``     (model-specific weakness)
    * A >= 0.3 AND  B > 0.6  -> ``FALSE_POSITIVE``    (drop)
    * anything else          -> ``REVIEW``            (inconclusive band)

Thresholds are configurable via :class:`DecisionConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Classification(str, Enum):
    BLACKLIST = "BLACKLIST"
    SUBSTITUTABLE = "SUBSTITUTABLE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class DecisionConfig:
    a_fail_below: float = 0.30      # A < this -> Model A fails
    b_fail_below: float = 0.50      # B < this -> Model B also fails
    b_pass_above: float = 0.60      # B > this -> Model B handles it well (lowered from 0.70 to capture 0.667 probe scores)


def classify(
    model_a_score: float,
    model_b_score: Optional[float],
    cfg: DecisionConfig | None = None,
) -> Classification:
    """Apply the configured decision matrix to a single token."""
    cfg = cfg or DecisionConfig()
    a = float(model_a_score)
    # Missing Model-B result: cannot distinguish blacklist vs. substitutable.
    if model_b_score is None:
        return Classification.REVIEW
    b = float(model_b_score)

    a_fails = a < cfg.a_fail_below
    b_fails = b < cfg.b_fail_below
    b_passes = b > cfg.b_pass_above

    if a_fails and b_fails:
        return Classification.BLACKLIST
    if a_fails and b_passes:
        return Classification.SUBSTITUTABLE
    if (not a_fails) and b_passes:
        return Classification.FALSE_POSITIVE
    return Classification.REVIEW


def model_a_score_from_layer1(entry: dict) -> float:
    """Derive a Model-A score in [0, 1] from a Layer-1 aggregated entry.

    Prefers the Magikarp ``max_prob`` style signal when present (higher =
    model handles the token better). Falls back to ``1 - confidence`` so a
    highly-confident Layer-1 hit maps to a low Model-A score.
    """
    if entry.get("magikarp_score") is not None:
        return float(entry["magikarp_score"])
    conf = float(entry.get("confidence", 0.0))
    return max(0.0, min(1.0, 1.0 - conf))
