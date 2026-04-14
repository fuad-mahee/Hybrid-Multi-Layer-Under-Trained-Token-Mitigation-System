"""Quality check for mitigated generations.

Three lightweight built-in scorers (``length_ok``, ``repetition``,
``char_diversity``) are combined by default. Add domain-specific scorers
— or an LLM-judge — via the ``extra_scorers`` argument::

    def judge_with_model_b(text, ctx):
        return ("model_b_judge", float(some_api_call(text)))
    qc = QualityChecker(extra_scorers=[judge_with_model_b])

Every scorer returns ``(name, score_in_0_1)``; ``QualityChecker.score``
reports the per-scorer values *and* a weighted mean as ``overall``.
``success`` is true iff ``overall >= pass_threshold``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


Scorer = Callable[[str, Optional[dict]], tuple[str, float]]


# ============================================================ built-in scorers
def length_ok(text: str, ctx: Optional[dict]) -> tuple[str, float]:
    """1.0 for replies >= 16 chars, linear up from 0 for shorter."""
    n = len(text or "")
    return "length_ok", max(0.0, min(1.0, n / 16.0))


_WORD_RE = re.compile(r"\w+")


def repetition(text: str, ctx: Optional[dict]) -> tuple[str, float]:
    """Penalise repeated-token stutter. 1.0 = no stutter, 0.0 = only one word."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < 4:
        return "repetition", 1.0
    unique_ratio = len(set(words)) / len(words)
    return "repetition", float(unique_ratio)


def char_diversity(text: str, ctx: Optional[dict]) -> tuple[str, float]:
    """Reject degenerate runs (e.g. 'aaaaaaaa')."""
    if not text:
        return "char_diversity", 0.0
    unique = len(set(text))
    return "char_diversity", min(1.0, unique / 20.0)


def default_scorers() -> list[Scorer]:
    return [length_ok, repetition, char_diversity]


# =========================================================== scoring dataclass
@dataclass
class QualityScore:
    overall: float
    success: bool
    per_scorer: dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "success": self.success,
            "pass_threshold": self.pass_threshold,
            "per_scorer": dict(self.per_scorer),
        }


# ================================================================== checker
class QualityChecker:
    def __init__(
        self,
        scorers: Optional[Sequence[Scorer]] = None,
        weights: Optional[dict[str, float]] = None,
        pass_threshold: float = 0.5,
        extra_scorers: Optional[Sequence[Scorer]] = None,
    ):
        self.scorers: list[Scorer] = list(scorers or default_scorers())
        if extra_scorers:
            self.scorers.extend(extra_scorers)
        self.weights = weights or {}
        self.pass_threshold = float(pass_threshold)

    def score(self, text: str, context: Optional[dict] = None) -> QualityScore:
        per: dict[str, float] = {}
        num = 0.0
        den = 0.0
        for scorer in self.scorers:
            name, value = scorer(text, context)
            per[name] = float(value)
            w = float(self.weights.get(name, 1.0))
            num += w * float(value)
            den += w
        overall = num / den if den else 0.0
        return QualityScore(
            overall=overall,
            success=overall >= self.pass_threshold,
            per_scorer=per,
            pass_threshold=self.pass_threshold,
        )
