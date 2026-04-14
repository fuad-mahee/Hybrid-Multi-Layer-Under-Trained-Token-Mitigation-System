"""Layer 2 — Two-Model Cross-Validation Engine.

Takes Layer-1 candidate tokens, tests each against one or more *superior*
Model B backends, and classifies them via the decision matrix from the
HMUTTMS flowchart (PDF, Layer 2):

    * Both A and B fail            -> CONFIRMED under-trained -> blacklist
    * A fails, B succeeds          -> model-specific weakness -> substitute
    * Both succeed                 -> false positive          -> drop
"""
from .decision_matrix import Classification, DecisionConfig, classify
from .ensemble import ensemble_vote
from .model_b_tester import (
    BaseModelBTester,
    CachedFileTester,
    LocalHFTester,
    TokenProbe,
)

__all__ = [
    "BaseModelBTester",
    "CachedFileTester",
    "Classification",
    "DecisionConfig",
    "LocalHFTester",
    "TokenProbe",
    "classify",
    "ensemble_vote",
]
