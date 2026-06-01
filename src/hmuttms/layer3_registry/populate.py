"""Populate the Layer-3 registry from Layer-2 classification output.

Reads a JSONL file produced by
``hmuttms.layer2_validation.run_layer2`` and writes:

* ``BLACKLIST``      -> ``blacklist_tokens`` sub-db
* ``SUBSTITUTABLE``  -> ``substitution_map`` sub-db
* all processed rows -> ``metadata`` sub-db (severity + observed backends)

``FALSE_POSITIVE`` rows are skipped. ``REVIEW`` rows are skipped by
default; pass ``include_review=True`` to persist them as low-severity
metadata (useful when feeding the feedback loop).

After writing, the bloom filter covering the blacklist + substitution
keys can be refreshed (see :func:`rebuild_bloom`) so Layer-1's runtime
gate stays in sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from ..layer1_detection.bloom_filter import BloomGate
from .registry import TokenRegistry
from .schemas import (
    BlacklistEntry,
    MetadataEntry,
    ReplacementStrategy,
    SubstitutionEntry,
)


# ---------------------------------------------------------------- thresholds
def _severity_from_scores(
    model_a_score: float, model_b_score: Optional[float]
) -> str:
    a = float(model_a_score)
    b = 1.0 if model_b_score is None else float(model_b_score)
    # Lower scores = worse == higher severity.
    worst = min(a, b)
    if worst < 0.15:
        return "high"
    if worst < 0.35:
        return "medium"
    return "low"


def _iter_layer2(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ======================================================================= core
def populate_from_layer2(
    layer2_path: str | Path,
    registry: TokenRegistry,
    *,
    include_review: bool = False,
) -> dict[str, int]:
    """Ingest a Layer-2 JSONL into ``registry``.

    Returns per-class insert counts.
    """
    path = Path(layer2_path)
    if not path.exists():
        raise FileNotFoundError(path)

    counts = {
        "BLACKLIST": 0,
        "SUBSTITUTABLE": 0,
        "FALSE_POSITIVE": 0,
        "REVIEW": 0,
        "metadata_written": 0,
    }

    for row in _iter_layer2(path):
        tid = int(row["token_id"])
        token_str = row.get("token_str")
        cls = row.get("classification", "REVIEW")
        a_score = float(row["model_a_score"])
        b_score = row.get("model_b_score")
        backends = list(row.get("backends") or [])
        sources = list(row.get("layer1_sources") or [])
        severity = _severity_from_scores(a_score, b_score)

        # -------------------- sub-DB writes ---------------------
        if cls == "BLACKLIST":
            registry.put_blacklist(
                BlacklistEntry(
                    token_id=tid,
                    token_string=token_str,
                    model_a_score=a_score,
                    model_b_score=b_score,
                    severity=severity,
                    sources=sources,
                    backends=backends,
                )
            )
            counts["BLACKLIST"] += 1

        elif cls == "SUBSTITUTABLE":
            strategy_dict = row.get("replacement_strategy") or {}
            # Inject Model B's embedding into the strategy so Track 3 can use it.
            if row.get("model_b_embedding") and not strategy_dict.get("fallback_embedding"):
                strategy_dict = dict(strategy_dict)
                strategy_dict["fallback_embedding"] = row["model_b_embedding"]
            registry.put_substitution(
                SubstitutionEntry(
                    token_id=tid,
                    token_string=token_str,
                    model_a_score=a_score,
                    model_b_score=b_score,
                    replacement_strategy=ReplacementStrategy.from_dict(strategy_dict),
                    sources=sources,
                    backends=backends,
                )
            )
            counts["SUBSTITUTABLE"] += 1

        elif cls == "FALSE_POSITIVE":
            counts["FALSE_POSITIVE"] += 1
            continue

        else:  # REVIEW
            counts["REVIEW"] += 1
            if not include_review:
                continue

        # -------------------- metadata write --------------------
        registry.upsert_metadata(
            MetadataEntry(
                token_id=tid,
                token_string=token_str,
                severity=severity,
                frequency=1,
                observed_in_models=backends,
                embedding=row.get("model_b_embedding"),
            )
        )
        counts["metadata_written"] += 1

    return counts


# =========================================================== bloom refresh
def rebuild_bloom(
    registry: TokenRegistry,
    bloom_path: str | Path,
    *,
    target_fpr: float = 0.001,
    include_substitution: bool = True,
) -> dict:
    """Rebuild the Layer-1 bloom filter from current registry contents.

    The bloom is the *runtime* fast-reject gate, so its membership should
    mirror the union of ``blacklist_tokens`` (always) and, optionally,
    ``substitution_map`` (so substitutable tokens also get caught before
    they hit the model raw).
    """
    ids: set[int] = {e.token_id for e in registry.iter_blacklist()}
    if include_substitution:
        ids.update(e.token_id for e in registry.iter_substitution())

    gate = BloomGate.build_from_ids(ids, target_fpr=target_fpr)
    gate.save(bloom_path)

    return {
        "output": str(bloom_path),
        "token_count": len(ids),
        "num_bits": gate.num_bits,
        "num_hashes": gate.num_hashes,
        "target_fpr": target_fpr,
        "expected_fpr": gate.expected_fpr,
        "include_substitution": include_substitution,
    }
