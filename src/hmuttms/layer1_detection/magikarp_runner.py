"""Run Magikarp detection on Model A and return a canonical candidate list.

Wraps magikarp.fishing.load_analyzers + candidates_for_verification +
verify_magikarp so Layer 1 can consume results in the unified schema:

    {
        "token_id": int,
        "token_str": str,
        "magikarp_score": float,    # max_prob across verification prompts, [0,1]
        "magikarp_indicator": float, # raw under-training indicator (lower = worse)
        "magikarp_label": str,       # strong_verified / weak_verified / weak_rejected / ...
        "source": ["magikarp"],
    }

Lower magikarp_score ⇒ more likely under-trained (model failed to repeat token).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from magikarp.fishing import (
    classify_verification,
    load_analyzers,
    verify_magikarp,
)
from magikarp.unused_tokens import UNUSED_TOKENS
from magikarp.utils import (
    DEFAULT_THRESHOLD_PERCENTILE,
    candidates_for_verification,
    select_placeholder_token,
    write_verification_results,
)


def run_magikarp(
    model_id: str,
    device: str = "cpu",
    threshold: Optional[float] = None,
    threshold_ratio: float = DEFAULT_THRESHOLD_PERCENTILE,
    trust_remote_code: bool = False,
    indicator_ix: int = 0,
    verify_missing: bool = True,
) -> list[dict]:
    """Run Magikarp on `model_id` and return candidates in the canonical schema.

    If cached verification results exist under magikarp's default store, they are
    reused and the model is only loaded when new verifications are needed.
    """
    known_unused = UNUSED_TOKENS.get(model_id, [])
    toka, moda, token_infos = load_analyzers(
        model_id,
        avoid_loading_model=not verify_missing,
        known_unused_tokens=known_unused,
        load_verifications=True,
        trust_remote_code=trust_remote_code,
        indicator_ix=indicator_ix,
    )

    candidates, _ = candidates_for_verification(
        token_infos, threshold_ratio=threshold_ratio, threshold=threshold
    )

    if verify_missing and moda is not None:
        remaining = [tc for tc in candidates if "verification" not in tc]
        if remaining:
            build_prompt_token = select_placeholder_token(toka, token_infos)
            moda.model.to(device)
            for ti in remaining:
                ti["verification"], _ = verify_magikarp(toka, moda, ti, build_prompt_token)
                classify_verification(ti)
            write_verification_results(token_infos, model_id)

    results: list[dict] = []
    for ti in candidates:
        if "verification" not in ti:
            continue
        max_prob = float(np.max([vr["max_prob"] for vr in ti["verification"]]))
        results.append(
            {
                "token_id": int(ti["i"]),
                "token_str": toka.vocab_to_readable_string(ti["i"]),
                "magikarp_score": max_prob,
                "magikarp_indicator": float(ti.get("main_indicator", float("nan"))),
                "magikarp_label": ti.get("magikarp", "unknown"),
                "source": ["magikarp"],
            }
        )
    return results
