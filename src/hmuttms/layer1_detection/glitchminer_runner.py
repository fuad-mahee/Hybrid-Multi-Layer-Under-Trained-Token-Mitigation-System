"""Run GlitchMiner detection on Model A and return a canonical candidate list.

Canonical schema:

    {
        "token_id": int,
        "token_str": str,
        "glitchminer_score": float,  # 1.0 = confirmed glitch by strict verify, else 0.0
        "glitchminer_raw_hit": bool, # hit by mining loop before strict verify
        "source": ["glitchminer"],
    }

GlitchMiner does not emit per-token continuous scores the way Magikarp does; it
yields a binary verdict (glitch / not-glitch). We surface both the raw mining
hit and the strict-verification result so the aggregator can weight them.
"""
from __future__ import annotations

from typing import Optional

from glitchminer import (
    GlitchMiner,
    initialize_model_and_tokenizer,
    strictly_glitch_verification,
)
from glitchminer.tokenfilter import TokenFilter


def run_glitchminer(
    model_path: str,
    num_iterations: int = 50,
    batch_size: int = 8,
    k: int = 32,
    device: str = "auto",
    quant_type: str = "bfloat16",
    skip_tokens: Optional[list[int]] = None,
    strict_verify: bool = True,
    print_language: str = "ENG",
    if_print: bool = False,
) -> list[dict]:
    """Run GlitchMiner mining + (optional) strict verification on `model_path`."""
    model, tokenizer = initialize_model_and_tokenizer(
        model_path, device=device, quant_type=quant_type
    )

    glitch_tokens, glitch_token_ids = GlitchMiner(
        model,
        tokenizer,
        num_iterations=num_iterations,
        batch_size=batch_size,
        k=k,
        if_print=if_print,
        print_language=print_language,
        skip_tokens=skip_tokens,
    )

    # Map raw hits into canonical dicts keyed by token_id.
    tf = TokenFilter(model, tokenizer)
    by_id: dict[int, dict] = {}
    for tok, tok_id in zip(glitch_tokens, glitch_token_ids):
        by_id[int(tok_id)] = {
            "token_id": int(tok_id),
            "token_str": tok,
            "glitchminer_score": 0.0,
            "glitchminer_raw_hit": True,
            "glitchminer_strict_verified": False,
            "source": ["glitchminer"],
        }

    if strict_verify and glitch_token_ids:
        _, verified_ids = strictly_glitch_verification(model, tokenizer, glitch_token_ids)
        for vid in verified_ids:
            if vid in by_id:
                by_id[vid]["glitchminer_strict_verified"] = True
                by_id[vid]["glitchminer_score"] = 1.0
    else:
        # Without strict verification treat raw hits as provisional positives.
        for entry in by_id.values():
            entry["glitchminer_score"] = 0.5

    # Re-decode token_str through the filter for consistency with Magikarp output.
    for entry in by_id.values():
        try:
            entry["token_str_readable"] = tf.decode([entry["token_id"]])
        except Exception:
            entry["token_str_readable"] = entry["token_str"]

    return list(by_id.values())
