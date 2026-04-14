"""Track 1 — Token Substitution (Novel).

For tokens in ``substitution_map`` (Model B handled them, Model A didn't)
we splice the pre-computed ``substitute_token_ids`` from the Layer-3
registry into the prompt in place of the original id. When only string
substitutes are stored, we re-tokenize them on the fly.

Failure modes:
    * Registry has no ``substitute_tokens``/``substitute_token_ids``.
    * Re-tokenization needs a tokenizer but none was provided.

On failure we return ``success=False`` so the pipeline escalates to
Track 2.
"""
from __future__ import annotations

from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)


class Track1Substitution(MitigationTrack):
    name = "track1_substitution"

    def applies(self, context: MitigationContext, registry_view: dict) -> bool:
        sub = registry_view.get("substitution_map")
        if not sub:
            return False
        rs = sub.get("replacement_strategy") or {}
        return bool(rs.get("substitute_token_ids") or rs.get("substitute_tokens"))

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _retokenize(tokenizer, pieces: list[str]) -> list[int]:
        ids: list[int] = []
        for piece in pieces:
            encoded = tokenizer.encode(piece, add_special_tokens=False)
            ids.extend(int(x) for x in encoded)
        return ids

    # --------------------------------------------------------------- apply
    def apply(
        self, context: MitigationContext, registry_view: dict
    ) -> MitigationResult:
        sub = registry_view.get("substitution_map") or {}
        rs = sub.get("replacement_strategy") or {}

        replacement_ids: list[int] = list(rs.get("substitute_token_ids") or [])
        substitute_tokens: list[str] = list(rs.get("substitute_tokens") or [])

        if not replacement_ids and substitute_tokens:
            if context.tokenizer is None:
                return MitigationResult(
                    original_token_id=context.current_id,
                    action=MitigationAction.SUBSTITUTE,
                    track=self.name,
                    success=False,
                    reason="string_substitutes_but_no_tokenizer",
                )
            try:
                replacement_ids = self._retokenize(
                    context.tokenizer, substitute_tokens
                )
            except Exception as e:  # pragma: no cover
                return MitigationResult(
                    original_token_id=context.current_id,
                    action=MitigationAction.SUBSTITUTE,
                    track=self.name,
                    success=False,
                    reason=f"retokenize_failed:{type(e).__name__}:{e}",
                )

        if not replacement_ids:
            return MitigationResult(
                original_token_id=context.current_id,
                action=MitigationAction.SUBSTITUTE,
                track=self.name,
                success=False,
                reason="empty_substitution",
            )

        # Splice into the prompt *in place*.
        tid = context.current_id
        context.token_ids[context.position : context.position + 1] = replacement_ids
        # Advance position cursor so the pipeline doesn't re-hit the same slot.
        context.position += len(replacement_ids) - 1

        return MitigationResult(
            original_token_id=tid,
            action=MitigationAction.SUBSTITUTE,
            track=self.name,
            replacement_token_ids=replacement_ids,
            success=True,
            details={
                "method": rs.get("method", "semantic_decomposition"),
                "substitute_tokens": substitute_tokens,
            },
        )
