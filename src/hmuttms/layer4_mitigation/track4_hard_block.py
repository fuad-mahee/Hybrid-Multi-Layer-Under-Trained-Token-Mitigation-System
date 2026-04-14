"""Track 4 — Hard Blocking with Graceful Degradation.

Last-resort track: replace the token with a configurable fallback
(``<unk>`` by default) or reject the entire prompt with an informative
error. Always "succeeds" in the trivial sense — it's the backstop for
the pipeline.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)


class BlockedTokenError(RuntimeError):
    """Raised when ``reject_mode`` is enabled and a token is hard-blocked."""

    def __init__(self, token_id: int, token_str: Optional[str], message: str):
        super().__init__(message)
        self.token_id = token_id
        self.token_str = token_str


class Track4HardBlock(MitigationTrack):
    """Replace the token with a safe fallback or raise :class:`BlockedTokenError`."""

    name = "track4_hard_block"

    def __init__(
        self,
        fallback_token_id: Optional[int] = None,
        fallback_string: str = "",
        reject_mode: bool = False,
    ):
        self.fallback_token_id = fallback_token_id
        self.fallback_string = fallback_string
        self.reject_mode = reject_mode

    # ---------------------------------------------------------------- applies
    def applies(self, context: MitigationContext, registry_view: dict) -> bool:
        # Always a valid backstop.
        return True

    # ----------------------------------------------------------- fallback id
    def _resolve_fallback_id(self, context: MitigationContext) -> Optional[int]:
        if self.fallback_token_id is not None:
            return int(self.fallback_token_id)
        tok = context.tokenizer
        if tok is None:
            return None
        for attr in ("unk_token_id", "pad_token_id", "eos_token_id"):
            v = getattr(tok, attr, None)
            if v is not None:
                return int(v)
        if self.fallback_string:
            try:
                ids = tok.encode(self.fallback_string, add_special_tokens=False)
                if ids:
                    return int(ids[0])
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------ apply
    def apply(
        self, context: MitigationContext, registry_view: dict
    ) -> MitigationResult:
        blacklist = registry_view.get("blacklist_tokens") or {}
        token_str = blacklist.get("token_string") or registry_view.get("token_string")
        tid = context.current_id

        if self.reject_mode:
            raise BlockedTokenError(
                tid,
                token_str,
                f"Hard-blocked under-trained token id={tid!r} ({token_str!r}). "
                f"Provide an alternative phrasing.",
            )

        fallback_id = self._resolve_fallback_id(context)
        if fallback_id is None:
            # No usable fallback — drop the slot entirely.
            del context.token_ids[context.position]
            context.position -= 1
            return MitigationResult(
                original_token_id=tid,
                action=MitigationAction.BLOCK,
                track=self.name,
                success=True,
                reason="dropped_no_fallback",
                details={"token_string": token_str, "mode": "drop"},
            )

        context.token_ids[context.position] = int(fallback_id)
        return MitigationResult(
            original_token_id=tid,
            action=MitigationAction.BLOCK,
            track=self.name,
            replacement_token_ids=[int(fallback_id)],
            success=True,
            details={
                "token_string": token_str,
                "fallback_token_id": int(fallback_id),
                "mode": "replace",
            },
        )
