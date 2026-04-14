"""Track 3 — Context-Aware Embedding Injection (Novel).

For high-severity blacklisted tokens, replace the input-embedding row for
that position with a weighted average of its neighbours. This keeps the
token slot present (so sequence length and positional encodings are
untouched) while nullifying the under-trained row.

Two sources are supported for the "safe" embedding, in priority order:

1. A ``fallback_embedding`` pre-computed by Model B and stored in the
   substitution / metadata entry.
2. A runtime weighted mean of the input-embedding rows of neighbour
   token ids (distance-weighted Gaussian).

As with Track 2 we **stage** the patch on ``context.extras`` so the
caller applies it during generation via ``apply_embedding_patches(...)``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)


class Track3EmbeddingInjection(MitigationTrack):
    name = "track3_embedding_injection"

    def __init__(self, sigma: float = 1.5):
        self.sigma = float(sigma)

    # ---------------------------------------------------------------- applies
    def applies(self, context: MitigationContext, registry_view: dict) -> bool:
        # Needs either a stored fallback embedding or a model with an embed table.
        has_fallback = bool(self._fallback_embedding(registry_view))
        has_model = context.model is not None
        return has_fallback or has_model

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _fallback_embedding(registry_view: dict) -> Optional[list[float]]:
        sub = registry_view.get("substitution_map") or {}
        rs = sub.get("replacement_strategy") or {}
        emb = rs.get("fallback_embedding")
        if emb:
            return list(emb)
        meta = registry_view.get("metadata") or {}
        if meta.get("embedding"):
            return list(meta["embedding"])
        return None

    def _contextual_embedding(self, context: MitigationContext) -> Optional[list[float]]:
        if context.model is None:
            return None
        try:
            import torch

            embed = context.model.get_input_embeddings()
            neighbors = context.neighbors()
            if not neighbors:
                return None
            ids = torch.tensor(neighbors, device=embed.weight.device, dtype=torch.long)
            with torch.no_grad():
                vecs = embed(ids)
            # Distance-weighted average relative to the flagged position.
            pos = context.position
            lo = max(pos - context.neighbor_window, 0)
            offsets = [i - pos for i in range(lo, lo + len(neighbors) + 1) if i != pos]
            weights = torch.tensor(
                [math.exp(-0.5 * (d / self.sigma) ** 2) for d in offsets],
                dtype=vecs.dtype,
                device=vecs.device,
            )
            weights = weights / weights.sum().clamp_min(1e-9)
            mean = (vecs * weights.unsqueeze(-1)).sum(dim=0)
            return mean.detach().float().cpu().tolist()
        except Exception:
            return None

    # ------------------------------------------------------------------ apply
    def apply(
        self, context: MitigationContext, registry_view: dict
    ) -> MitigationResult:
        emb: Optional[Sequence[float]] = self._fallback_embedding(registry_view)
        source = "fallback_embedding"
        if emb is None:
            emb = self._contextual_embedding(context)
            source = "neighbor_weighted_mean"

        if emb is None:
            return MitigationResult(
                original_token_id=context.current_id,
                action=MitigationAction.INJECT_EMBEDDING,
                track=self.name,
                success=False,
                reason="no_embedding_source",
            )

        context.extras.setdefault("embedding_patches", []).append(
            {
                "position": context.position,
                "token_id": context.current_id,
                "embedding": list(emb),
                "source": source,
            }
        )

        return MitigationResult(
            original_token_id=context.current_id,
            action=MitigationAction.INJECT_EMBEDDING,
            track=self.name,
            replacement_embedding=list(emb),
            success=True,
            details={"source": source, "dim": len(emb)},
        )


# ---------------------------------------------------- runtime application utils
def apply_embedding_patches(model, input_ids, patches: list[dict]):
    """Return ``inputs_embeds`` tensor with patched rows, for ``model.generate``.

    Callers should invoke the model with ``inputs_embeds=`` instead of
    ``input_ids=`` when patches are present::

        embeds = apply_embedding_patches(model, input_ids, patches)
        out = model.generate(inputs_embeds=embeds, attention_mask=mask, ...)
    """
    import torch

    embed = model.get_input_embeddings()
    with torch.no_grad():
        base = embed(input_ids)
    for patch in patches or []:
        pos = patch["position"]
        if pos >= base.shape[-2]:
            continue
        vec = torch.tensor(
            patch["embedding"], dtype=base.dtype, device=base.device
        )
        base[..., pos, :] = vec
    return base
