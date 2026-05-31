"""``MitigationPipeline`` — orchestrates Bloom + Registry + Track 3.

Pipeline::

    Input Text -> Tokenizer -> Bloom Filter Check -> RocksDB Lookup -> Track 3 -> Model -> Output
                                   (0.1 µs)              (8 µs)        (50-200 µs)

Only Track 3 (Context-Aware Embedding Injection) is applied to flagged
tokens. Embedding patches are staged in ``extras`` so the caller applies
them via ``apply_embedding_patches()`` before ``model.generate()``.

Usage::

    pipeline = MitigationPipeline(
        bloom=BloomGate.load("data/mitigation_system/bloom_filter.bin"),
        registry=TokenRegistry("data/mitigation_system/rocksdb_storage"),
        model=hf_model,
    )
    new_ids, audit, extras = pipeline.mitigate(token_ids)

    if extras["embedding_patches"]:
        embeds = apply_embedding_patches(model, torch.tensor([new_ids]),
                                         extras["embedding_patches"])
        out = model.generate(inputs_embeds=embeds, ...)
    else:
        out = model.generate(input_ids=torch.tensor([new_ids]), ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..layer1_detection.bloom_filter import BloomGate
from ..layer3_registry.registry import TokenRegistry
from .base import (
    MitigationAction,
    MitigationContext,
    MitigationResult,
    MitigationTrack,
)
from .severity_router import RouterConfig, SeverityRouter
from .track3_embedding_injection import Track3EmbeddingInjection


@dataclass
class PipelineStats:
    total_tokens: int = 0
    bloom_hits: int = 0
    registry_hits: int = 0
    actions: dict[str, int] = None
    escalations: int = 0

    def __post_init__(self):
        if self.actions is None:
            self.actions = {a.value: 0 for a in MitigationAction}

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "bloom_hits": self.bloom_hits,
            "registry_hits": self.registry_hits,
            "actions": dict(self.actions),
            "escalations": self.escalations,
        }


class MitigationPipeline:
    def __init__(
        self,
        bloom: Optional[BloomGate],
        registry: TokenRegistry,
        tokenizer: Any = None,
        model: Any = None,
        router: Optional[SeverityRouter] = None,
        tracks: Optional[dict[str, MitigationTrack]] = None,
        neighbor_window: int = 3,
    ):
        self.bloom = bloom
        self.registry = registry
        self.tokenizer = tokenizer
        self.model = model
        self.router = router or SeverityRouter(RouterConfig())
        self.neighbor_window = neighbor_window
        self.tracks: dict[str, MitigationTrack] = tracks or {
            "track3_embedding_injection": Track3EmbeddingInjection(),
        }
        self.stats = PipelineStats()

    # ================================================================ runtime
    def mitigate(
        self, token_ids: Sequence[int]
    ) -> tuple[list[int], list[dict], dict]:
        """Run the embedding injection track over a prompt.

        Returns ``(new_token_ids, audit_records, extras)`` where:
          * ``new_token_ids``    — id list (unchanged; embedding patches
            are applied at generation time, not here).
          * ``audit_records``    — per-token :class:`MitigationResult` dicts.
          * ``extras``           — staged ``embedding_patches`` to be applied
            via :func:`~hmuttms.layer4_mitigation.track3_embedding_injection.apply_embedding_patches`.
        """
        context = MitigationContext(
            token_ids=list(int(t) for t in token_ids),
            position=0,
            neighbor_window=self.neighbor_window,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        audit: list[dict] = []

        while context.position < len(context.token_ids):
            self.stats.total_tokens += 1
            tid = context.current_id

            # ---- Bloom: fast reject ------------------------------------
            if self.bloom is not None and not self.bloom.contains(tid):
                self.stats.actions[MitigationAction.PASS_THROUGH.value] += 1
                context.position += 1
                continue
            if self.bloom is not None:
                self.stats.bloom_hits += 1

            # ---- Registry lookup --------------------------------------
            view = self.registry.lookup(tid)
            if not any(view.values()):
                self.stats.actions[MitigationAction.PASS_THROUGH.value] += 1
                context.position += 1
                continue
            self.stats.registry_hits += 1

            # ---- Track 3: embedding injection -------------------------
            result = self._dispatch(context, view, audit)
            audit.append(result.to_dict())
            self.stats.actions[result.action.value] += 1
            context.position += 1

        extras = {
            "embedding_patches": context.extras.get("embedding_patches", []),
        }
        return context.token_ids, audit, extras

    # --------------------------------------------------------------- dispatch
    def _dispatch(
        self,
        context: MitigationContext,
        view: dict,
        audit: list[dict],
    ) -> MitigationResult:
        track = self.tracks.get("track3_embedding_injection")
        if track is not None and track.applies(context, view):
            return track.apply(context, view)

        # Track 3 not applicable (no embedding source available).
        return MitigationResult(
            original_token_id=context.current_id,
            action=MitigationAction.PASS_THROUGH,
            track=None,
            success=False,
            reason="no_embedding_source",
        )
