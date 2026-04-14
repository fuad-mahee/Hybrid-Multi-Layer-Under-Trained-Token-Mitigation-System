"""``MitigationPipeline`` — orchestrates Bloom + Registry + Tracks.

The pipeline is the production entry-point for Layer 4::

    pipeline = MitigationPipeline(
        bloom=BloomGate.load("data/mitigation_system/bloom_filter.bin"),
        registry=TokenRegistry("data/mitigation_system/rocksdb_storage"),
        tokenizer=hf_tok,
        model=hf_model,          # optional (needed for Tracks 2/3)
    )
    new_ids, audit = pipeline.mitigate(token_ids)

``new_ids`` is the (possibly rewritten) id list to feed the model.
``audit`` is a list of per-action :class:`MitigationResult` dicts that
Layer 5's feedback loop uses to update success counters.

Activation & embedding patches aren't applied to the ids themselves;
they're staged in ``audit_extras`` so the caller can install them around
``model.generate(...)``::

    with install_hooks_for(model, audit_extras["activation_patches"]):
        if audit_extras["embedding_patches"]:
            embeds = apply_embedding_patches(model, torch.tensor([new_ids]),
                                             audit_extras["embedding_patches"])
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
from .track1_substitution import Track1Substitution
from .track2_activation_repair import Track2ActivationRepair
from .track3_embedding_injection import Track3EmbeddingInjection
from .track4_hard_block import Track4HardBlock


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
            "track1_substitution": Track1Substitution(),
            "track2_activation_repair": Track2ActivationRepair(),
            "track3_embedding_injection": Track3EmbeddingInjection(),
            "track4_hard_block": Track4HardBlock(),
        }
        self.stats = PipelineStats()

    # ================================================================ runtime
    def mitigate(
        self, token_ids: Sequence[int]
    ) -> tuple[list[int], list[dict], dict]:
        """Run the four-track dispatcher over a prompt.

        Returns ``(new_token_ids, audit_records, extras)`` where:
          * ``new_token_ids``  — id list after substitution/block rewrites.
          * ``audit_records``  — per-action dicts suitable for Layer 5.
          * ``extras``         — staged ``activation_patches`` and
            ``embedding_patches`` to be applied at generation time.
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
                # Bloom false-positive; no registry row.
                self.stats.actions[MitigationAction.PASS_THROUGH.value] += 1
                context.position += 1
                continue
            self.stats.registry_hits += 1

            # ---- Severity routing + track dispatch ---------------------
            result = self._dispatch(context, view, audit)
            audit.append(result.to_dict())
            self.stats.actions[result.action.value] += 1
            context.position += 1

        extras = {
            "activation_patches": context.extras.get("activation_patches", []),
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
        order = self.router.order(view)
        last_result: Optional[MitigationResult] = None

        for name in order:
            track = self.tracks.get(name)
            if track is None:
                continue
            if not track.applies(context, view):
                continue
            result = track.apply(context, view)
            if result.success:
                return result

            # Record the failed attempt as an ESCALATED note.
            self.stats.escalations += 1
            audit.append(
                MitigationResult(
                    original_token_id=context.current_id,
                    action=MitigationAction.ESCALATED,
                    track=name,
                    success=False,
                    reason=result.reason,
                ).to_dict()
            )
            last_result = result

        # If even Track 4's ``applies`` returned False we still synthesize
        # a result so callers never see a missing row.
        return last_result or MitigationResult(
            original_token_id=context.current_id,
            action=MitigationAction.PASS_THROUGH,
            track=None,
            success=False,
            reason="no_track_applicable",
        )
