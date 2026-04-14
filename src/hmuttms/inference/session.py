"""InferenceSession — single object that runs the full HMUTTMS stack.

Encapsulates:
  * The target HuggingFace model + tokenizer (Model A)
  * A fully-configured :class:`~hmuttms.layer4_mitigation.MitigationPipeline`
  * An optional :class:`~hmuttms.layer5_feedback.RuntimeMonitor`

The public API is intentionally minimal::

    session = InferenceSession.from_paths(
        model_id="gpt2",
        bloom_path="data/mitigation_system/bloom_filter.bin",
        registry_root="data/mitigation_system/rocksdb_storage",
        device="cuda",
    )
    output = session.generate("Tell me about solidgoldmagikarp")

Internally, ``generate`` runs:

    1. Base tokenizer → raw token ids
    2. MitigationPipeline.mitigate() → rewritten ids + staged patches
    3. If Track-2 patches exist → install forward hooks around generate
    4. If Track-3 patches exist → switch to inputs_embeds mode
    5. model.generate() → output ids
    6. tokenizer.decode() → text
    7. RuntimeMonitor records the event

``session.stats`` returns cumulative :class:`~hmuttms.layer4_mitigation.PipelineStats`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..layer1_detection.bloom_filter import BloomGate
from ..layer3_registry.registry import TokenRegistry
from ..layer4_mitigation.pipeline import MitigationPipeline, PipelineStats
from ..layer4_mitigation.severity_router import RouterConfig, SeverityRouter
from ..layer4_mitigation.track2_activation_repair import install_hooks_for
from ..layer4_mitigation.track3_embedding_injection import apply_embedding_patches
from ..layer5_feedback.monitor import RuntimeMonitor
from ..layer5_feedback.quality import QualityChecker
from .mitigating_tokenizer import MitigatingTokenizer


@dataclass
class GenerateResult:
    text: str
    new_token_ids: list[int]
    pipeline_ms: float
    generation_ms: float
    audit: list[dict]
    quality: Optional[dict] = None
    stats: Optional[dict] = None


class InferenceSession:
    """Holds model + tokenizer + full HMUTTMS stack; exposes ``generate()``."""

    def __init__(
        self,
        model,
        tokenizer,
        pipeline: MitigationPipeline,
        monitor: Optional[RuntimeMonitor] = None,
        max_new_tokens: int = 256,
        device: str = "cpu",
    ):
        self._model = model
        self._base_tok = tokenizer
        self._pipeline = pipeline
        self._monitor = monitor
        self.max_new_tokens = max_new_tokens
        self.device = device

        # Expose the mitigating tokenizer for external callers that pass
        # ``tokenizer=session.tokenizer`` to HF pipelines.
        self.tokenizer = MitigatingTokenizer(tokenizer, pipeline=pipeline)

    # ------------------------------------------------------------ factory
    @classmethod
    def from_paths(
        cls,
        model_id: str,
        bloom_path: str | Path,
        registry_root: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        backend_kind: str = "auto",
        router_config: Optional[RouterConfig] = None,
        monitor_log: Optional[str | Path] = None,
        quality_checker: Optional[QualityChecker] = None,
        max_new_tokens: int = 256,
    ) -> "InferenceSession":
        """Load everything from disk and return a ready session."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        bloom: Optional[BloomGate] = None
        bp = Path(bloom_path)
        if bp.exists():
            bloom = BloomGate.load(bp)

        registry = TokenRegistry(registry_root, backend_kind=backend_kind)
        pipeline = MitigationPipeline(
            bloom=bloom,
            registry=registry,
            tokenizer=tokenizer,
            model=model,
            router=SeverityRouter(router_config or RouterConfig()),
        )

        monitor: Optional[RuntimeMonitor] = None
        if monitor_log is not None:
            monitor = RuntimeMonitor(
                pipeline,
                event_log_path=monitor_log,
                quality_checker=quality_checker,
            )

        return cls(
            model=model,
            tokenizer=tokenizer,
            pipeline=pipeline,
            monitor=monitor,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    # ---------------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        extra_generate_kwargs: Optional[dict] = None,
    ) -> GenerateResult:
        """Run the full HMUTTMS pipeline + model generation.

        1. Encode → mitigate → get rewritten ids + staged patches
        2. Install Track-2 hooks (if any)
        3. Switch to ``inputs_embeds`` mode for Track-3 patches (if any)
        4. model.generate()
        5. Record telemetry via RuntimeMonitor
        """
        import torch

        max_new = max_new_tokens or self.max_new_tokens
        extra = extra_generate_kwargs or {}

        # ---- 1. Tokenise + mitigate -----------------------------------------
        raw_ids: list[int] = self._base_tok.encode(
            prompt, add_special_tokens=True
        )

        t0 = time.perf_counter()
        new_ids, audit, extras = self._pipeline.mitigate(raw_ids)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        ids_tensor = torch.tensor([new_ids], device=self._model.device)
        attn_mask = torch.ones_like(ids_tensor)

        activation_patches = extras.get("activation_patches", [])
        embedding_patches = extras.get("embedding_patches", [])

        # ---- 2-4. Model generation (with optional patches) ------------------
        t1 = time.perf_counter()
        with install_hooks_for(self._model, activation_patches):
            if embedding_patches:
                inputs_embeds = apply_embedding_patches(
                    self._model, ids_tensor, embedding_patches
                )
                out_ids = self._model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new,
                    pad_token_id=(
                        self._base_tok.pad_token_id
                        or self._base_tok.eos_token_id
                    ),
                    **extra,
                )
            else:
                out_ids = self._model.generate(
                    input_ids=ids_tensor,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new,
                    pad_token_id=(
                        self._base_tok.pad_token_id
                        or self._base_tok.eos_token_id
                    ),
                    **extra,
                )
        generation_ms = (time.perf_counter() - t1) * 1000.0

        # Decode only the *new* tokens (after the prompt).
        prompt_len = ids_tensor.shape[-1]
        new_out_ids = out_ids[0][prompt_len:].tolist()
        text = self._base_tok.decode(new_out_ids, skip_special_tokens=True)

        # ---- 5. Monitor / telemetry -----------------------------------------
        quality: Optional[dict] = None
        if self._monitor is not None:
            result = self._monitor.run(
                raw_ids,
                generated_text=text,
                generation_ms=generation_ms,
            )
            quality = result.quality

        return GenerateResult(
            text=text,
            new_token_ids=new_ids,
            pipeline_ms=pipeline_ms,
            generation_ms=generation_ms,
            audit=audit,
            quality=quality,
            stats=self._pipeline.stats.to_dict(),
        )

    # -------------------------------------------------------------- misc
    @property
    def stats(self) -> PipelineStats:
        return self._pipeline.stats

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.close()

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
