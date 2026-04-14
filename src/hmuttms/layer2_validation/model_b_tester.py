"""Model-B probing backends for Layer-2 cross-validation.

A *tester* takes a ``token_id`` (plus optional token string) and returns
a :class:`TokenProbe` carrying a normalised score in ``[0, 1]`` where

    * ``1.0`` == Model B handles the token perfectly (not under-trained)
    * ``0.0`` == Model B is as broken on the token as Model A (under-trained)

Three backends are provided:

``LocalHFTester``
    Loads a HuggingFace causal-LM locally and runs the same echo-style
    prompts used by Magikarp verification (input/echo/complete). Score is
    derived from how many tests the model passes.

``CachedFileTester``
    Reuses *already-collected* per-token results such as
    ``data/kimi_k2_magikarp_results.json``. Cheap and deterministic — no
    model needed on the local dev machine.

``BaseModelBTester``
    Abstract base. Subclass it to plug in API-backed superior models
    (GPT-4, Claude 3.5, Gemma 2, …) for the optional multi-model ensemble.

All backends expose the same ``probe(token_id, token_str=None)`` and
``probe_many(...)`` API so the CLI can swap them freely.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------- types
@dataclass
class TokenProbe:
    """Per-token result from a Model-B probe."""

    token_id: int
    token_str: Optional[str]
    score: float                        # 0.0 = undertrained, 1.0 = handled well
    verdict: str                        # "undertrained" | "handled" | "inconclusive"
    backend: str                        # identifier for which Model-B produced this
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------- base API
class BaseModelBTester(ABC):
    """Abstract Model-B probe backend."""

    name: str = "base"

    @abstractmethod
    def probe(self, token_id: int, token_str: Optional[str] = None) -> TokenProbe:
        """Return a :class:`TokenProbe` for a single token."""

    def probe_many(
        self,
        candidates: Iterable[dict],
        progress: bool = False,
    ) -> list[TokenProbe]:
        """Probe an iterable of Layer-1 candidate dicts.

        Each dict must contain ``token_id`` (int); ``token_str`` is optional.
        """
        items = list(candidates)
        iterator = items
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(items, desc=f"[layer2:{self.name}]")
            except ImportError:  # pragma: no cover - tqdm is optional
                iterator = items
        return [self.probe(int(c["token_id"]), c.get("token_str")) for c in iterator]


# --------------------------------------------------------- cached-file backend
_KIMI_SCORE_MAP = {
    # classification -> Model-B score (how well Model B handles the token)
    "strong_rejected": 0.95,      # Model B passes all probes  -> not undertrained
    "weak_rejected": 0.80,
    "borderline": 0.50,
    "weak_verified": 0.25,        # Model B mostly fails       -> also undertrained
    "strong_verified": 0.10,
}
_KIMI_VERDICT_MAP = {
    "strong_rejected": "handled",
    "weak_rejected": "handled",
    "borderline": "inconclusive",
    "weak_verified": "undertrained",
    "strong_verified": "undertrained",
}


class CachedFileTester(BaseModelBTester):
    """Reuse pre-collected Model-B verification results.

    The input file format is the one produced by the Magikarp
    verification pipeline (see ``data/kimi_k2_magikarp_results.json``):

        {
          "metadata": {"model_id": "...", ...},
          "results": [
            {"token_id": 7717, "token_str": "/**", "classification": "...",
             "metrics": {...}, "is_undertrained": false},
            ...
          ]
        }

    If a queried token is not present in the cache, the probe is returned
    with ``verdict="inconclusive"`` and ``score=None``-replacement (0.5).
    """

    def __init__(self, path: str | Path, name: Optional[str] = None):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        self._meta: dict = blob.get("metadata", {})
        self._by_id: dict[int, dict] = {
            int(r["token_id"]): r for r in blob.get("results", [])
        }
        self.name = name or self._meta.get("model_id") or f"cache:{path.name}"

    # ------------------------------------------------------------------ probe
    def probe(self, token_id: int, token_str: Optional[str] = None) -> TokenProbe:
        row = self._by_id.get(int(token_id))
        if row is None:
            return TokenProbe(
                token_id=int(token_id),
                token_str=token_str,
                score=0.5,
                verdict="inconclusive",
                backend=self.name,
                details={"reason": "not_in_cache"},
            )
        cls = row.get("classification", "borderline")
        score = _KIMI_SCORE_MAP.get(cls, 0.5)
        verdict = _KIMI_VERDICT_MAP.get(cls, "inconclusive")
        # Prefer in_output_rate as a finer-grained signal if available.
        metrics = row.get("metrics") or {}
        if "in_output_rate" in metrics and cls != "error":
            # Blend: 60% classification bucket, 40% observed in_output_rate.
            score = 0.6 * score + 0.4 * float(metrics["in_output_rate"])
        return TokenProbe(
            token_id=int(token_id),
            token_str=token_str or row.get("token_str"),
            score=float(score),
            verdict=verdict,
            backend=self.name,
            details={
                "classification": cls,
                "metrics": metrics,
                "is_undertrained": row.get("is_undertrained"),
            },
        )


# ---------------------------------------------------------- local HF backend
class LocalHFTester(BaseModelBTester):
    """Probe a locally-loaded HuggingFace causal-LM.

    Runs three Magikarp-style prompts per token and derives a score from
    how many pass. Intentionally lightweight so it can run on the
    research PC without bespoke evaluation scripts.
    """

    DEFAULT_TEMPLATES = (
        # (template, kind) — kind controls how we judge the output
        ("Please repeat the following token exactly: {tok}\nAnswer:", "echo"),
        ("Complete: {tok}", "complete"),
        ("What does this token mean: {tok}?\nAnswer:", "in_output"),
    )

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 24,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        name: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.name = name or model_id
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------- lazy load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    # ---------------------------------------------------------------- probe
    def probe(self, token_id: int, token_str: Optional[str] = None) -> TokenProbe:
        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model

        # Recover a printable form of the token if caller did not supply one.
        if token_str is None:
            token_str = tok.decode([int(token_id)], skip_special_tokens=False)
        probe_text = token_str

        passes = 0
        total = 0
        per_test: list[dict] = []
        for template, kind in self.DEFAULT_TEMPLATES:
            total += 1
            prompt = template.format(tok=probe_text)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            import torch

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            ok = probe_text.strip() in text
            if kind == "echo":
                ok = ok and text.strip().startswith(probe_text.strip()[:8])
            passes += int(ok)
            per_test.append({"kind": kind, "ok": bool(ok), "output": text[:120]})

        score = passes / max(total, 1)
        if score >= 0.7:
            verdict = "handled"
        elif score <= 0.3:
            verdict = "undertrained"
        else:
            verdict = "inconclusive"
        return TokenProbe(
            token_id=int(token_id),
            token_str=token_str,
            score=float(score),
            verdict=verdict,
            backend=self.name,
            details={"per_test": per_test, "passes": passes, "total": total},
        )
