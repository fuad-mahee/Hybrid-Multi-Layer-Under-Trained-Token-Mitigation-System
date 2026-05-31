"""MitigatingTokenizer — drop-in HuggingFace tokenizer wrapper.

Intercepts every ``encode`` / ``__call__`` at the *tokenizer level* so
the mitigation pipeline fires before the model ever receives a token id.

This is the Bloom-filter deployment point described in the HMUTTMS PDF::

    Input Text -> Tokenizer -> Bloom Filter Check -> RocksDB Lookup
                                   (0.1 µs)              (8 µs)
                     ↓                 ↓
                  Token IDs    99 % pass through

Usage::

    from transformers import AutoTokenizer
    from hmuttms.layer4_mitigation import MitigationPipeline
    from hmuttms.inference import MitigatingTokenizer

    base_tok = AutoTokenizer.from_pretrained("gpt2")
    tok = MitigatingTokenizer(base_tok, pipeline=pipeline)

    # Standard HF calls now run mitigation automatically.
    ids   = tok.encode("Hello <glitch-token> world")
    batch = tok(["prompt1", "prompt2"], return_tensors="pt")

The wrapper is *transparent*: every attribute/method not explicitly
overridden is proxied to the underlying tokenizer, so it can be passed
to ``AutoModelForCausalLM.generate()`` without changes.

Audit records are accumulated in ``tok.audit_log`` and can be forwarded
to the Layer-5 monitor at generation time.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..layer4_mitigation.pipeline import MitigationPipeline


class MitigatingTokenizer:
    """Wraps a HuggingFace tokenizer; applies mitigation on every encode."""

    def __init__(
        self,
        tokenizer,
        pipeline: MitigationPipeline,
        return_audit: bool = False,
    ):
        object.__setattr__(self, "_tok", tokenizer)
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_return_audit", return_audit)
        object.__setattr__(self, "audit_log", [])

    # ---------------------------------------------------------------- proxy
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_tok"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        tok = object.__getattribute__(self, "_tok")
        setattr(tok, name, value)

    # --------------------------------------------------------- core: encode
    def encode(
        self,
        text: Union[str, list],
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[int]:
        tok = object.__getattribute__(self, "_tok")
        pipe = object.__getattribute__(self, "_pipeline")

        raw: list[int] = tok.encode(
            text, add_special_tokens=add_special_tokens, **kwargs
        )
        new_ids, audit, extras = pipe.mitigate(raw)
        self._append_audit(audit, extras)
        return new_ids

    # --------------------------------------------------------- __call__
    def __call__(
        self,
        text: Union[str, list, None] = None,
        *args,
        **kwargs,
    ) -> Any:
        tok = object.__getattribute__(self, "_tok")
        pipe = object.__getattribute__(self, "_pipeline")

        result = tok(text, *args, **kwargs)

        # Patch ``input_ids`` tensor / list in the returned BatchEncoding.
        if hasattr(result, "input_ids"):
            import torch

            ids_raw = result.input_ids
            if isinstance(ids_raw, torch.Tensor):
                batched = ids_raw.tolist()
                new_batched, all_audit, all_extras = [], [], {}
                for seq in batched:
                    new_ids, audit, extras = pipe.mitigate(seq)
                    new_batched.append(new_ids)
                    all_audit.extend(audit)
                    for k, v in extras.items():
                        all_extras.setdefault(k, []).extend(v)
                # Pad to equal length.
                max_len = max(len(s) for s in new_batched) if new_batched else 0
                pad_id = tok.pad_token_id or tok.eos_token_id or 0
                padded = [
                    s + [pad_id] * (max_len - len(s)) for s in new_batched
                ]
                result["input_ids"] = torch.tensor(padded, dtype=ids_raw.dtype)
                if "attention_mask" in result:
                    masks = [
                        [1] * len(s) + [0] * (max_len - len(s))
                        for s in new_batched
                    ]
                    result["attention_mask"] = torch.tensor(
                        masks, dtype=result["attention_mask"].dtype
                    )
                self._append_audit(all_audit, all_extras)
            else:
                # Plain list (non-tensor mode).
                if isinstance(ids_raw[0], int):
                    new_ids, audit, extras = pipe.mitigate(ids_raw)
                    result["input_ids"] = new_ids
                    self._append_audit(audit, extras)
        return result

    # ---------------------------------------------------- audit helpers
    def _append_audit(self, audit: list[dict], extras: dict) -> None:
        log = object.__getattribute__(self, "audit_log")
        log.append({"audit": audit, "extras": extras})

    def pop_audit(self) -> list[dict]:
        """Return and clear the accumulated audit records."""
        log = object.__getattribute__(self, "audit_log")
        out = list(log)
        log.clear()
        return out

    def last_extras(self) -> dict:
        """Return the most-recent activation/embedding patches."""
        log = object.__getattribute__(self, "audit_log")
        if not log:
            return {"embedding_patches": []}
        return log[-1].get("extras", {})
