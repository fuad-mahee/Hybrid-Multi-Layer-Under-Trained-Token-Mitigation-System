"""Layer 6 — Inference Glue.

Ties all five HMUTTMS layers into two entry-points:

* :class:`MitigatingTokenizer` — drop-in wrapper for any HuggingFace
  ``PreTrainedTokenizer``. Override ``encode`` / ``__call__`` so the
  mitigation pipeline fires *before* the model ever sees a token id.

* :class:`InferenceSession` — self-contained session that holds the
  tokenizer, model, pipeline, and monitor together and exposes a single
  ``generate(text) -> str`` interface. Handles activation patching
  (Track 2) and embedding injection (Track 3) transparently.
"""
from .mitigating_tokenizer import MitigatingTokenizer
from .session import InferenceSession

__all__ = ["InferenceSession", "MitigatingTokenizer"]
