"""Layer 1 — Enhanced Detection (Magikarp + GlitchMiner in parallel)."""
from .magikarp_runner import run_magikarp
from .glitchminer_runner import run_glitchminer
from .aggregator import aggregate_candidates

__all__ = ["run_magikarp", "run_glitchminer", "aggregate_candidates"]
