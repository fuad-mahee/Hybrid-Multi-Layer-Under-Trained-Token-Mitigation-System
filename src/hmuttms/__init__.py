"""HMUTTMS — Hybrid Multi-Layer Under-Trained Token Mitigation System.

Layer map
---------
Layer 1  :mod:`hmuttms.layer1_detection`   Magikarp + GlitchMiner detection + Bloom filter
Layer 2  :mod:`hmuttms.layer2_validation`  Two-model cross-validation + ensemble
Layer 3  :mod:`hmuttms.layer3_registry`    Persistent RocksDB token registry
Layer 4  :mod:`hmuttms.layer4_mitigation`  Four-track runtime mitigation pipeline
Layer 5  :mod:`hmuttms.layer5_feedback`    Continuous learning + feedback loop
Layer 6  :mod:`hmuttms.inference`          HF tokenizer wrapper + InferenceSession glue

Full pipeline CLI
-----------------
Offline (one-time setup)::

    python -m hmuttms.run_full_pipeline offline \\
        --model-id  meta-llama/Llama-3.2-3B-Instruct \\
        --model-b-cache data/kimi_k2_magikarp_results.json \\
        --device cuda

Runtime (interactive demo)::

    python -m hmuttms.run_full_pipeline runtime \\
        --model-id  meta-llama/Llama-3.2-3B-Instruct \\
        --device cuda
"""
from .inference import InferenceSession, MitigatingTokenizer
from .layer1_detection import BloomGate
from .layer3_registry import TokenRegistry
from .layer4_mitigation import MitigationPipeline

__version__ = "0.1.0"
__all__ = [
    "BloomGate",
    "InferenceSession",
    "MitigatingTokenizer",
    "MitigationPipeline",
    "TokenRegistry",
]
