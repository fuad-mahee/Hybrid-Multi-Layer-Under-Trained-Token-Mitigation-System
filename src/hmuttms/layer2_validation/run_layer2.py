"""Layer-2 CLI — cross-validate Layer-1 candidates against one or more Model-B backends.

Reads aggregated Layer-1 candidates JSONL (default
``data/layer1_candidates.jsonl``) and writes a per-token classification
JSONL (default ``data/layer2_classified.jsonl``).

Output schema (one line per candidate processed):

    {
      "token_id": 12345,
      "token_str": "solidgoldmagikarp",
      "model_a_score": 0.12,
      "model_b_score": 0.38,              # weighted mean across Model-B backends
      "classification": "BLACKLIST",      # BLACKLIST | SUBSTITUTABLE | FALSE_POSITIVE | REVIEW
      "verdict": "undertrained",          # majority verdict from ensemble
      "backends": ["moonshotai/Kimi-K2-Instruct", "meta-llama/Llama-3.1-70B"],
      "probes": [ ...per-backend details... ],
      "layer1_confidence": 0.78
    }

Backends are specified with one or more repeatable flags::

    --cached-file  data/kimi_k2_magikarp_results.json
    --cached-file  data/<other>_results.json
    --hf-model     meta-llama/Llama-3.1-70B-Instruct
    --hf-model     google/gemma-2-27b-it

At least one backend must be provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from .decision_matrix import (
    Classification,
    DecisionConfig,
    classify,
    model_a_score_from_layer1,
)
from .ensemble import ensemble_vote
from .model_b_tester import (
    BaseModelBTester,
    CachedFileTester,
    LocalHFTester,
    TokenProbe,
)


# -------------------------------------------------------------------- helpers
def _iter_candidates(path: Path, min_confidence: float) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if float(entry.get("confidence", 0.0)) < min_confidence:
                continue
            yield entry


def _build_backends(args: argparse.Namespace) -> list[BaseModelBTester]:
    backends: list[BaseModelBTester] = []
    for path in args.cached_file or []:
        backends.append(CachedFileTester(path))
    for model_id in args.hf_model or []:
        backends.append(
            LocalHFTester(
                model_id=model_id,
                device=args.hf_device,
                trust_remote_code=args.hf_trust_remote_code,
                dtype=args.hf_dtype,
                max_new_tokens=args.hf_max_new_tokens,
            )
        )
    return backends


def _parse_weights(pairs: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--backend-weight expects NAME=FLOAT, got {p!r}")
        name, val = p.split("=", 1)
        out[name.strip()] = float(val)
    return out


# ------------------------------------------------------------------------ CLI
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMUTTMS Layer 2 — cross-validation")
    p.add_argument("--input", default="data/layer1_candidates.jsonl")
    p.add_argument("--output", default="data/layer2_classified.jsonl")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip Layer-1 candidates below this confidence",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N candidates (debug / smoke-test)",
    )

    # Backends
    p.add_argument(
        "--cached-file",
        action="append",
        help="Path to a Magikarp-style verification JSON (repeatable)",
    )
    p.add_argument(
        "--hf-model",
        action="append",
        help="HuggingFace causal-LM id for a local Model-B (repeatable)",
    )
    p.add_argument("--hf-device", default="auto")
    p.add_argument("--hf-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hf-trust-remote-code", action="store_true")
    p.add_argument("--hf-max-new-tokens", type=int, default=24)

    # Ensemble weights
    p.add_argument(
        "--backend-weight",
        action="append",
        help="NAME=FLOAT ensemble weight (e.g. moonshotai/Kimi-K2-Instruct=1.5)",
    )

    # Decision thresholds (PDF defaults)
    p.add_argument("--a-fail-below", type=float, default=0.30)
    p.add_argument("--b-fail-below", type=float, default=0.50)
    p.add_argument("--b-pass-above", type=float, default=0.70)

    p.add_argument("--progress", action="store_true", help="Show tqdm progress")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[layer2] input not found: {in_path}", file=sys.stderr)
        return 2

    backends = _build_backends(args)
    if not backends:
        print(
            "[layer2] no Model-B backends supplied "
            "(use --cached-file and/or --hf-model)",
            file=sys.stderr,
        )
        return 2

    weights = _parse_weights(args.backend_weight)
    cfg = DecisionConfig(
        a_fail_below=args.a_fail_below,
        b_fail_below=args.b_fail_below,
        b_pass_above=args.b_pass_above,
    )

    candidates = list(_iter_candidates(in_path, args.min_confidence))
    if args.limit:
        candidates = candidates[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {c.value: 0 for c in Classification}
    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for cand in candidates:
            probes: list[TokenProbe] = [
                b.probe(int(cand["token_id"]), cand.get("token_str"))
                for b in backends
            ]
            ens = ensemble_vote(probes, weights=weights)
            a_score = model_a_score_from_layer1(cand)
            cls = classify(a_score, ens.score, cfg)
            counts[cls.value] += 1

            # Collect the first non-None embedding from any backend probe.
            model_b_embedding: Optional[list[float]] = None
            for p in probes:
                if p.embedding is not None:
                    model_b_embedding = p.embedding
                    break

            record = {
                "token_id": int(cand["token_id"]),
                "token_str": cand.get("token_str") or ens.token_str,
                "model_a_score": a_score,
                "model_b_score": ens.score,
                "classification": cls.value,
                "verdict": ens.verdict,
                "backends": ens.backends,
                "probes": ens.per_backend,
                "layer1_confidence": float(cand.get("confidence", 0.0)),
                "layer1_sources": cand.get("sources", []),
                "model_b_embedding": model_b_embedding,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "processed": written,
        "backends": [b.name for b in backends],
        "thresholds": {
            "a_fail_below": cfg.a_fail_below,
            "b_fail_below": cfg.b_fail_below,
            "b_pass_above": cfg.b_pass_above,
        },
        "classification_counts": counts,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
