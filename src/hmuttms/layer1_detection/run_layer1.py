"""Layer 1 CLI — run Magikarp + GlitchMiner on Model A and dump aggregated candidates.

Example (on research machine):

    python -m hmuttms.layer1_detection.run_layer1 \
        --model-id meta-llama/Llama-3.2-3B-Instruct \
        --output data/layer1_candidates.jsonl \
        --device cuda

Magikarp and GlitchMiner are each optional; pass --skip-magikarp or
--skip-glitchminer to run only one detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregator import aggregate_candidates, write_jsonl
from .glitchminer_runner import run_glitchminer
from .magikarp_runner import run_magikarp


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMUTTMS Layer 1 — Magikarp + GlitchMiner detection")
    p.add_argument("--model-id", required=True, help="HuggingFace model id / local path for Model A")
    p.add_argument("--output", default="data/layer1_candidates.jsonl", help="Aggregated output path")
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda', 'cuda:0', or 'auto'")
    p.add_argument("--trust-remote-code", action="store_true")

    # Magikarp knobs
    p.add_argument("--skip-magikarp", action="store_true")
    p.add_argument("--magikarp-threshold", type=float, default=None)
    p.add_argument("--magikarp-threshold-ratio", type=float, default=None)
    p.add_argument("--magikarp-indicator-ix", type=int, default=0)

    # GlitchMiner knobs
    p.add_argument("--skip-glitchminer", action="store_true")
    p.add_argument("--gm-iterations", type=int, default=50)
    p.add_argument("--gm-batch-size", type=int, default=8)
    p.add_argument("--gm-k", type=int, default=32)
    p.add_argument("--gm-quant", default="bfloat16", choices=["bfloat16", "float16", "int4"])
    p.add_argument("--gm-no-strict-verify", action="store_true")

    # Aggregation
    p.add_argument("--weight-magikarp", type=float, default=0.45)
    p.add_argument("--weight-glitchminer", type=float, default=0.55)
    p.add_argument(
        "--raw-dir",
        default=None,
        help="Optional dir to also dump per-detector raw results for debugging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    magikarp_results: list[dict] = []
    glitchminer_results: list[dict] = []

    if not args.skip_magikarp:
        print(f"[layer1] Running Magikarp on {args.model_id} ...", flush=True)
        mk_kwargs = dict(
            model_id=args.model_id,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            indicator_ix=args.magikarp_indicator_ix,
        )
        if args.magikarp_threshold is not None:
            mk_kwargs["threshold"] = args.magikarp_threshold
        if args.magikarp_threshold_ratio is not None:
            mk_kwargs["threshold_ratio"] = args.magikarp_threshold_ratio
        magikarp_results = run_magikarp(**mk_kwargs)
        print(f"[layer1] Magikarp produced {len(magikarp_results)} candidates", flush=True)

    if not args.skip_glitchminer:
        print(f"[layer1] Running GlitchMiner on {args.model_id} ...", flush=True)
        # GlitchMiner's 'auto' device mapping needs a GPU; allow --device cuda or auto.
        gm_device = args.device if args.device != "cpu" else "auto"
        glitchminer_results = run_glitchminer(
            model_path=args.model_id,
            num_iterations=args.gm_iterations,
            batch_size=args.gm_batch_size,
            k=args.gm_k,
            device=gm_device,
            quant_type=args.gm_quant,
            strict_verify=not args.gm_no_strict_verify,
        )
        print(f"[layer1] GlitchMiner produced {len(glitchminer_results)} candidates", flush=True)

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "magikarp_raw.json").write_text(
            json.dumps(magikarp_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (raw_dir / "glitchminer_raw.json").write_text(
            json.dumps(glitchminer_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    merged = aggregate_candidates(
        magikarp_results,
        glitchminer_results,
        weights={
            "magikarp": args.weight_magikarp,
            "glitchminer": args.weight_glitchminer,
        },
    )
    out_path = write_jsonl(args.output, merged)
    print(f"[layer1] Wrote {len(merged)} aggregated candidates to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
