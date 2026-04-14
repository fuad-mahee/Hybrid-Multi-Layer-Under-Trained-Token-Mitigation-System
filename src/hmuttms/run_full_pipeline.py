"""HMUTTMS full-pipeline runner.

Two modes:

offline
    Runs Layers 1 -> 2 -> 3 (detection + cross-validation + registry
    population) as a one-time setup job. Run this on the research PC
    before serving any inference.

runtime
    Interactive demo loop that loads the fully-built registry and runs
    Layer-4 mitigation + Layer-5 monitoring on every prompt entered at
    the terminal.

---------------------------------------------------------------------
Offline usage::

    python -m hmuttms.run_full_pipeline offline \\
        --model-id   meta-llama/Llama-3.2-3B-Instruct \\
        --model-b-cache  data/kimi_k2_magikarp_results.json \\
        --device     cuda

Runtime usage::

    python -m hmuttms.run_full_pipeline runtime \\
        --model-id   meta-llama/Llama-3.2-3B-Instruct \\
        --device     cuda

The offline step must be run at least once before ``runtime`` will work.
---------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Default paths (can be overridden via flags).
_DEFAULT_BLOOM     = "data/mitigation_system/bloom_filter.bin"
_DEFAULT_REG_ROOT  = "data/mitigation_system/rocksdb_storage"
_DEFAULT_L1_OUT    = "data/layer1_candidates.jsonl"
_DEFAULT_L2_OUT    = "data/layer2_classified.jsonl"
_DEFAULT_EVENT_LOG = "data/mitigation_system/feedback/events.jsonl"


# ================================================================== offline
def _run_offline(args: argparse.Namespace) -> int:
    """Run Layers 1 -> 2 -> 3 sequentially (detection, validation, populate)."""
    base = [sys.executable, "-m"]

    # ---- Layer 1: detection ------------------------------------------------
    l1_cmd = (
        base
        + ["hmuttms.layer1_detection.run_layer1"]
        + ["--model-id", args.model_id]
        + ["--output",   args.l1_output]
        + ["--device",   args.device]
    )
    if args.trust_remote_code:
        l1_cmd.append("--trust-remote-code")
    if args.skip_magikarp:
        l1_cmd.append("--skip-magikarp")
    if args.skip_glitchminer:
        l1_cmd.append("--skip-glitchminer")

    print("\n=== Layer 1: Detection ===")
    rc = subprocess.call(l1_cmd)
    if rc != 0:
        return rc

    # ---- Layer 2: cross-validation -----------------------------------------
    l2_cmd = (
        base
        + ["hmuttms.layer2_validation.run_layer2"]
        + ["--input",  args.l1_output]
        + ["--output", args.l2_output]
    )
    for f in args.model_b_cache or []:
        l2_cmd += ["--cached-file", f]
    for m in args.hf_model_b or []:
        l2_cmd += ["--hf-model", m]
    if args.progress:
        l2_cmd.append("--progress")

    print("\n=== Layer 2: Cross-validation ===")
    rc = subprocess.call(l2_cmd)
    if rc != 0:
        return rc

    # ---- Layer 3: populate registry ----------------------------------------
    l3_cmd = (
        base
        + ["hmuttms.layer3_registry.run_populate"]
        + ["--input",        args.l2_output]
        + ["--registry-root", args.registry_root]
        + ["--backend",      args.backend]
        + ["--bloom-output", args.bloom_output]
    )
    if args.include_review:
        l3_cmd.append("--include-review")

    print("\n=== Layer 3: Registry population + Bloom build ===")
    rc = subprocess.call(l3_cmd)
    if rc != 0:
        return rc

    print("\n=== Offline setup complete ===")
    summary = {
        "bloom": args.bloom_output,
        "registry": args.registry_root,
        "l1_candidates": args.l1_output,
        "l2_classified": args.l2_output,
    }
    print(json.dumps(summary, indent=2))
    return 0


# ================================================================= runtime
def _run_runtime(args: argparse.Namespace) -> int:
    """Interactive loop: load session, read prompts, print mitigated output."""
    from hmuttms.inference import InferenceSession
    from hmuttms.layer5_feedback.quality import QualityChecker

    bloom_path = Path(args.bloom_output)
    if not bloom_path.exists():
        print(
            f"[runtime] Bloom filter not found at {bloom_path}.\n"
            "Run 'offline' mode first.",
            file=sys.stderr,
        )
        return 2

    print(f"[runtime] Loading model {args.model_id} …")
    session = InferenceSession.from_paths(
        model_id=args.model_id,
        bloom_path=bloom_path,
        registry_root=args.registry_root,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        backend_kind=args.backend,
        monitor_log=args.event_log,
        quality_checker=QualityChecker(),
        max_new_tokens=args.max_new_tokens,
    )
    print("[runtime] Session ready. Enter prompts (Ctrl-C to quit).\n")

    try:
        with session:
            while True:
                try:
                    prompt = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt:
                    continue
                result = session.generate(prompt)
                print(f"[pipeline {result.pipeline_ms:.2f} ms | "
                      f"gen {result.generation_ms:.1f} ms]\n"
                      f"{result.text}\n")
                if args.verbose:
                    print("audit:", json.dumps(result.audit, indent=2))
                    print("stats:", json.dumps(result.stats, indent=2))
    except Exception as exc:
        print(f"[runtime] error: {exc}", file=sys.stderr)
        return 1
    return 0


# ================================================================= CLI
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HMUTTMS full-pipeline runner (offline setup + runtime demo)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- shared args ---------------------------------------------------------
    def _add_shared(sp):
        sp.add_argument("--model-id", required=True, help="Model A (target) HF id")
        sp.add_argument("--bloom-output",   default=_DEFAULT_BLOOM)
        sp.add_argument("--registry-root",  default=_DEFAULT_REG_ROOT)
        sp.add_argument("--backend",
                        choices=["auto", "rocksdict", "jsondir"], default="auto")
        sp.add_argument("--device",    default="auto")
        sp.add_argument("--dtype",     default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
        sp.add_argument("--trust-remote-code", action="store_true")

    # ---- offline -------------------------------------------------------------
    off = sub.add_parser("offline", help="Run Layers 1-3 offline setup")
    _add_shared(off)
    off.add_argument("--l1-output",     default=_DEFAULT_L1_OUT)
    off.add_argument("--l2-output",     default=_DEFAULT_L2_OUT)
    off.add_argument("--model-b-cache", action="append",
                     help="Cached Model-B JSON (repeatable)")
    off.add_argument("--hf-model-b",    action="append",
                     help="Live HF Model-B id (repeatable)")
    off.add_argument("--skip-magikarp",   action="store_true")
    off.add_argument("--skip-glitchminer", action="store_true")
    off.add_argument("--include-review",  action="store_true")
    off.add_argument("--progress",        action="store_true")
    off.set_defaults(func=_run_offline)

    # ---- runtime -------------------------------------------------------------
    rt = sub.add_parser("runtime", help="Interactive mitigated generation demo")
    _add_shared(rt)
    rt.add_argument("--event-log",       default=_DEFAULT_EVENT_LOG)
    rt.add_argument("--max-new-tokens",  type=int, default=256)
    rt.add_argument("--verbose",         action="store_true",
                    help="Print audit + stats for each turn")
    rt.set_defaults(func=_run_runtime)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
