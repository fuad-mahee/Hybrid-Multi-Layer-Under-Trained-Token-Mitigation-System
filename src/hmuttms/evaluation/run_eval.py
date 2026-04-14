"""HMUTTMS Evaluation CLI.

Four sub-commands:

    detect    — compare Magikarp-only baseline vs. Layer-1 hybrid detection
    mitigate  — measure per-track mitigation success from an event log
    latency   — micro-benchmark Bloom, RocksDB, and full pipeline overhead
    full      — run all three and emit a consolidated report vs. PDF targets

Example usage on the research PC::

    # After the offline pipeline has run:

    python -m hmuttms.evaluation.run_eval detect \\
        --ground-truth  data/layer2_classified.jsonl \\
        --magikarp-csv  data/magikarp_comparison.csv \\
        --layer1-jsonl  data/layer1_candidates.jsonl

    python -m hmuttms.evaluation.run_eval mitigate \\
        --event-log     data/mitigation_system/feedback/events.jsonl

    python -m hmuttms.evaluation.run_eval latency \\
        --layer1-jsonl  data/layer1_candidates.jsonl

    python -m hmuttms.evaluation.run_eval full \\
        --ground-truth  data/layer2_classified.jsonl \\
        --event-log     data/mitigation_system/feedback/events.jsonl \\
        --report-out    data/eval/hmuttms_report.json \\
        --model-a-id    meta-llama/Llama-3.2-3B-Instruct \\
        --model-b-id    moonshotai/Kimi-K2-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# =========================================================================
# helpers
# =========================================================================

def _load_token_ids_from_layer1(path: Path, min_confidence: float = 0.0) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if float(row.get("confidence", 0.0)) >= min_confidence:
                    ids.append(int(row["token_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return ids


def _load_token_ids_from_registry(registry) -> list[int]:
    ids: list[int] = []
    ids.extend(e.token_id for e in registry.iter_blacklist())
    ids.extend(e.token_id for e in registry.iter_substitution())
    return list(set(ids))


# =========================================================================
# sub-command: detect
# =========================================================================

def _cmd_detect(args: argparse.Namespace) -> int:
    from .detection_eval import run_detection_eval_from_files
    from .report import EvalReport

    if not Path(args.ground_truth).exists():
        print(f"[eval:detect] ground-truth not found: {args.ground_truth}",
              file=sys.stderr)
        return 2

    result = run_detection_eval_from_files(
        ground_truth_path=args.ground_truth,
        magikarp_csv=args.magikarp_csv,
        layer1_jsonl=args.layer1_jsonl,
        min_confidence=args.min_confidence,
    )

    report = EvalReport(
        model_a_id=args.model_a_id or "",
        model_b_ids=[args.model_b_id] if args.model_b_id else [],
        dataset_info={
            "ground_truth": args.ground_truth,
            "magikarp_csv": args.magikarp_csv,
            "layer1_jsonl": args.layer1_jsonl,
            "universe_size": result.universe,
            "ground_truth_positives": result.ground_truth_size,
        },
    )
    report.attach_detection(result)
    report.print_summary()

    out = json.dumps(report.to_dict(), indent=2)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(out, encoding="utf-8")
        print(f"[eval] Report saved to {args.report_out}")
    else:
        print(out)
    return 0


# =========================================================================
# sub-command: mitigate
# =========================================================================

def _cmd_mitigate(args: argparse.Namespace) -> int:
    from .mitigation_eval import from_event_log
    from .report import EvalReport

    if not Path(args.event_log).exists():
        print(f"[eval:mitigate] event log not found: {args.event_log}",
              file=sys.stderr)
        return 2

    result = from_event_log(args.event_log)

    report = EvalReport(
        model_a_id=args.model_a_id or "",
        dataset_info={"event_log": args.event_log,
                      "events_analysed": result.events_analysed},
    )
    report.attach_mitigation(result)
    report.print_summary()

    out = json.dumps(report.to_dict(), indent=2)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(out, encoding="utf-8")
        print(f"[eval] Report saved to {args.report_out}")
    else:
        print(out)
    return 0


# =========================================================================
# sub-command: latency
# =========================================================================

def _cmd_latency(args: argparse.Namespace) -> int:
    from ..layer1_detection.bloom_filter import BloomGate
    from ..layer3_registry.registry import TokenRegistry
    from ..layer4_mitigation.pipeline import MitigationPipeline
    from .latency_eval import run_all_benches
    from .report import EvalReport

    bloom_path = Path(args.bloom_path)
    if not bloom_path.exists():
        print(f"[eval:latency] bloom not found at {bloom_path}. "
              "Run offline setup first.", file=sys.stderr)
        return 2

    bloom = BloomGate.load(bloom_path)

    with TokenRegistry(args.registry_root, backend_kind=args.backend) as registry:
        # Build a minimal no-model pipeline for timing.
        pipeline = MitigationPipeline(bloom=bloom, registry=registry)

        # Token id sample: prefer flagged ids for realistic hit-rate.
        l1_path = Path(args.layer1_jsonl)
        if l1_path.exists():
            token_ids = _load_token_ids_from_layer1(l1_path)
        else:
            token_ids = _load_token_ids_from_registry(registry)

        if not token_ids:
            print("[eval:latency] no token ids found; using synthetic ids 0..999")
            token_ids = list(range(1000))

        # Use a mix: ~50% flagged (bloom hits), ~50% random (pass-throughs).
        import random
        random.seed(42)
        random_ids = random.sample(range(200_000), min(len(token_ids), 2000))
        mixed_ids = token_ids[:2000] + random_ids
        random.shuffle(mixed_ids)

        bench = run_all_benches(
            bloom=bloom,
            registry=registry,
            pipeline=pipeline,
            token_ids=mixed_ids,
            bloom_iters=args.bloom_iters,
            registry_iters=args.registry_iters,
            pipeline_iters=args.pipeline_iters,
        )

    report = EvalReport(
        model_a_id=args.model_a_id or "",
        dataset_info={
            "token_sample_size": len(mixed_ids),
            "bloom_iters":    args.bloom_iters,
            "registry_iters": args.registry_iters,
            "pipeline_iters": args.pipeline_iters,
        },
    )
    report.attach_latency(bench)
    report.print_summary()

    # Print latency table separately.
    _print_latency_table(bench)

    out = json.dumps(report.to_dict(), indent=2)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(out, encoding="utf-8")
        print(f"[eval] Report saved to {args.report_out}")
    else:
        print(out)
    return 0


def _print_latency_table(bench: dict) -> None:
    targets = bench.get("pdf_targets", {})
    meets   = bench.get("meets_targets", {})
    rows = [
        ("Bloom lookup",    "bloom_lookup_us",         targets.get("bloom_lookup_us",      0.1)),
        ("Registry lookup", "registry_lookup_us",      targets.get("registry_lookup_us",   8.0)),
        ("Pipeline/token",  "pipeline_per_token_us",   targets.get("pipeline_per_token_us",20.0)),
        ("Pipeline/call",   "pipeline_per_call_us",    None),
        ("Bloom build",     "bloom_build_us",          None),
    ]
    print(f"\n  {'Stage':<22} {'Mean µs':>9} {'P95 µs':>9} {'P99 µs':>9} "
          f"{'Target µs':>10} {'Meet?':>6}")
    print("  " + "─" * 70)
    for label, key, target in rows:
        s = bench.get(key, {})
        mean = s.get("mean_us", "–")
        p95  = s.get("p95_us",  "–")
        p99  = s.get("p99_us",  "–")
        target_str = f"≤ {target:.1f}" if target is not None else "—"
        meet_key = {
            "bloom_lookup_us":         "bloom_lookup",
            "registry_lookup_us":      "registry_lookup",
            "pipeline_per_token_us":   "pipeline_per_token",
        }.get(key)
        meet_str = ""
        if meet_key:
            v = meets.get(meet_key)
            meet_str = "✓" if v else ("✗" if v is False else "N/A")
        print(f"  {label:<22} {str(mean):>9} {str(p95):>9} {str(p99):>9} "
              f"{target_str:>10} {meet_str:>6}")
    print()


# =========================================================================
# sub-command: full
# =========================================================================

def _cmd_full(args: argparse.Namespace) -> int:
    from ..layer1_detection.bloom_filter import BloomGate
    from ..layer3_registry.registry import TokenRegistry
    from ..layer4_mitigation.pipeline import MitigationPipeline
    from .detection_eval import run_detection_eval_from_files
    from .latency_eval import run_all_benches
    from .mitigation_eval import from_event_log
    from .report import EvalReport

    report = EvalReport(
        model_a_id=args.model_a_id or "",
        model_b_ids=[args.model_b_id] if args.model_b_id else [],
        dataset_info={},
    )

    # ---- Detection section -------------------------------------------------
    gt_path = Path(args.ground_truth) if args.ground_truth else None
    if gt_path and gt_path.exists():
        det = run_detection_eval_from_files(
            ground_truth_path=gt_path,
            magikarp_csv=args.magikarp_csv,
            layer1_jsonl=args.layer1_jsonl,
            min_confidence=args.min_confidence,
        )
        report.attach_detection(det)
        report.dataset_info.update({
            "ground_truth": str(gt_path),
            "universe_size": det.universe,
            "ground_truth_positives": det.ground_truth_size,
        })
    else:
        print("[eval:full] --ground-truth not provided/found; skipping detection.",
              file=sys.stderr)

    # ---- Mitigation section ------------------------------------------------
    ev_path = Path(args.event_log) if args.event_log else None
    if ev_path and ev_path.exists():
        mit = from_event_log(ev_path)
        report.attach_mitigation(mit)
        report.dataset_info["events_analysed"] = mit.events_analysed
    else:
        print("[eval:full] --event-log not found; skipping mitigation eval.",
              file=sys.stderr)

    # ---- Latency section ---------------------------------------------------
    bloom_path = Path(args.bloom_path)
    if bloom_path.exists():
        bloom = BloomGate.load(bloom_path)
        with TokenRegistry(args.registry_root, backend_kind=args.backend) as registry:
            pipeline = MitigationPipeline(bloom=bloom, registry=registry)
            l1_path = Path(args.layer1_jsonl)
            token_ids = (
                _load_token_ids_from_layer1(l1_path)
                if l1_path.exists()
                else _load_token_ids_from_registry(registry)
            )
            if not token_ids:
                token_ids = list(range(1000))
            bench = run_all_benches(
                bloom=bloom,
                registry=registry,
                pipeline=pipeline,
                token_ids=token_ids[:4000],
                bloom_iters=args.bloom_iters,
                registry_iters=args.registry_iters,
                pipeline_iters=args.pipeline_iters,
            )
        report.attach_latency(bench)
    else:
        print("[eval:full] bloom filter not found; skipping latency bench.",
              file=sys.stderr)

    # ---- Print + save -------------------------------------------------------
    report.print_summary()
    if bloom_path.exists():
        _print_latency_table(report.latency or {})

    out = json.dumps(report.to_dict(), indent=2)
    if args.report_out:
        saved = report.save(args.report_out)
        print(f"[eval] Full report saved to {saved}")
    else:
        print(out)
    return 0


# =========================================================================
# parser
# =========================================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-a-id",   default="",   help="Model A HF id (metadata only)")
    p.add_argument("--model-b-id",   default="",   help="Model B HF id (metadata only)")
    p.add_argument("--report-out",   default=None, help="Path to write the JSON report")


def _add_registry_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--registry-root",
                   default="data/mitigation_system/rocksdb_storage")
    p.add_argument("--backend",
                   choices=["auto", "rocksdict", "jsondir"], default="auto")
    p.add_argument("--bloom-path",
                   default="data/mitigation_system/bloom_filter.bin")


def _add_detection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ground-truth",
                   default=None,
                   help="Ground-truth file: layer2 JSONL, kimi JSON, or custom JSONL")
    p.add_argument("--magikarp-csv",
                   default="data/magikarp_comparison.csv")
    p.add_argument("--layer1-jsonl",
                   default="data/layer1_candidates.jsonl")
    p.add_argument("--min-confidence", type=float, default=0.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HMUTTMS Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- detect ------------------------------------------------------------
    det = sub.add_parser("detect",
                         help="Detection accuracy: Magikarp-only vs. hybrid Layer-1")
    _add_common(det)
    _add_detection_args(det)
    det.set_defaults(func=_cmd_detect)

    # ---- mitigate ----------------------------------------------------------
    mit = sub.add_parser("mitigate",
                         help="Mitigation success rates from a RuntimeMonitor event log")
    _add_common(mit)
    mit.add_argument("--event-log",
                     default="data/mitigation_system/feedback/events.jsonl")
    mit.set_defaults(func=_cmd_mitigate)

    # ---- latency -----------------------------------------------------------
    lat = sub.add_parser("latency",
                         help="Micro-benchmark Bloom / RocksDB / pipeline overhead")
    _add_common(lat)
    _add_registry_args(lat)
    lat.add_argument("--layer1-jsonl", default="data/layer1_candidates.jsonl")
    lat.add_argument("--bloom-iters",    type=int, default=10_000)
    lat.add_argument("--registry-iters", type=int, default=1_000)
    lat.add_argument("--pipeline-iters", type=int, default=500)
    lat.set_defaults(func=_cmd_latency)

    # ---- full --------------------------------------------------------------
    full = sub.add_parser("full",
                          help="Run detect + mitigate + latency and emit one report")
    _add_common(full)
    _add_detection_args(full)
    _add_registry_args(full)
    full.add_argument("--event-log",
                      default="data/mitigation_system/feedback/events.jsonl")
    full.add_argument("--bloom-iters",    type=int, default=10_000)
    full.add_argument("--registry-iters", type=int, default=1_000)
    full.add_argument("--pipeline-iters", type=int, default=500)
    full.set_defaults(func=_cmd_full)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
