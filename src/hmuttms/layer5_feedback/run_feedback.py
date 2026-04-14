"""Layer-5 CLI — drive the feedback loop offline.

Two sub-commands:

* ``escalate``  — fold a RuntimeMonitor event log into registry counters
  and promote under-performing tokens up the track ladder.

* ``retest``    — re-run Layer-2 classification against one or more
  Model-B backends and update the registry where newer evidence
  disagrees with the stored label.

Example::

    python -m hmuttms.layer5_feedback.run_feedback escalate \
        --event-log data/mitigation_system/feedback/events.jsonl

    python -m hmuttms.layer5_feedback.run_feedback retest \
        --cached-file data/kimi_k3_magikarp_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..layer2_validation.model_b_tester import (
    BaseModelBTester,
    CachedFileTester,
    LocalHFTester,
)
from ..layer3_registry.registry import DEFAULT_ROOT, TokenRegistry
from .escalator import EscalationConfig, Escalator
from .retest_scheduler import RetestScheduler


# ============================================================== subcommand: escalate
def _cmd_escalate(args: argparse.Namespace) -> int:
    cfg = EscalationConfig(
        promote_below=args.promote_below,
        restore_above=args.restore_above,
        min_attempts=args.min_attempts,
    )
    with TokenRegistry(args.registry_root, backend_kind=args.backend) as reg:
        esc = Escalator(reg, config=cfg, cursor_path=args.cursor)
        summary = esc.process_log(args.event_log)
    print(json.dumps(summary, indent=2))
    return 0


# =============================================================== subcommand: retest
def _build_testers(args: argparse.Namespace) -> list[BaseModelBTester]:
    testers: list[BaseModelBTester] = []
    for path in args.cached_file or []:
        testers.append(CachedFileTester(path))
    for model_id in args.hf_model or []:
        testers.append(
            LocalHFTester(
                model_id=model_id,
                device=args.hf_device,
                trust_remote_code=args.hf_trust_remote_code,
                dtype=args.hf_dtype,
                max_new_tokens=args.hf_max_new_tokens,
            )
        )
    return testers


def _cmd_retest(args: argparse.Namespace) -> int:
    testers = _build_testers(args)
    if not testers:
        print(
            "[feedback:retest] need at least one --cached-file or --hf-model",
            file=sys.stderr,
        )
        return 2

    with TokenRegistry(args.registry_root, backend_kind=args.backend) as reg:
        sched = RetestScheduler(reg, testers=testers)
        report = sched.run(
            refresh_bloom=not args.skip_bloom_refresh,
            bloom_path=args.bloom_output,
        )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


# ============================================================ arg parsers
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HMUTTMS Layer 5 — feedback loop")
    p.add_argument("--registry-root", default=str(DEFAULT_ROOT))
    p.add_argument(
        "--backend",
        choices=["auto", "rocksdict", "jsondir"],
        default="auto",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # escalate
    esc = sub.add_parser("escalate", help="Fold event log into registry counters")
    esc.add_argument(
        "--event-log",
        default="data/mitigation_system/feedback/events.jsonl",
    )
    esc.add_argument(
        "--cursor",
        default="data/mitigation_system/feedback/.cursor",
    )
    esc.add_argument("--promote-below", type=float, default=0.40)
    esc.add_argument("--restore-above", type=float, default=0.85)
    esc.add_argument("--min-attempts", type=int, default=5)
    esc.set_defaults(func=_cmd_escalate)

    # retest
    rt = sub.add_parser("retest", help="Re-classify stored tokens against new Model-B(s)")
    rt.add_argument("--cached-file", action="append")
    rt.add_argument("--hf-model", action="append")
    rt.add_argument("--hf-device", default="auto")
    rt.add_argument(
        "--hf-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    rt.add_argument("--hf-trust-remote-code", action="store_true")
    rt.add_argument("--hf-max-new-tokens", type=int, default=24)
    rt.add_argument(
        "--bloom-output",
        default="data/mitigation_system/bloom_filter.bin",
    )
    rt.add_argument("--skip-bloom-refresh", action="store_true")
    rt.set_defaults(func=_cmd_retest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
