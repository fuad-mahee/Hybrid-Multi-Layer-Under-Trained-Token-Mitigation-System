"""Layer-3 CLI — populate the RocksDB registry and refresh the bloom filter.

Typical end-to-end usage on the research PC::

    python -m hmuttms.layer1_detection.run_layer1 --model-id <A> ...
    python -m hmuttms.layer2_validation.run_layer2 --cached-file data/kimi_k2_magikarp_results.json
    python -m hmuttms.layer3_registry.run_populate                # default paths

Flags of note:

* ``--backend {auto,rocksdict,jsondir}`` — KV implementation. ``auto``
  tries RocksDict first, falls back to JsonDir if the native lib is
  missing (so the pipeline still runs on a laptop).
* ``--skip-bloom-refresh`` — skip rebuilding ``bloom_filter.bin``.
* ``--include-review`` — also persist ``REVIEW`` rows as metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .populate import populate_from_layer2, rebuild_bloom
from .registry import DEFAULT_ROOT, TokenRegistry


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMUTTMS Layer 3 — populate registry")
    p.add_argument("--input", default="data/layer2_classified.jsonl")
    p.add_argument("--registry-root", default=str(DEFAULT_ROOT))
    p.add_argument(
        "--backend",
        choices=["auto", "rocksdict", "jsondir"],
        default="auto",
    )
    p.add_argument("--include-review", action="store_true")

    # Bloom refresh
    p.add_argument(
        "--bloom-output",
        default="data/mitigation_system/bloom_filter.bin",
    )
    p.add_argument("--bloom-fpr", type=float, default=0.001)
    p.add_argument(
        "--bloom-exclude-substitution",
        action="store_true",
        help="Don't include SUBSTITUTABLE token ids in the bloom filter",
    )
    p.add_argument("--skip-bloom-refresh", action="store_true")

    p.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to write the full summary as JSON",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[layer3] input not found: {in_path}", file=sys.stderr)
        return 2

    summary: dict = {
        "input": str(in_path),
        "registry_root": args.registry_root,
        "backend": args.backend,
    }

    with TokenRegistry(args.registry_root, backend_kind=args.backend) as reg:
        counts = populate_from_layer2(
            in_path, reg, include_review=args.include_review
        )
        summary["ingest_counts"] = counts
        summary["sub_db_sizes"] = reg.counts()

        if not args.skip_bloom_refresh:
            summary["bloom"] = rebuild_bloom(
                reg,
                bloom_path=args.bloom_output,
                target_fpr=args.bloom_fpr,
                include_substitution=not args.bloom_exclude_substitution,
            )

    print(json.dumps(summary, indent=2))
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
