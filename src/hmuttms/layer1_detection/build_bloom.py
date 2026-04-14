"""Build the Layer-1 Bloom filter from aggregated candidate tokens.

Reads the JSONL produced by :mod:`hmuttms.layer1_detection.run_layer1`
(default ``data/layer1_candidates.jsonl``) and writes the binary bloom
filter to ``data/mitigation_system/bloom_filter.bin``.

Usage (on the research PC)::

    python -m hmuttms.layer1_detection.build_bloom \
        --input  data/layer1_candidates.jsonl \
        --output data/mitigation_system/bloom_filter.bin \
        --fpr    0.001

Only candidates whose ``confidence`` is >= ``--min-confidence`` (default
0.5) are inserted. Lower the threshold if you want the bloom to catch
every Layer-1 hit regardless of confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from .bloom_filter import BloomGate


def _iter_candidates(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_ids(path: Path, min_confidence: float) -> list[int]:
    ids: list[int] = []
    for entry in _iter_candidates(path):
        conf = float(entry.get("confidence", 0.0))
        if conf < min_confidence:
            continue
        try:
            ids.append(int(entry["token_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMUTTMS Layer 1 — build bloom filter")
    p.add_argument(
        "--input",
        default="data/layer1_candidates.jsonl",
        help="Aggregated Layer-1 candidates JSONL",
    )
    p.add_argument(
        "--output",
        default="data/mitigation_system/bloom_filter.bin",
        help="Where to write the serialized bloom filter",
    )
    p.add_argument(
        "--fpr",
        type=float,
        default=0.001,
        help="Target false-positive rate (default 0.001)",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Skip candidates with confidence below this cutoff (default 0.5)",
    )
    p.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to write a small JSON stats sidecar",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[bloom] input not found: {in_path}", file=sys.stderr)
        return 2

    ids = _collect_ids(in_path, args.min_confidence)
    if not ids:
        print(
            f"[bloom] no candidates at confidence >= {args.min_confidence} "
            f"in {in_path}",
            file=sys.stderr,
        )
        return 1

    gate = BloomGate.build_from_ids(ids, target_fpr=args.fpr)
    out_path = gate.save(args.output)

    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "unique_tokens": len(set(ids)),
        "inserted_items": gate.count,
        "num_bits": gate.num_bits,
        "num_hashes": gate.num_hashes,
        "target_fpr": args.fpr,
        "expected_fpr": gate.expected_fpr,
        "min_confidence": args.min_confidence,
        "size_bytes": (gate.num_bits + 7) // 8,
    }
    print(json.dumps(stats, indent=2))

    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
