"""Latency evaluation — micro-benchmark every HMUTTMS lookup stage.

Measures five distinct overhead sources independently so the thesis can
report each against the PDF's target of ~10-20 µs total per token:

    Stage                        PDF target    Measured here
    ─────────────────────────────────────────────────────────
    Bloom filter lookup          0.1 µs        bloom_bench()
    RocksDB / JsonDir lookup     8   µs        registry_bench()
    Full pipeline (per token)    10–20 µs      pipeline_bench()
    Bloom build (one-time)       —             bloom_build_bench()
    Registry population (1 row)  —             populate_bench()

All timings are in **microseconds (µs)**. Each bench returns a
:class:`~hmuttms.evaluation.metrics.LatencyStats` populated with
``n_iterations`` samples.
"""
from __future__ import annotations

import time
from typing import Optional

from .metrics import LatencyStats


# ------------------------------------------------------------- bloom lookup
def bloom_bench(
    bloom,
    token_ids: list[int],
    n_iterations: int = 10_000,
) -> LatencyStats:
    """Benchmark :meth:`BloomGate.contains` over ``token_ids``."""
    stats = LatencyStats()
    ids_cycle = token_ids * (n_iterations // max(len(token_ids), 1) + 1)
    ids_cycle = ids_cycle[:n_iterations]
    for tid in ids_cycle:
        t0 = time.perf_counter_ns()
        bloom.contains(tid)
        stats.add((time.perf_counter_ns() - t0) / 1_000.0)
    return stats


# ------------------------------------------------------------- registry lookup
def registry_bench(
    registry,
    token_ids: list[int],
    n_iterations: int = 1_000,
) -> LatencyStats:
    """Benchmark :meth:`TokenRegistry.lookup` for each token id."""
    stats = LatencyStats()
    ids_cycle = token_ids * (n_iterations // max(len(token_ids), 1) + 1)
    ids_cycle = ids_cycle[:n_iterations]
    for tid in ids_cycle:
        t0 = time.perf_counter_ns()
        registry.lookup(tid)
        stats.add((time.perf_counter_ns() - t0) / 1_000.0)
    return stats


# -------------------------------------------------------------- full pipeline
def pipeline_bench(
    pipeline,
    sample_sequences: list[list[int]],
    n_iterations: int = 500,
) -> dict[str, LatencyStats]:
    """Benchmark :meth:`MitigationPipeline.mitigate` end-to-end.

    Returns per-token and per-call :class:`LatencyStats`.
    """
    per_call = LatencyStats()
    per_token = LatencyStats()

    seqs_cycle = sample_sequences * (n_iterations // max(len(sample_sequences), 1) + 1)
    seqs_cycle = seqs_cycle[:n_iterations]

    for seq in seqs_cycle:
        t0 = time.perf_counter_ns()
        pipeline.mitigate(seq)
        elapsed_us = (time.perf_counter_ns() - t0) / 1_000.0
        per_call.add(elapsed_us)
        if seq:
            per_token.add(elapsed_us / len(seq))

    return {"per_call": per_call, "per_token": per_token}


# ------------------------------------------------------- bloom build bench
def bloom_build_bench(
    token_ids: list[int],
    target_fpr: float = 0.001,
    n_iterations: int = 20,
) -> LatencyStats:
    """Benchmark :meth:`BloomGate.build_from_ids` (offline, not on hot path)."""
    from ..layer1_detection.bloom_filter import BloomGate

    stats = LatencyStats()
    for _ in range(n_iterations):
        t0 = time.perf_counter_ns()
        BloomGate.build_from_ids(token_ids, target_fpr=target_fpr)
        stats.add((time.perf_counter_ns() - t0) / 1_000.0)
    return stats


# ------------------------------------------------- registry populate bench
def populate_bench(
    registry,
    sample_entries: list[dict],
    n_iterations: int = 200,
) -> LatencyStats:
    """Benchmark writing one blacklist row + one metadata row per call."""
    from ..layer3_registry.schemas import BlacklistEntry, MetadataEntry

    stats = LatencyStats()
    entries_cycle = sample_entries * (n_iterations // max(len(sample_entries), 1) + 1)
    entries_cycle = entries_cycle[:n_iterations]

    for row in entries_cycle:
        tid = int(row.get("token_id", 0))
        t0 = time.perf_counter_ns()
        registry.put_blacklist(
            BlacklistEntry(
                token_id=tid,
                token_string=row.get("token_string"),
                model_a_score=float(row.get("model_a_score", 0.1)),
                model_b_score=float(row.get("model_b_score", 0.2)),
            )
        )
        registry.upsert_metadata(MetadataEntry(token_id=tid))
        stats.add((time.perf_counter_ns() - t0) / 1_000.0)
    return stats


# ----------------------------------------------------------------- summary
def run_all_benches(
    bloom,
    registry,
    pipeline,
    token_ids: list[int],
    *,
    bloom_iters: int = 10_000,
    registry_iters: int = 1_000,
    pipeline_iters: int = 500,
    sample_seq_len: int = 64,
) -> dict:
    """Run all five benchmarks and return a unified summary dict."""
    # Build sample sequences from the given token_ids.
    sample_seqs: list[list[int]] = []
    for i in range(0, max(len(token_ids), sample_seq_len), sample_seq_len):
        chunk = token_ids[i: i + sample_seq_len]
        if chunk:
            sample_seqs.append(chunk)
    if not sample_seqs:
        sample_seqs = [token_ids[:sample_seq_len] or [0]]

    bloom_stats      = bloom_bench(bloom, token_ids, n_iterations=bloom_iters)
    registry_stats   = registry_bench(registry, token_ids, n_iterations=registry_iters)
    pipeline_results = pipeline_bench(pipeline, sample_seqs, n_iterations=pipeline_iters)
    build_stats      = bloom_build_bench(token_ids, n_iterations=20)

    return {
        "bloom_lookup_us":   bloom_stats.to_dict(),
        "registry_lookup_us": registry_stats.to_dict(),
        "pipeline_per_call_us": pipeline_results["per_call"].to_dict(),
        "pipeline_per_token_us": pipeline_results["per_token"].to_dict(),
        "bloom_build_us":    build_stats.to_dict(),
        "pdf_targets": {
            "bloom_lookup_us":       0.1,
            "registry_lookup_us":    8.0,
            "pipeline_per_token_us": 20.0,
        },
        "meets_targets": {
            "bloom_lookup":   bloom_stats.mean <= 0.1,
            "registry_lookup": registry_stats.mean <= 8.0,
            "pipeline_per_token": (
                pipeline_results["per_token"].mean <= 20.0
            ),
        },
    }
