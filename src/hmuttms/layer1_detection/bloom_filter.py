"""Bloom filter pre-screening layer for HMUTTMS Layer 1.

Deployed at tokenizer level so the runtime can reject the vast majority of
normal tokens in sub-microsecond time before any RocksDB lookup.

Design notes
------------
* Pure-Python / ``hashlib``-only — zero external dependencies, works the
  same on Windows and Linux.
* Double-hashing scheme (Kirsch-Mitzenmacher): derive ``k`` hash positions
  from a single BLAKE2b digest. Faster than computing ``k`` independent
  hashes and still has well-studied false-positive behaviour.
* Keyed by **token_id** (int). The string form is optional, only stored in
  RocksDB metadata in Layer 3.
* Serialization format (v1): a small JSON header followed by the raw
  bit-array bytes — trivial to inspect and version-bump later.

Example
-------
>>> gate = BloomGate.build_from_ids([12345, 67890], target_fpr=0.001)
>>> gate.contains(12345)
True
>>> gate.contains(99999)            # may very rarely return True (FPR)
False
>>> gate.save("data/mitigation_system/bloom_filter.bin")
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAGIC = b"HMBF"
VERSION = 1


def _optimal_params(n: int, fpr: float) -> tuple[int, int]:
    """Return (num_bits, num_hashes) for n items at target false-positive rate."""
    if n <= 0:
        n = 1
    fpr = max(min(fpr, 0.5), 1e-9)
    m = -(n * math.log(fpr)) / (math.log(2) ** 2)
    k = (m / n) * math.log(2)
    return max(int(math.ceil(m)), 8), max(int(round(k)), 1)


def _hash_positions(key: bytes, num_bits: int, num_hashes: int) -> Iterable[int]:
    """Derive ``num_hashes`` bit positions from a single BLAKE2b digest.

    Uses the Kirsch-Mitzenmacher trick: ``g_i(x) = h1(x) + i * h2(x) mod m``.
    """
    digest = hashlib.blake2b(key, digest_size=16).digest()
    h1 = int.from_bytes(digest[:8], "little", signed=False)
    h2 = int.from_bytes(digest[8:], "little", signed=False) or 1
    for i in range(num_hashes):
        yield (h1 + i * h2) % num_bits


@dataclass
class BloomGate:
    """Space-efficient membership filter for flagged token IDs."""

    num_bits: int
    num_hashes: int
    bits: bytearray
    count: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def empty(cls, expected_items: int, target_fpr: float = 0.001) -> "BloomGate":
        num_bits, num_hashes = _optimal_params(expected_items, target_fpr)
        return cls(num_bits=num_bits, num_hashes=num_hashes, bits=bytearray((num_bits + 7) // 8))

    @classmethod
    def build_from_ids(
        cls,
        token_ids: Iterable[int],
        target_fpr: float = 0.001,
    ) -> "BloomGate":
        ids = list({int(t) for t in token_ids})
        gate = cls.empty(expected_items=max(len(ids), 1), target_fpr=target_fpr)
        for tid in ids:
            gate.add(tid)
        return gate

    # --------------------------------------------------------------- mutation
    def _key(self, token_id: int) -> bytes:
        return struct.pack("<q", int(token_id))

    def add(self, token_id: int) -> None:
        for pos in _hash_positions(self._key(token_id), self.num_bits, self.num_hashes):
            self.bits[pos >> 3] |= 1 << (pos & 7)
        self.count += 1

    def contains(self, token_id: int) -> bool:
        for pos in _hash_positions(self._key(token_id), self.num_bits, self.num_hashes):
            if not (self.bits[pos >> 3] >> (pos & 7)) & 1:
                return False
        return True

    __contains__ = contains

    # ------------------------------------------------------------- properties
    @property
    def expected_fpr(self) -> float:
        """Current expected false-positive rate given items inserted."""
        if self.count == 0:
            return 0.0
        k, m, n = self.num_hashes, self.num_bits, self.count
        return (1.0 - math.exp(-k * n / m)) ** k

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "version": VERSION,
                "num_bits": self.num_bits,
                "num_hashes": self.num_hashes,
                "count": self.count,
            }
        ).encode("utf-8")
        with path.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(header)))
            f.write(header)
            f.write(bytes(self.bits))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BloomGate":
        path = Path(path)
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Not a BloomGate file: {path} (magic={magic!r})")
            (hlen,) = struct.unpack("<I", f.read(4))
            header = json.loads(f.read(hlen).decode("utf-8"))
            if header.get("version") != VERSION:
                raise ValueError(f"Unsupported bloom version: {header.get('version')}")
            bits = bytearray(f.read())
        expected_bytes = (header["num_bits"] + 7) // 8
        if len(bits) != expected_bytes:
            raise ValueError(
                f"Corrupt bloom bits: got {len(bits)} bytes, expected {expected_bytes}"
            )
        return cls(
            num_bits=header["num_bits"],
            num_hashes=header["num_hashes"],
            bits=bits,
            count=header.get("count", 0),
        )
