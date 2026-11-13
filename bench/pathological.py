"""Worst cases for the `switch` codec: many cases, layouts that alternate every chunk.

    python pathological.py [chunks]

An in-memory store, tiny chunks (1 x 32 x 32 float32), so what is timed is the codec's own
overhead and not I/O or decompression. Chunk i uses case i % K: the layouts alternate
every chunk, which is the worst input for a table that is run-length encoded by position.

Arms: `plain` (an ordinary array, one chain; the floor), `position`, `key`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import SwitchCodec
from zarr.storage import MemoryStore, WrapperStore

zarr.config.set({"array.rectilinear_chunks": True})
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
SHAPE, CHUNK = (N, 32, 32), (1, 32, 32)
KEY_BYTES = 8


def chain(i: int) -> list[dict]:
    """A chain that differs from every other case's: its own blosc block size."""
    return [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle", "typesize": 4, "blocksize": 1024 + i}},
    ]


def key_of(i: int) -> bytes:
    from zarr.codecs.switch import case_key

    return case_key({"codecs": chain(i)}, KEY_BYTES)


class Prefixing(WrapperStore[MemoryStore]):
    """Serves the chunks of `key/` from those of `position/`, each with its case's key
    in front: what a store whose references carry a literal prefix does."""

    gets = 0

    def __init__(self, store: MemoryStore, keys: list[bytes] | None = None) -> None:
        super().__init__(store)
        self._keys = keys or []

    def _with_store(self, store: MemoryStore) -> "Prefixing":
        return type(self)(store, self._keys)

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        if not key.startswith("key/c/"):
            return await self._store.get(key, prototype, byte_range)
        whole = await self._store.get("position" + key[3:], prototype)
        i = int(key[6:].split("/")[0])
        data = self._keys[i % len(self._keys)] + whole.to_bytes()
        type(self).gets += 1
        if byte_range is None:
            return prototype.buffer.from_bytes(data)
        if hasattr(byte_range, "start"):
            return prototype.buffer.from_bytes(data[byte_range.start : byte_range.end])
        if hasattr(byte_range, "offset"):
            return prototype.buffer.from_bytes(data[byte_range.offset :])
        return prototype.buffer.from_bytes(data[-byte_range.suffix :])


def timed(fn) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run(k: int, data: np.ndarray, picks: np.ndarray) -> dict:
    store = MemoryStore()
    # position: chunk i is case i % k, so with k > 1 every chunk is its own run
    if k == 1:
        position_cases = [{"chunks": N, "codecs": chain(0)}]
    else:
        position_cases = [{"chunks": 1, "codecs": chain(i % k)} for i in range(N)]
    key_cases = [{"key": key_of(i).hex(), "codecs": chain(i)} for i in range(k)]
    common: dict = dict(shape=SHAPE, chunks=CHUNK, dtype="float32", fill_value=0, compressors=None, filters=None)
    writer = zarr.create_array(store, name="position", serializer=SwitchCodec(axis=0, cases=position_cases), **common)
    _, write_seconds = timed(lambda: writer.__setitem__(..., data))
    zarr.create_array(store, name="key", serializer=SwitchCodec(select="key", key_bytes=KEY_BYTES, cases=key_cases), **common)
    out: dict = {"cases": k, "write_seconds_position": round(write_seconds, 2), "arms": {}}
    arms = {"position": store, "key": Prefixing(store, [key_of(i) for i in range(k)])}
    if k == 1:
        plain = zarr.create_array(store, name="plain", **{**common, "compressors": None}, serializer="auto")
        plain[...] = data
        arms = {"plain": store, **arms}
    for name, s in arms.items():
        array, open_seconds = timed(lambda: zarr.open_array(s, path=name, mode="r"))
        got, all_seconds = timed(lambda: array[...])
        Prefixing.gets = 0
        some, some_seconds = timed(lambda: [array[int(i)] for i in picks])
        gets = Prefixing.gets
        out["arms"][name] = {
            "zarr_json_bytes": len(json.dumps(array.metadata.to_dict(), default=str)),
            "open_seconds": round(open_seconds, 4),
            "read_all_seconds": round(all_seconds, 2),
            "microseconds_per_chunk": round(all_seconds / N * 1e6, 1),
            "one_chunk_read_milliseconds": round(some_seconds / len(picks) * 1e3, 2),
            **({"store_reads_per_chunk_read": gets / len(picks)} if name == "key" else {}),
            "equal": bool(np.array_equal(got, data) and all(np.array_equal(a, data[int(i)]) for a, i in zip(some, picks))),
        }
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.normal(size=SHAPE).astype("float32")
    picks = rng.integers(0, N, 200)
    report = {"chunks": N, "chunk_shape": CHUNK, "runs": []}
    for k in (1, 2, 100, 2_000, N):
        rec = run(k, data, picks)
        report["runs"].append(rec)
        print(json.dumps(rec), flush=True)
    label = sys.argv[2] if len(sys.argv) > 2 else "pathological"
    out = Path(__file__).resolve().parent.parent / "results" / f"{label}.json"
    out.write_text(json.dumps(report, indent=1))
