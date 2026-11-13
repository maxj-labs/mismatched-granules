"""How each option grows with the number of chunks, in the two worst cases.

    python scaling.py [chunks ...]        (default 1000 4000)

- alternate: two layouts (gzip; shuffle + gzip) that alternate chunk by chunk.
- packed:    every chunk is int16 with a scale and offset of its own.

Each chunk is one piece: a ManifestArray of one chunk cut from a real netCDF file, so the
planners and writers see what they would see for that many one-chunk granules.
"""

from __future__ import annotations

import json
import os
import sys
import time

import h5py
import netCDF4
import numpy as np
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.parsers import HDFParser
from zarr.core.metadata.v3 import ArrayV3Metadata

import arms
from cases import DATA, NX, NY, Case
from common import RESULTS, NotExpressible, registry, same

ARMS = ["switch_position", "switch_key", "switch_case_id", "switch_key_parameters", "switch_index_prefix", "switch_sidecar",
        "switch_store_context", "conditional_codec", "array_concat", "zarr_reference", "native_copy"]
REPEATS = 3  # every timing is the median of this many, each on a newly opened array


def source(name: str, n: int, dtype: str, **compression) -> tuple[ManifestArray, np.ndarray]:
    path = DATA / "scaling" / f"{name}_{n}.nc"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(len(name) + n)
        with netCDF4.Dataset(path, "w") as ds:
            ds.set_fill_off()
            for d, size in (("time", n), ("y", NY), ("x", NX)):
                ds.createDimension(d, size)
            v = ds.createVariable("v", dtype, ("time", "y", "x"), chunksizes=(1, NY, NX), **compression)
            v.set_auto_maskandscale(False)
            data = rng.normal(scale=300, size=(n, NY, NX))
            v[:] = np.round(data).astype(dtype) if dtype == "i2" else data.astype(dtype)
    ma = HDFParser()(url=f"file://{path}", registry=registry)._group.arrays["v"]
    with h5py.File(path, "r") as h:
        return ma, h["v"][...]


def one_chunk(ma: ManifestArray, i: int, codecs_in_front: list[dict] | None = None, dtype: str | None = None) -> ManifestArray:
    m = ma.manifest
    manifest = ChunkManifest.from_arrays(paths=np.asarray(m._paths)[i : i + 1], offsets=np.asarray(m._offsets)[i : i + 1],
                                         lengths=np.asarray(m._lengths)[i : i + 1])
    md = json.loads(json.dumps(ma.metadata.to_dict(), default=str))
    md["shape"] = [1, NY, NX]
    if codecs_in_front:
        md["codecs"] = [*codecs_in_front, *md["codecs"]]
        md["data_type"], md["fill_value"] = dtype, "NaN"
    return ManifestArray(metadata=ArrayV3Metadata.from_dict(md), chunkmanifest=manifest)


def scenario(kind: str, n: int) -> tuple[Case, list[ManifestArray]]:
    if kind == "alternate":
        a, raw_a = source("gzip", n // 2, "f4", zlib=True, complevel=1, shuffle=False)
        b, raw_b = source("shuffle_gzip", n // 2, "f4", zlib=True, complevel=1, shuffle=True)
        pieces = [one_chunk(a if i % 2 == 0 else b, i // 2) for i in range(n)]
        truth = np.empty((n, NY, NX), "float32")
        truth[0::2], truth[1::2] = raw_a, raw_b
        return Case(f"scaling_alternate_{n}", "two layouts alternating", truth=truth), pieces
    src, raw = source("int16", n, "i2", zlib=True, complevel=1, shuffle=False)
    scales = 0.01 + 0.000001 * np.arange(n)
    offsets = (np.arange(n) % 7) - 3.0
    pieces = [one_chunk(src, i, [
        {"name": "scale_offset", "configuration": {"offset": float(offsets[i]), "scale": float(1.0 / scales[i])}},
        {"name": "cast_value", "configuration": {"data_type": "int16", "rounding": "nearest-even", "scalar_map": {"encode": [["NaN", 0]]}}},
    ], "float64") for i in range(n)]
    truth = raw.astype("float64") * scales[:, None, None] + offsets[:, None, None]
    return Case(f"scaling_packed_{n}", "every chunk packed its own way", truth=truth, packed=True), pieces


def measure(kind: str, n: int) -> dict:
    case, pieces = scenario(kind, n)
    rng = np.random.default_rng(3)
    picks = rng.integers(0, n, 200)
    out: dict = {}
    for arm in [a for a in ARMS if a in os.environ.get("ONLY_ARMS", a)]:
        rec: dict = {}
        out[arm] = rec
        t0 = time.perf_counter()
        try:
            built = arms.ARMS[arm](case, list(pieces))
        except NotExpressible as err:
            rec.update(outcome="not expressible", why=str(err))
            continue
        except Exception as err:
            rec.update(outcome="error", why=f"{type(err).__name__}: {str(err)[:200]}")
            continue
        rec["build_seconds"] = round(time.perf_counter() - t0, 2)
        rec.update({k: built.info[k] for k in ("zarr_json_bytes", "metadata_bytes", "manifest_bytes", "virtual_bytes", "store_bytes", "native_chunk_bytes")})
        opens, ones, alls = [], [], []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            array = built.reopen()
            opens.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            some = [array[int(i)] for i in picks]
            ones.append((time.perf_counter() - t0) / len(picks))
            array = built.reopen()
            t0 = time.perf_counter()
            everything = array[...]
            alls.append(time.perf_counter() - t0)
        rec["open_seconds"] = round(float(np.median(opens)), 4)
        rec["one_chunk_milliseconds"] = round(float(np.median(ones)) * 1e3, 2)
        rec["read_all_seconds"] = round(float(np.median(alls)), 3)
        rec["read_all_spread"] = round((max(alls) - min(alls)) / float(np.median(alls)), 2)
        # wall time to read every chunk, divided by the number of chunks; chunks are read ten at a time
        rec["microseconds_per_chunk"] = round(rec["read_all_seconds"] / n * 1e6)
        ok = same(case, everything, case.truth) and all(same(case, s, case.truth[int(i)]) for s, i in zip(some, picks))
        rec["outcome"] = "equal" if ok else "differs"
        print(kind, n, arm, json.dumps(rec), flush=True)
    return out


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 4000]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "scaling.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for n in sizes:
        for kind in ("alternate", "packed"):
            report.setdefault(f"{kind}|{n}", {}).update(measure(kind, n))
            out_path.write_text(json.dumps(report, indent=1))
