"""Fill values that differ between granules: show the problem and check the fix.

Three small HDF5 "granules" per scenario. Each has its own HDF5 fill value, leaves some
chunks unwritten (so they read as that fill value), and marks "no data" with its own
sentinel (the CF `_FillValue` attribute). The joined array must read exactly as the
files do, raw and after masking the sentinel.

    python test_fill.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import zarr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "bench"))

import switch_codec  # noqa: E402,F401
from build import DATA, new_repo  # noqa: E402
from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import LocalStore  # noqa: E402
from unify import Granule, plan, write  # noqa: E402
from virtualizarr import open_virtual_dataset  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True, "array.sharding_coalesce_max_gap_bytes": 0})
SHAPE = (8, 20, 12)
FILLS = [-9999.0, 0.0, 7.5]  # HDF5 fill value of each granule
SENTINELS = [-9999.0, -1.0, -32767.0]  # CF _FillValue of each granule
SCENARIOS = {
    "switch (same chunk shape)": [(2, 5, 6), (2, 5, 6), (2, 5, 6)],
    "shards (chunk shape changes)": [(2, 5, 6), (4, 10, 3), (2, 5, 6)],
}
out_dir = DATA / "fill_test"
out_dir.mkdir(exist_ok=True)
registry = ObjectStoreRegistry({"file://": LocalStore()})
report = {}


def make(path: Path, k: int, chunks: tuple[int, ...]) -> None:
    rng = np.random.default_rng(k)
    with h5py.File(path, "w") as f:
        d = f.create_dataset("v", shape=SHAPE, chunks=chunks, dtype="f4", fillvalue=FILLS[k], compression="gzip", shuffle=bool(k % 2))
        d.attrs["_FillValue"] = np.float32(SENTINELS[k])
        data = rng.normal(size=SHAPE).astype("f4")
        data[rng.random(SHAPE) < 0.05] = SENTINELS[k]  # "no data" cells
        d[...] = data  # allocate everything first, then recreate with gaps
    # rewrite, leaving regions unwritten: a 4x10x6 block (a whole shard) and scattered chunks
    with h5py.File(path, "w") as f:
        d = f.create_dataset("v", shape=SHAPE, chunks=chunks, dtype="f4", fillvalue=FILLS[k], compression="gzip", shuffle=bool(k % 2))
        d.attrs["_FillValue"] = np.float32(SENTINELS[k])
        for idx in np.ndindex(*[s // c for s, c in zip(SHAPE, chunks)]):
            region = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunks))
            in_empty_block = all(r.start < e for r, e in zip(region, (4, 10, 6)))
            if in_empty_block or rng.random() < 0.2:
                continue
            d[region] = data[region]


for scenario, layouts in SCENARIOS.items():
    key = scenario.split()[0]
    paths = [out_dir / f"{key}_{k}.h5" for k in range(3)]
    for k, (p, chunks) in enumerate(zip(paths, layouts)):
        make(p, k, chunks)
    raw = [h5py.File(p, "r")["v"][...] for p in paths]  # h5py applies each file's own fill value
    want_raw = np.concatenate(raw)
    want_masked = np.concatenate([np.where(r == s, np.nan, r) for r, s in zip(raw, SENTINELS)])
    vdss = [open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=[]) for p in paths]
    stored = [int(h5py.File(p, "r")["v"].id.get_num_chunks()) for p in paths]
    rows = {}
    for label, kwargs in {
        "before: one fill value, one sentinel": dict(per_granule_fill=False, remap_sentinels=False),
        "per-granule fill only": dict(per_granule_fill=True, remap_sentinels=False),
        "per-granule fill and sentinel remap": dict(per_granule_fill=True, remap_sentinels=True),
    }.items():
        granules = [Granule.from_manifest_array(v["v"].data, stack=False, sentinel=SENTINELS[k]) for k, v in enumerate(vdss)]
        p = plan(granules, **kwargs)
        repo = new_repo(f"fill_{key}_{len(rows)}")
        session = repo.writable_session("main")
        arr = write(session.store, "v", p, granules[0].dtype)
        session.commit("fill test")
        got = zarr.open_array(repo.readonly_session("main").store, path="v", mode="r")[...]
        sentinel = p.sentinel
        masked = np.where(got == sentinel, np.nan, got)
        # the raw comparison is only meaningful when no value is rewritten
        raw_wrong = int((got != want_raw).sum()) if not kwargs["remap_sentinels"] else None
        rows[label] = {
            "strategy": p.strategy, "cases": len(p.cases), "array_fill": p.fill_json, "array_sentinel": sentinel,
            "raw_cells_wrong": raw_wrong, "masked_cells_wrong": int((~np.isclose(masked, want_masked, equal_nan=True)).sum()),
            "cells": int(got.size), "granules_with_own_fill": p.notes["granules_with_their_own_fill"],
            "granules_remapped": p.notes["granules_with_a_remapped_sentinel"],
            # partial reads take another path through the codec than whole-array reads
            "partial_reads_equal": bool(np.array_equal(arr[3:21, 2:17, 1:11], got[3:21, 2:17, 1:11])
                                        and np.array_equal(arr.vindex[[0, 9, 17, 23], [0, 19, 3, 11], [0, 5, 11, 2]],
                                                           got[[0, 9, 17, 23], [0, 19, 3, 11], [0, 5, 11, 2]])),
        }
        print(scenario, "|", label, "|", json.dumps(rows[label]), flush=True)
    report[scenario] = {"chunks_stored_per_granule": stored, "chunks_per_granule": [int(np.prod([s // c for s, c in zip(SHAPE, l)])) for l in layouts], "runs": rows}
(HERE / "results" / "fill_test.json").write_text(json.dumps(report, indent=1))
