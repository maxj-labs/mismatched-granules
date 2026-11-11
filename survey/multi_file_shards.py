"""Virtual shards whose source chunks sit in several files: MODIS-Aqua chlorophyll, daily
4 km maps across the 2021 chunk-shape change, with three days per shard.

    BENCH_HTTP=http://127.0.0.1:8766/ python multi_file_shards.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import icechunk
import numpy as np
import zarr
from zarr.codecs import SwitchCodec

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "bench"))

from build import DATA, HTTP, new_repo, open_repo  # noqa: E402
from compare_util import Counting, reset_source, source_stats  # noqa: E402
from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import LocalStore  # noqa: E402
from virtualizarr.manifests import ManifestArray  # noqa: E402
from virtualizarr.manifests.array_api import _concatenate_as_one_array, stack  # noqa: E402
from virtualizarr.manifests.concat import concatenate_keeping_pieces  # noqa: E402
from virtualizarr.manifests.plan import plan_joined_array  # noqa: E402
from virtualizarr.manifests.utils import copy_and_replace_metadata  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402

warnings.filterwarnings("ignore")
assert HTTP
zarr.config.set({"array.rectilinear_chunks": True})
files = [DATA / f for f in json.loads((Path(__file__).parent / "picks.json").read_text())["MODISA_L3m_CHL"]["files"]]
registry = ObjectStoreRegistry({"file://": LocalStore()})
arrays = []
for f in files:
    a = HDFParser()(url=f"file://{f}", registry=registry)._group.arrays["chlor_a"]
    a = ManifestArray(metadata=copy_and_replace_metadata(a.metadata, new_dimension_names=None), chunkmanifest=a.manifest)
    arrays.append(stack([a], axis=0).rename_paths(lambda p: p.replace(f"file://{DATA}/", HTTP)))
joined = concatenate_keeping_pieces(arrays, 0, merge=_concatenate_as_one_array)
truth = np.stack([h5py.File(f, "r")["chlor_a"][...] for f in files])
report = {"pieces": [p.shape[0] for p in joined.pieces], "rows": []}
rng = np.random.default_rng(0)
valid = np.argwhere(truth[0] != -32767)
py, px = (int(v) for v in valid[len(valid) // 2])
READS = {
    "6-day series at one cell": (slice(None), py, px),
    "6 days x 200x200 cells": (slice(None), slice(py - 100, py + 100), slice(px - 100, px + 100)),
    "one whole day": (slice(4, 5), slice(None), slice(None)),
}
for label, days in {"one day per shard": None, "three days per shard": 3}.items():
    plan = plan_joined_array(joined.pieces, axis=0, shard_length_along_axis=days)
    name = "modisa_shards_" + label.split()[0]
    repo = new_repo(name)
    session = repo.writable_session("main")
    arr = zarr.create_array(session.store, name="chlor_a", shape=plan.shape, chunks=plan.chunk_grid, dtype=plan.dtype, fill_value=plan.fill_value,
                            serializer=SwitchCodec(axis=0, cases=plan.cases), compressors=None, filters=None)
    assert not session.store.set_virtual_refs("chlor_a", [icechunk.VirtualChunkSpec(**r) for r in plan.refs], validate_containers=True)
    session.commit(label)
    files_per_ref = [1 + len({loc for loc, _, _ in (r.get("more_ranges") or []) if loc}) for r in plan.refs]
    info = {"layout": label, "references": len(plan.refs), "source_chunks": plan.source_chunks, "files_per_reference": sorted(set(files_per_ref)),
            "shard_shape": plan.notes["shard_shape"]}
    for read, sel in READS.items():
        times = []
        for _ in range(3):
            Counting.gets = 0
            reset_source()
            t0 = time.perf_counter()
            got = zarr.open_array(Counting(open_repo(name).readonly_session("main").store), path="chlor_a", mode="r")[sel]
            times.append(time.perf_counter() - t0)
        report["rows"].append({**info, "read": read, "equal_to_files": bool(np.array_equal(got, truth[sel])), "seconds": round(sorted(times)[1], 3),
                               "zarr_reads": Counting.gets, **source_stats()})
        print(json.dumps(report["rows"][-1]), flush=True)
(HERE / "results_http" / "multi_file_shards.json").write_text(json.dumps(report, indent=1))
