"""North America 2019-2021 as ONE ordinary array, across the chunk-shape change: virtual shards.

Each Zarr chunk is a shard of 10 x 3000 x 3000 cells, the least common multiple of the
two layouts (1 x 1000 x 1000 before 2020, 10 x 300 x 300 after). A shard is not a stored
object. Its reference is a concatenation: a literal shard index, then the byte ranges of
the source chunks it holds (90 of the old layout, or 100 of the new). zarr's own
`sharding_indexed` codec reads it by range, so a small read fetches the index and only
the source chunks it needs.

A position-based `switch` codec picks the sharding configuration per era and crops the
shards that are cut short (the 5-day shard at each year's end, and the grid's edges).

Needs the Icechunk branch that adds `prefix` and `more_ranges` to virtual references.
"""

import json
import statistics
import time
import warnings

import icechunk
import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from build import RESULTS, STORES, remote, DATA, HERE, LOADABLE, new_repo, open_repo
from compare_util import Counting, reset_source, source_stats
from switch_codec import SwitchCodec
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from workloads import Workloads

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True})
var, years = "dayl", [2019, 2020, 2021]
SHARD = (10, 3000, 3000)
MISSING = 2**64 - 1
paths = [DATA / f"daymet_v4_daily_na_{var}_{y}.nc" for y in years]
registry = ObjectStoreRegistry({"file://": LocalStore()})
vdss = [open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=LOADABLE) for p in paths]
vdss = [remote(v) for v in vdss]
arrays = [v[var].data for v in vdss]
dims = vdss[0][var].dims
report: dict = {"icechunk": icechunk.__version__}


def edges(n: int, c: int) -> list[int]:
    return [c] * (n // c) + ([n % c] if n % c else [])


t0 = time.time()
specs, cases, time_chunks, base = [], [], [], 0
n_ranges = 0
for a in arrays:
    md, m = a.metadata, a.manifest
    chunks = tuple(md.chunks)
    per_shard = tuple(s // c for s, c in zip(SHARD, chunks))
    assert all(s % c == 0 for s, c in zip(SHARD, chunks))
    index_nbytes = 16 * int(np.prod(per_shard))
    file_time = edges(a.shape[0], SHARD[0])
    n_shards = (len(file_time), -(-a.shape[1] // SHARD[1]), -(-a.shape[2] // SHARD[2]))
    grid = m._paths.shape
    for shard in np.ndindex(n_shards):
        index = np.full((*per_shard, 2), MISSING, dtype="<u8")
        parts, pos, location = [], index_nbytes, None
        for inner in np.ndindex(per_shard):
            src = tuple(s * p + i for s, p, i in zip(shard, per_shard, inner))
            if all(x < g for x, g in zip(src, grid)) and m._paths[src]:
                length = int(m._lengths[src])
                index[inner] = (pos, length)
                parts.append((int(m._offsets[src]), length))
                location = str(m._paths[src])
                pos += length
        if parts:
            n_ranges += len(parts)
            specs.append(icechunk.VirtualChunkSpec(
                index=[base + shard[0], shard[1], shard[2]], location=location, offset=parts[0][0], length=parts[0][1],
                prefix=index.tobytes(), more_ranges=[(None, o, n) for o, n in parts[1:]] or None))
    sharding = {"name": "sharding_indexed", "configuration": {
        "chunk_shape": list(chunks), "codecs": list(md.to_dict()["codecs"]),
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_location": "start"}}
    if cases and cases[-1]["codecs"] == [sharding]:
        cases[-1]["chunks"] += len(file_time)
    else:
        cases.append({"chunks": len(file_time), "codecs": [sharding], "encoded_chunk_shape": list(SHARD)})
    time_chunks += file_time
    base += len(file_time)

shape = (sum(a.shape[0] for a in arrays), *arrays[0].shape[1:])
repo = new_repo("bench_na_dayl_shards")
session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="a")
root.create_array(
    var, shape=shape, chunks=[time_chunks, edges(shape[1], SHARD[1]), edges(shape[2], SHARD[2])],
    dtype=arrays[0].dtype, fill_value=arrays[-1].metadata.fill_value,
    serializer=SwitchCodec(axis=0, cases=cases), compressors=None, filters=None, dimension_names=dims,
)
failed = session.store.set_virtual_refs(var, specs, validate_containers=True)
session.commit("virtual shards")
path = STORES / "bench_na_dayl_shards"
report["build"] = {
    "seconds": round(time.time() - t0, 1), "shards_referenced": len(specs), "source_chunks_referenced": n_ranges,
    "refs_failed": failed and len(failed), "store_bytes": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
    "cases": [{"shards_along_time": c["chunks"], "inner_chunk_shape": c["codecs"][0]["configuration"]["chunk_shape"]} for c in cases],
    "missing_source_chunks_2019": int((arrays[0].manifest._paths == "").sum()),
    "fill_values_per_file": [repr(a.metadata.fill_value) for a in arrays],
}
print(json.dumps(report["build"]), flush=True)

W = Workloads("na", var, 2019, 2021)
py, px = W.selections["series: every day at one cell"][1:]
by, bx = py - 300, px - 300
WORKLOADS = {
    "box: 30 days x 600x600 cells, across the 2019/2020 chunk-shape change": (slice(350, 380), slice(by, by + 600), slice(bx, bx + 600)),
    "box: 30 days x 600x600 cells, across the 2020/2021 file boundary": (slice(715, 745), slice(by, by + 600), slice(bx, bx + 600)),
    "map: 1 day, whole grid, 2019": (slice(100, 101), slice(None), slice(None)),
    "map: 1 day, whole grid, 2021": (slice(900, 901), slice(None), slice(None)),
    "series: every day (1,095) at one cell": (slice(0, 1095), py, px),
    "point: 1 day at one cell, 2021": (slice(900, 901), py, px),
}
NO_MERGE = {"array.sharding_coalesce_max_gap_bytes": 0}
ARMS = [
    ("concat, 3 pieces", "bench_na_dayl_concat", {}),
    ("hybrid, 2 pieces", "bench_na_dayl_hybrid", {}),
    # zarr merges ranges of a shard that lie within 1 MiB of each other. In a virtual
    # shard, neighbours in the layout are separate source chunks, so merging reads
    # chunks nobody asked for.
    ("virtual shards, zarr defaults", "bench_na_dayl_shards", {}),
    ("virtual shards, no range merging", "bench_na_dayl_shards", NO_MERGE),
]
rows = []
for workload, sel in WORKLOADS.items():
    want = W.truth(*sel)
    for approach, repo_name, zarr_config in ARMS:
        times = []
        for _ in range(3):
            Counting.gets = Counting.nbytes = 0
            reset_source()
            t0 = time.perf_counter()
            with zarr.config.set(zarr_config):
                got = zarr.open_array(Counting(open_repo(repo_name).readonly_session("main").store), path=var, mode="r")[sel]
            times.append(time.perf_counter() - t0)
        rows.append({"workload": workload, "approach": approach, "shape": list(got.shape), "equal_to_files": bool(np.array_equal(got, want)),
                     "seconds_median": round(statistics.median(times), 3), "store_gets": Counting.gets, "store_megabytes": round(Counting.nbytes / 1e6, 1), **source_stats()})
        print(json.dumps(rows[-1]), flush=True)
report["rows"] = rows
arr = zarr.open_array(open_repo("bench_na_dayl_shards").readonly_session("main").store, path=var, mode="r")
report["zarr_json_bytes"] = len(json.dumps(arr.metadata.to_dict(), default=str))
(RESULTS / "bench_na_dayl_shards.json").write_text(json.dumps(report, indent=1))
