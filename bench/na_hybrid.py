"""North America, where the chunk shape changes: two ways to make one array of 2019-2021.

- concat : array_concat over 3 pieces (2019 | 2020 | 2021). 2020 and 2021 cannot merge,
           because 365 days in chunks of 10 leaves a partial chunk at the end of each file.
- hybrid : 2020 and 2021 become one ordinary array on a rectilinear time grid
           ([10]*36 + [5] per year), whose switch codec decodes each stored chunk at its
           stored shape (10, 300, 300) and crops it. array_concat then joins 2 pieces
           (2019 | 2020-2021): the chunk shape change is the only seam left.
"""

import json
import statistics
import time
import warnings

import h5py
import icechunk
import numpy as np
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from zarr.codecs import ArrayConcatCodec

from build import RESULTS, STORES, remote, DATA, HERE, LOADABLE, new_repo, open_repo
from compare_util import Counting, reset_source, source_stats
from switch_codec import SwitchCodec
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.writers.icechunk import write_virtual_variable_to_icechunk

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True})
var, years = "dayl", [2019, 2020, 2021]
paths = [DATA / f"daymet_v4_daily_na_{var}_{y}.nc" for y in years]
registry = ObjectStoreRegistry({"file://": LocalStore()})
vdss = [open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=LOADABLE) for p in paths]
vdss = [remote(v) for v in vdss]
arrays = [v[var].data for v in vdss]
dims = vdss[0][var].dims
report: dict = {}

# ---- hybrid store
t0 = time.time()
repo = new_repo("bench_na_dayl_hybrid")
session = repo.writable_session("main")
store = session.store
root = zarr.open_group(store, mode="a")
pieces = root.require_group(f"_pieces/{var}")
write_virtual_variable_to_icechunk(store=store, group=pieces, name="0", var=xr.Variable(dims, arrays[0]))
era2 = arrays[1:]
md = era2[0].metadata
ct, cy, cx = md.chunks


def edges(n: int, c: int) -> list[int]:
    return [c] * (n // c) + ([n % c] if n % c else [])


time_chunks = [n for a in era2 for n in [ct] * (a.shape[0] // ct) + ([a.shape[0] % ct] if a.shape[0] % ct else [])]
pieces.create_array(
    "1",
    shape=(sum(a.shape[0] for a in era2), *md.shape[1:]),
    # Icechunk 2.3.1 rejects a bare integer for a regular dimension of a rectilinear
    # grid (zarr-python writes `[..., 300, 300]`), so every dimension lists its chunks.
    chunks=[time_chunks, edges(md.shape[1], cy), edges(md.shape[2], cx)],
    dtype=md.data_type.to_native_dtype(),
    fill_value=md.fill_value,
    serializer=SwitchCodec(axis=0, cases=[{"chunks": len(time_chunks), "codecs": list(md.to_dict()["codecs"]), "encoded_chunk_shape": [ct, cy, cx]}]),
    compressors=None,
    filters=None,
    dimension_names=dims,
)
specs, base = [], 0
for a in era2:
    m = a.manifest
    for idx in np.ndindex(m._paths.shape):
        if m._paths[idx]:
            specs.append(icechunk.VirtualChunkSpec(index=[base + idx[0], idx[1], idx[2]], location=str(m._paths[idx]),
                                                   offset=int(m._offsets[idx]), length=int(m._lengths[idx])))
    base += m._paths.shape[0]
failed = store.set_virtual_refs(f"_pieces/{var}/1", specs, validate_containers=True)
root.create_array(
    var,
    shape=(sum(a.shape[0] for a in arrays), *md.shape[1:]),
    chunks=(sum(a.shape[0] for a in arrays), *md.shape[1:]),
    dtype=md.data_type.to_native_dtype(),
    fill_value=md.fill_value,
    serializer=ArrayConcatCodec(axis=0, pieces=[{"path": f"../_pieces/{var}/0", "size": arrays[0].shape[0]},
                                                {"path": f"../_pieces/{var}/1", "size": sum(a.shape[0] for a in era2)}]),
    compressors=None,
    filters=None,
    dimension_names=dims,
)
session.commit("hybrid")
report["hybrid_build"] = {"seconds": round(time.time() - t0, 1), "virtual_refs": len(specs), "refs_failed": failed and len(failed),
                          "time_chunks_era2": f"{len(time_chunks)} chunks: [10]*36 + [5], twice", "pieces": [365, 730]}

# ---- the reads
files = [h5py.File(p, "r") for p in paths]
offsets = [0, 365, 730, 1095]
valid = np.argwhere(files[0][var][0] != -9999)
py, px = (int(v) for v in valid[len(valid) // 2])
by, bx = py - 300, px - 300


def truth(t, y, x):
    parts = []
    for i, f in enumerate(files):
        lo, hi = max(t.start, offsets[i]), min(t.stop, offsets[i + 1])
        if lo < hi:
            parts.append(f[var][lo - offsets[i] : hi - offsets[i], y, x])
    return np.concatenate(parts)


WORKLOADS = {
    "box: 30 days x 600x600 cells, across the 2019/2020 chunk-shape change": (slice(350, 380), slice(by, by + 600), slice(bx, bx + 600)),
    "box: 30 days x 600x600 cells, across the 2020/2021 file boundary": (slice(715, 745), slice(by, by + 600), slice(bx, bx + 600)),
    "map: 1 day, whole grid, 2021": (slice(900, 901), slice(None), slice(None)),
    "series: every day (1,095) at one cell": (slice(0, 1095), py, px),
}
rows = []
for workload, sel in WORKLOADS.items():
    want = truth(*sel)
    for approach, repo_name in (("concat, 3 pieces", "bench_na_dayl_concat"), ("hybrid, 2 pieces", "bench_na_dayl_hybrid")):
        times = []
        for _ in range(3):
            Counting.gets = Counting.nbytes = 0; reset_source()
            t0 = time.perf_counter()
            got = zarr.open_array(Counting(open_repo(repo_name).readonly_session("main").store), path=var, mode="r")[sel]
            times.append(time.perf_counter() - t0)
        rows.append({"workload": workload, "approach": approach, "shape": list(got.shape), "equal_to_files": bool(np.array_equal(got, want)),
                     "seconds_median": round(statistics.median(times), 3), "store_gets": Counting.gets, "store_megabytes": round(Counting.nbytes / 1e6, 1), **source_stats()})
        print(json.dumps(rows[-1]), flush=True)
report["rows"] = rows
(RESULTS / "bench_na_dayl_hybrid.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report["hybrid_build"]))
