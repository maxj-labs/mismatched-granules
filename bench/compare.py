"""Read the same selections through each approach and compare: compare.py <region> <var> <first> <last>

Approaches
- concat   : one array, array_concat codec over pieces (zarr-python branch + VirtualiZarr branch)
- switch   : one ordinary array, per-chunk switch codec (plugin, released zarr-python works)
- separate : the status quo workaround: the pieces as separate arrays, joined by hand in user code
- files    : no virtual store: xarray.open_mfdataset on the netCDF files (h5netcdf + dask)

Every result is compared with values read straight from the files with h5py.
Timings are on local disk with a warm page cache: they show overhead, not network cost.
"""

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import zarr
from zarr.storage import WrapperStore

import switch_codec  # noqa: F401  (registers the codec)
from compare_util import reset_source, source_stats
from build import RESULTS, STORES, remote, DATA, HERE, open_repo

warnings.filterwarnings("ignore")
region, var, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
years = list(range(first, last + 1))
paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in years]
name = f"bench_{region}_{var}"
build = json.loads((RESULTS / f"{name}_build.json").read_text())
REPEATS = 5


class Counting(WrapperStore):
    """Counts the store reads zarr makes: how many, and how many bytes came back."""

    gets = 0
    nbytes = 0

    async def get(self, key, prototype, byte_range=None):
        out = await self._store.get(key, prototype, byte_range)
        type(self).gets += 1
        type(self).nbytes += 0 if out is None else len(out)
        return out


def store_for(kind: str) -> Counting:
    Counting.gets = Counting.nbytes = 0
    reset_source()
    return Counting(open_repo(f"{name}_{kind}").readonly_session("main").store)


# ---- truth, straight from the files
files = [h5py.File(p, "r") for p in paths]
nt = [f[var].shape[0] for f in files]
offsets = np.concatenate([[0], np.cumsum(nt)])
NT, NY, NX = int(offsets[-1]), *files[0][var].shape[1:]


def truth(t: slice, y, x) -> np.ndarray:
    parts = []
    for i, f in enumerate(files):
        lo, hi = max(t.start, offsets[i]), min(t.stop, offsets[i + 1])
        if lo < hi:
            parts.append(f[var][lo - offsets[i] : hi - offsets[i], y, x])
    return np.concatenate(parts)


# ---- the selections
seam = int(offsets[years.index(2020)]) if 2020 in years and first < 2020 else NT // 2
valid = np.argwhere(files[0][var][0] != -9999)
py, px = (int(v) for v in valid[len(valid) // 2])
by, bx = int(np.clip(py - 25, 0, NY - 50)), int(np.clip(px - 25, 0, NX - 50))
decade = slice(max(0, seam - 5 * 365), min(NT, seam + 5 * 365))
WORKLOADS = {
    "map: 2 days, whole grid, across the 2020 change": (slice(seam - 1, seam + 1), slice(None), slice(None)),
    "box: 30 days x 50x50 cells, across the change": (slice(seam - 15, seam + 15), slice(by, by + 50), slice(bx, bx + 50)),
    "series: 10 years at one cell, across the change": (decade, py, px),
    "series: every day at one cell": (slice(0, NT), py, px),
}
TRUTH = {k: truth(*sel) for k, sel in WORKLOADS.items()}


# ---- readers: each opens from scratch and returns the selection
def read_concat(sel):
    return zarr.open_array(store_for("concat"), path=var, mode="r")[sel]


def read_switch(sel):
    return zarr.open_array(store_for("switch"), path=var, mode="r")[sel]


def read_separate(sel):
    """What a user writes today: open each piece and join the parts by hand."""
    store = store_for("concat")
    group = zarr.open_group(store, path=f"_pieces/{var}", mode="r")
    t, y, x = sel
    parts, start = [], 0
    for i in range(len(build["concat"]["pieces"])):
        arr = group[str(i)]
        lo, hi = max(t.start, start), min(t.stop, start + arr.shape[0])
        if lo < hi:
            parts.append(arr[lo - start : hi - start, y, x])
        start += arr.shape[0]
    return np.concatenate(parts)


def read_files(sel):
    Counting.gets = Counting.nbytes = 0; reset_source()
    with xr.open_mfdataset([str(p) for p in paths], engine="h5netcdf", combine="nested", concat_dim="time",
                           data_vars="minimal", coords="minimal", compat="override", mask_and_scale=False) as ds:
        t, y, x = sel
        return ds[var].isel(time=t, y=y, x=x).values


READERS = {"concat": read_concat, "switch": read_switch, "separate": read_separate, "files": read_files}
if not build["switch"].get("feasible"):
    del READERS["switch"]

rows = []
for workload, sel in WORKLOADS.items():
    for approach, reader in READERS.items():
        repeats = 2 if (approach == "files" or "every day" in workload) else REPEATS
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            got = reader(sel)
            times.append(time.perf_counter() - t0)
        rows.append({
            "workload": workload, "approach": approach, "shape": list(got.shape),
            "equal_to_files": bool(np.array_equal(got, TRUTH[workload])),
            "seconds_median": round(statistics.median(times), 3), "seconds_min": round(min(times), 3),
            "store_gets": Counting.gets if approach != "files" else None,
            "store_megabytes": round(Counting.nbytes / 1e6, 2) if approach != "files" else None,
            **(source_stats() if approach != "files" else {}),
        })
        print(json.dumps(rows[-1]), flush=True)

# ---- what each array looks like to a user
shape = {}
for kind in ("concat", "switch"):
    if kind == "switch" and "switch" not in READERS:
        continue
    store = open_repo(f"{name}_{kind}").readonly_session("main").store
    arr = zarr.open_array(store, path=var, mode="r")
    meta = zarr.core.sync.sync(store.get(f"{var}/zarr.json", zarr.core.buffer.default_buffer_prototype()))
    t0 = time.perf_counter()
    ds = xr.open_zarr(store, consolidated=False)
    shape[kind] = {
        "zarr_json_bytes": len(meta), "zarr_chunks": list(arr.chunks), "codecs": [c["name"] for c in arr.metadata.to_dict()["codecs"]],
        "xarray_open_seconds": round(time.perf_counter() - t0, 3),
        "default_dask_chunk_megabytes": round(np.prod(ds[var].data.chunksize) * 4 / 1e6, 1),
        "default_dask_chunks": int(ds[var].data.npartitions),
        "nodes_in_store": 1 + sum(1 for _ in zarr.open_group(store, mode="r").members(max_depth=None)),
        "store_bytes": build[kind]["store_bytes"],
    }
print(json.dumps(shape, indent=1))
out = RESULTS / f"{name}_compare.json"
out.write_text(json.dumps({"years": [first, last], "shape": [NT, NY, NX], "seam_index": seam, "build": build, "arrays": shape, "rows": rows}, indent=1))
