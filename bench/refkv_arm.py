# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "refkv[virtualizarr]>=0.1.0",
#     "obstore>=0.11.1",
#     "virtualizarr>=2.7.3",
#     "zarr>=3.4.0",
#     "cast-value-rs>=0.4.2",
#     "xarray>=2026.9.0",
#     "h5py>=3.16.0",
#     "h5netcdf>=1.8.1",
#     "matplotlib>=3.10",
#     "dask",
# ]
# ///
"""The refkv approach on the same files: refkv_arm.py <region> <var> <first> <last>

Uses the `switch` and `crop` codecs and the `combine` function of d-v-b/refkv PR #73
(vendor/refkv_switch_demo.py, unmodified): each chunk's reference is a literal header
naming its codec chain, followed by the byte range in the file. Released zarr-python,
released VirtualiZarr, released refkv; no Icechunk.
"""

import importlib.util
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import virtualizarr as vz
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr.parsers import HDFParser
from zarr.storage import WrapperStore

import refkv
import refkv.zarr

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
spec = importlib.util.spec_from_file_location("refkv_switch_demo", Path(__file__).parent / "vendor" / "refkv_switch_demo.py")
demo = importlib.util.module_from_spec(spec)
sys.modules["refkv_switch_demo"] = demo
spec.loader.exec_module(demo)

region, var, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ARM = sys.argv[5] if len(sys.argv) > 5 else "switch"  # or "conditional"
years = list(range(first, last + 1))
paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in years]
name = f"bench_{region}_{var}"
report: dict = {"versions": {"zarr": zarr.__version__, "virtualizarr": vz.__version__, "refkv": refkv.__version__}, "years": [first, last]}


class Counting(WrapperStore):
    gets = 0
    nbytes = 0

    async def get(self, key, prototype, byte_range=None):
        out = await self._store.get(key, prototype, byte_range)
        type(self).gets += 1
        type(self).nbytes += 0 if out is None else len(out)
        return out


t0 = time.time()
registry = ObjectStoreRegistry({f"file://{DATA}": LocalStore(prefix=DATA)})
vdss = [vz.open_virtual_dataset(f"file://{p}", registry=registry, parser=HDFParser(), loadable_variables=["time", "x", "y"]) for p in paths]
report["parse_seconds"] = round(time.time() - t0, 1)
SUFFIX = "refkv" if ARM == "switch" else f"refkv_{ARM}"
archive = HERE / "stores" / f"{name}_{first}_{last}_{SUFFIX}.vzip"
archive.parent.mkdir(exist_ok=True)
try:
    t0 = time.time()
    if ARM == "conditional":
        import header_codecs  # noqa: F401  (registers `conditional`)
        from refkv.pb import Concat, Range, Source

        mapping, sources = demo.build(vdss, name=var, dim="time")  # coordinates and group, as the switch arm
        chains = [[c.to_dict() for c in v[var].data.metadata.codecs] for v in vdss]
        longest = max(chains, key=len)
        head, wrapped = longest[0], longest[1:]
        meta = json.loads(mapping[f"{var}/zarr.json"])
        meta["codecs"] = [head, {"name": "conditional", "configuration": {"codecs": wrapped, "header_bits": 8}}]
        meta["fill_value"] = float(vdss[0][var].data.metadata.fill_value)
        mapping[f"{var}/zarr.json"] = json.dumps(meta).encode()
        base = 0
        for v, chain in zip(vdss, chains):
            if chain[0] != head or any(c not in wrapped for c in chain[1:]):
                raise ValueError("a file's codecs are not a subset of the longest chain")
            mask = sum(1 << i for i, c in enumerate(wrapped) if c in chain[1:]).to_bytes(1, "little")
            ma = v[var].data
            if ma.shape[0] % ma.metadata.chunks[0]:
                raise ValueError("conditional cannot crop a file's short last chunk")
            for idx, e in ma.manifest.dict().items():
                i, *js = (int(x) for x in idx.split("."))
                src = next(k for k, s_ in enumerate(sources) if s_.url == e["path"])
                mapping[f"{var}/c/" + "/".join(map(str, (base + i, *js)))] = Concat(
                    (Range(data=mask), Range(source=src, offset=e["offset"], length=e["length"])))
            base += ma.manifest.shape_chunk_grid[0]
        store = refkv.ReferenceStore(mapping, sources)
    else:
        store = demo.combine(vdss, name=var, dim="time")
    store.save(str(archive))
    report["build_seconds"] = round(time.time() - t0, 1)
    report["feasible"] = True
except Exception as err:
    report["feasible"] = False
    report["why"] = f"{type(err).__name__}: {err}"
    (HERE / "results" / f"{name}_{first}_{last}_{SUFFIX}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(0)
report["store_bytes"] = archive.stat().st_size
meta = json.loads(store.get(f"{var}/zarr.json"))
report["zarr_json_bytes"] = len(store.get(f"{var}/zarr.json"))
report["codecs"] = meta["codecs"] if ARM == "conditional" else [[c["name"] for c in case["codecs"]] for case in meta["codecs"][0]["configuration"]["cases"].values()]
report["chunk_grid"] = meta["chunk_grid"]["name"]

files = [h5py.File(p, "r") for p in paths]
nt = [f[var].shape[0] for f in files]
offsets = np.concatenate([[0], np.cumsum(nt)])
NT, NY, NX = int(offsets[-1]), *files[0][var].shape[1:]


def truth(t, y, x):
    parts = []
    for i, f in enumerate(files):
        lo, hi = max(t.start, offsets[i]), min(t.stop, offsets[i + 1])
        if lo < hi:
            parts.append(f[var][lo - offsets[i] : hi - offsets[i], y, x])
    return np.concatenate(parts)


seam = int(offsets[years.index(2020)]) if 2020 in years and first < 2020 else int(offsets[len(offsets) // 2])
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
if NY * NX > 5e6:  # North America: keep the reads small
    WORKLOADS = {
        "box: 30 days x 600x600 cells, across the file boundary": (slice(seam - 15, seam + 15), slice(by, by + 600), slice(bx, bx + 600)),
        "series: every day at one cell": (slice(0, NT), py, px),
    }

rows = []
with zarr.config.set({"array.rectilinear_chunks": True}):
    for workload, sel in WORKLOADS.items():
        want = truth(*sel)
        times = []
        for _ in range(2 if "every day" in workload else 5):
            Counting.gets = Counting.nbytes = 0
            t0 = time.perf_counter()
            arr = zarr.open_array(Counting(refkv.zarr.store(refkv.open(str(archive)))), path=var, mode="r")
            got = arr[sel]
            times.append(time.perf_counter() - t0)
        rows.append({"workload": workload, "approach": f"refkv {ARM}", "shape": list(got.shape),
                     "equal_to_files": bool(np.array_equal(got, want)), "seconds_median": round(statistics.median(times), 3),
                     "seconds_min": round(min(times), 3), "store_gets": Counting.gets, "store_megabytes": round(Counting.nbytes / 1e6, 2)})
        print(json.dumps(rows[-1]), flush=True)
    arr = zarr.open_array(refkv.zarr.store(refkv.open(str(archive))), path=var, mode="r")
    report["zarr_chunk_grid_cells"] = int(np.prod(arr.cdata_shape)) if hasattr(arr, "cdata_shape") else None
    try:
        ds = xr.open_zarr(refkv.zarr.store(refkv.open(str(archive))), consolidated=False, chunks={})
        report["default_dask_chunks"] = int(ds[var].data.npartitions)
    except Exception as err:
        report["default_dask_chunks"] = f"{type(err).__name__}: {str(err)[:120]}"
report["rows"] = rows
(HERE / "results" / f"{name}_{first}_{last}_{SUFFIX}.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
