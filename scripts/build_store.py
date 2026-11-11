"""Virtualize local Daymet yearly files of one region and variable into one Icechunk store,
then check the store against the files.

    uv run --project demo python scripts/build_store.py <region> <var> <year> [<year> ...]
"""

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import virtualizarr
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ConcatenatedManifestArray
from virtualizarr.parsers import HDFParser

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
region, var, *years = sys.argv[1:]
paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in years]
prefix = f"file://{DATA}/"
report: dict = {"region": region, "variable": var, "years": years}

# 1. virtualize each file, then concatenate, keeping incompatible runs as pieces
registry = ObjectStoreRegistry({"file://": LocalStore()})
t0 = time.time()
vdss = [
    open_virtual_dataset(
        url=str(p),
        registry=registry,
        parser=HDFParser(),
        loadable_variables=["time", "x", "y", "time_bnds", "yearday", "lambert_conformal_conic"],
    )
    for p in paths
]
report["per_file"] = [
    {
        "file": p.name,
        "chunks": list(v[var].data.metadata.chunks),
        "codecs": [c["name"] if isinstance(c, dict) else str(c) for c in v[var].data.metadata.to_dict()["codecs"]],
        "fill_value": repr(v[var].data.metadata.fill_value),
        "refs": v.vz.nrefs(),
    }
    for p, v in zip(paths, vdss)
]
try:
    xr.concat(vdss, dim="time", coords="minimal", compat="override", data_vars="minimal")
    report["plain_concat"] = "succeeded"
except Exception as err:
    report["plain_concat"] = f"{type(err).__name__}: {str(err)[:150]}"
with virtualizarr.set_options(mismatched_concat="pieces"):
    vds = xr.concat(vdss, dim="time", coords="minimal", compat="override", data_vars="minimal")
data = vds[var].data
report["virtualize_seconds"] = round(time.time() - t0, 1)
report["result"] = repr(data)
report["pieces"] = (
    [{"size": p.shape[0], "chunks": list(p.metadata.chunks)} for p in data.pieces]
    if isinstance(data, ConcatenatedManifestArray)
    else None
)

# 2. write to a local Icechunk repository (released Icechunk, unmodified)
repo_path = HERE / "stores" / f"{region}_{var}"
shutil.rmtree(repo_path, ignore_errors=True)
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(url_prefix=prefix, store=icechunk.local_filesystem_store(prefix))
)
repo = icechunk.Repository.create(
    storage=icechunk.local_filesystem_storage(str(repo_path)),
    config=config,
    authorize_virtual_chunk_access={prefix: None},
)
session = repo.writable_session("main")
t0 = time.time()
vds.vz.to_icechunk(session.store)
session.commit(f"Daymet {region} {var} {years[0]}-{years[-1]}")
report["write_seconds"] = round(time.time() - t0, 1)
report["icechunk"] = icechunk.__version__
report["store_bytes"] = sum(f.stat().st_size for f in repo_path.rglob("*") if f.is_file())
report["source_bytes"] = sum(p.stat().st_size for p in paths)

# 3. read it back as one array and compare with the files
store = repo.readonly_session("main").store
arr = zarr.open_array(store, path=var, mode="r")
report["array"] = {"shape": list(arr.shape), "codecs": [c["name"] for c in arr.metadata.to_dict()["codecs"]]}
actual = xr.open_zarr(store, consolidated=False, chunks=None, mask_and_scale=False)[var]
truth = [xr.open_dataset(p, engine="h5netcdf", mask_and_scale=False)[var] for p in paths]
nt = [t.sizes["time"] for t in truth]
offsets = np.concatenate([[0], np.cumsum(nt)])
ny, nx = truth[0].sizes["y"], truth[0].sizes["x"]
checks = []


def check(name: str, got: np.ndarray, want: np.ndarray, t0: float) -> None:
    checks.append({"check": name, "shape": list(got.shape), "equal": bool(np.array_equal(got, want, equal_nan=True)),
                   "seconds": round(time.time() - t0, 2), "valid_fraction": round(float((want != -9999).mean()), 3)})
    print(json.dumps(checks[-1]), flush=True)


rng = np.random.default_rng(0)
valid = np.argwhere(truth[0].isel(time=0).values != -9999)
# a window of days across every seam between files, centred on a cell with data
for i in range(1, len(paths)):
    s = offsets[i]
    cy, cx = valid[rng.integers(len(valid))]
    y0, x0 = int(np.clip(cy - 300, 0, max(ny - 600, 0))), int(np.clip(cx - 300, 0, max(nx - 600, 0)))
    win = (slice(y0, y0 + 600), slice(x0, x0 + 600))
    t0 = time.time()
    got = actual.isel(time=slice(s - 3, s + 12), y=win[0], x=win[1]).values
    want = np.concatenate([truth[i - 1].isel(time=slice(nt[i - 1] - 3, None), y=win[0], x=win[1]).values,
                           truth[i].isel(time=slice(0, 12), y=win[0], x=win[1]).values])
    check(f"15 days across the {years[i-1]}/{years[i]} seam, {got.shape[1]}x{got.shape[2]} window", got, want, t0)
# time series over every year at points with data
for y, x in valid[rng.choice(len(valid), size=3, replace=False)]:
    t0 = time.time()
    got = actual.isel(y=int(y), x=int(x)).values
    want = np.concatenate([t.isel(y=int(y), x=int(x)).values for t in truth])
    check(f"all {len(got)} days at point y={y}, x={x}", got, want, t0)
# whole array when it is small, else one full map per file
if arr.nbytes < 2e9:
    t0 = time.time()
    check("the whole array", actual.values, np.concatenate([t.values for t in truth]), t0)
else:
    for i in range(len(paths)):
        day = int(rng.integers(0, nt[i]))
        t0 = time.time()
        check(f"full map, day {day} of {years[i]}", actual.isel(time=offsets[i] + day).values, truth[i].isel(time=day).values, t0)
# scattered points (coordinate selection through zarr directly)
n = 200
tt, yy, xx = rng.integers(0, arr.shape[0], n), rng.integers(0, ny, n), rng.integers(0, nx, n)
if arr.nbytes < 2e9:
    t0 = time.time()
    full = np.concatenate([t.values for t in truth])
    check("200 scattered points (vindex)", arr.vindex[tt, yy, xx], full[tt, yy, xx], t0)

# xarray with dask and decoded values
with xr.open_zarr(store, consolidated=False, chunks={"time": 73, "y": 1000, "x": 1000}) as dask_ds:
    report["dask_chunks_example"] = str(dask_ds[var].data.chunksize)
    t0 = time.time()
    cy, cx = valid[rng.integers(len(valid))]
    y0, x0 = int(np.clip(cy - 100, 0, max(ny - 200, 0))), int(np.clip(cx - 100, 0, max(nx - 200, 0)))
    sel = dict(time=slice(offsets[-2] - 5, offsets[-2] + 5), y=slice(y0, y0 + 200), x=slice(x0, x0 + 200))
    got = dask_ds[var].isel(**sel).compute().values
    with xr.open_mfdataset([str(p) for p in paths], engine="h5netcdf", combine="nested", concat_dim="time", data_vars="minimal", coords="minimal", compat="override") as mf:
        want = mf[var].isel(**sel).compute().values
    checks.append({"check": "decoded values through dask vs open_mfdataset", "shape": list(got.shape),
                   "equal": bool(np.array_equal(got, want, equal_nan=True)), "seconds": round(time.time() - t0, 2),
                   "valid_fraction": round(float(np.isfinite(want).mean()), 3)})
    print(json.dumps(checks[-1]), flush=True)

report["checks"] = checks
report["all_equal"] = all(c["equal"] for c in checks)
out = HERE / "results" / f"{region}_{var}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=1))
