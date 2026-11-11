"""Build the stores the comparison reads: build.py <region> <var> <first_year> <last_year>

- `<name>_concat`: VirtualiZarr pieces + the array_concat codec (zarr-python branch).
- `<name>_switch`: one ordinary array whose `switch` codec picks a codec chain per chunk.
"""

import dataclasses
import os
import json
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import icechunk
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import virtualizarr
from switch_codec import SwitchCodec, rle_cases
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.array_api import _concat_manifests
from virtualizarr.parsers import HDFParser

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
# BENCH_HTTP=http://127.0.0.1:<port>/ points the virtual references at range_server.py,
# which counts the requests that reach the source files.
HTTP = os.environ.get("BENCH_HTTP")
PREFIX = HTTP or f"file://{DATA}/"
STORES = HERE / ("stores_http" if HTTP else "stores")
RESULTS = HERE / ("results_http" if HTTP else "results")
STORES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def chunk_store():
    if HTTP:
        return icechunk.http_store(opts={"allow_http": "true"})
    return icechunk.local_filesystem_store(PREFIX)


def remote(vds):
    """Point a virtual dataset's references at the HTTP server, when one is in use."""
    return vds.vz.rename_paths(lambda p: p.replace(f"file://{DATA}/", HTTP)) if HTTP else vds
LOADABLE = ["time", "x", "y", "time_bnds", "yearday", "lambert_conformal_conic"]


def new_repo(name: str) -> icechunk.Repository:
    path = STORES / name
    shutil.rmtree(path, ignore_errors=True)
    config = icechunk.RepositoryConfig.default()
    if os.environ.get("BENCH_OBJECT_STORE_SETTINGS", "1") == "1":
        # A local-filesystem repository defaults to 4 KiB requests, five at a time, and
        # virtual chunk fetches inherit that. Use what an S3 repository would.
        config.storage = icechunk.StorageSettings(
            concurrency=icechunk.StorageConcurrencySettings(max_concurrent_requests_for_object=18, ideal_concurrent_request_size=12 << 20)
        )
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(url_prefix=PREFIX, store=chunk_store())
    )
    return icechunk.Repository.create(
        storage=icechunk.local_filesystem_storage(str(path)),
        config=config,
        authorize_virtual_chunk_access={PREFIX: None},
    )


def open_repo(name: str) -> icechunk.Repository:
    return icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(STORES / name)),
        authorize_virtual_chunk_access={PREFIX: None},
    )


def main() -> None:
    region, var, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    years = list(range(first, last + 1))
    paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in years]
    name = f"bench_{region}_{var}"
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    report: dict = {"files": len(paths), "source_bytes": sum(p.stat().st_size for p in paths)}

    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        vdss = list(pool.map(
            lambda p: open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=LOADABLE),
            paths,
        ))
    report["parse_seconds"] = round(time.time() - t0, 1)
    vdss = [remote(v) for v in vdss]

    # --- concat: pieces
    t0 = time.time()
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="time", coords="minimal", compat="override", data_vars="minimal")
    repo = new_repo(f"{name}_concat")
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store)
    session.commit("concat")
    data = vds[var].data
    report["concat"] = {"build_seconds": round(time.time() - t0, 1), "pieces": [p.shape[0] for p in getattr(data, "pieces", [data])]}

    # --- switch: one array, codec chain chosen per chunk
    t0 = time.time()
    arrays = [v[var].data for v in vdss]
    chunk0 = arrays[0].metadata.chunks
    if any(a.metadata.chunks != chunk0 for a in arrays):
        report["switch"] = {"feasible": False, "why": "chunk shapes differ between files: "
                            + str(sorted({tuple(a.metadata.chunks) for a in arrays}))}
    else:
        codecs = [list(a.metadata.to_dict()["codecs"]) for a in arrays]
        cases = rle_cases(codecs, [a.manifest.shape_chunk_grid[0] for a in arrays])
        metadata = dataclasses.replace(
            arrays[0].metadata,
            shape=(sum(a.shape[0] for a in arrays), *arrays[0].shape[1:]),
            codecs=(SwitchCodec(axis=0, cases=cases),),
        )
        marr = ManifestArray(metadata=metadata, chunkmanifest=_concat_manifests([a.manifest for a in arrays], axis=0))
        vds_switch = vds.copy()
        vds_switch[var] = xr.Variable(vds[var].dims, marr, attrs=vds[var].attrs, encoding=vds[var].encoding)
        repo = new_repo(f"{name}_switch")
        session = repo.writable_session("main")
        vds_switch.vz.to_icechunk(session.store)
        session.commit("switch")
        report["switch"] = {"feasible": True, "build_seconds": round(time.time() - t0, 1),
                            "cases": [{"chunks": c["chunks"], "codecs": [x["name"] for x in c["codecs"]]} for c in cases]}

    for kind in ("concat", "switch"):
        path = STORES / f"{name}_{kind}"
        if path.exists():
            report[kind]["store_bytes"] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    out = RESULTS / f"{name}_build.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
