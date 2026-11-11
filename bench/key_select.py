"""The `switch` codec finding a chunk's case by position, and by a key in the reference.

    BENCH_HTTP=http://127.0.0.1:8765/ python key_select.py <region> <var> <first_year> <last_year>

Both arrays are planned and written by VirtualiZarr (`switch_select` option) into the
modified Icechunk, from the same parsed files. The same reads run against each: values
against the files, time, and the requests and bytes that reach the source.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import virtualizarr
from build import DATA, HTTP, LOADABLE, RESULTS, STORES, new_repo, open_repo, remote
from compare_util import reset_source, source_stats
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from workloads import Workloads

warnings.filterwarnings("ignore")
assert HTTP, "set BENCH_HTTP and start bench/range_server.py"
zarr.config.set({"array.rectilinear_chunks": True})
REPEATS = 3


def main() -> None:
    region, var, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in range(first, last + 1)]
    name = f"bench_{region}_{var}"
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    with ThreadPoolExecutor(8) as pool:
        vdss = list(pool.map(
            lambda p: open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=LOADABLE),
            paths,
        ))
    vdss = [remote(v) for v in vdss]
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="time", coords="minimal", compat="override", data_vars="minimal")

    report: dict = {"files": len(paths), "arms": {}}
    for select in ("position", "key"):
        repo = new_repo(f"{name}_select_{select}")
        session = repo.writable_session("main")
        t0 = time.time()
        with virtualizarr.set_options(switch_select=select):
            vds.vz.to_icechunk(session.store)
        session.commit(select)
        store = STORES / f"{name}_select_{select}"
        array = zarr.open_array(repo.readonly_session("main").store, path=var, mode="r")
        meta = array.metadata.to_dict()
        configuration = meta["codecs"][0]["configuration"]
        report["arms"][select] = {
            "write_seconds": round(time.time() - t0, 1),
            "strategy": "shards" if configuration["cases"][0]["codecs"][0]["name"] == "sharding_indexed" else "switch",
            "cases": len(configuration["cases"]),
            "zarr_chunks": int(np.prod([len(d) if isinstance(d, list) else 1 for d in meta["chunk_grid"]["configuration"]["chunk_shapes"]])),
            "zarr_json_bytes": len(json.dumps(meta, default=str)),
            "manifest_bytes": sum(f.stat().st_size for f in (store / "manifests").rglob("*") if f.is_file()),
            "store_bytes": sum(f.stat().st_size for f in store.rglob("*") if f.is_file()),
            "reads": {},
        }

    work = Workloads(region, var, first, last)
    for label, sel in work.selections.items():
        want = work.truth(*sel)
        for select in ("position", "key"):
            seconds = []
            for _ in range(REPEATS):
                # a new session each time: nothing is cached between repeats
                array = zarr.open_array(open_repo(f"{name}_select_{select}").readonly_session("main").store, path=var, mode="r")
                reset_source()
                t0 = time.perf_counter()
                got = array[sel]
                seconds.append(time.perf_counter() - t0)
                stats = source_stats()
            report["arms"][select]["reads"][label] = {
                "equal_to_files": bool(np.array_equal(got, want, equal_nan=True)),
                "seconds": round(float(np.median(seconds)), 3), **stats,
            }
            print(select, label, report["arms"][select]["reads"][label], flush=True)

    (RESULTS / f"{name}_select.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "reads"} for k, v in report["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
