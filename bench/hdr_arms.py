"""Per-chunk headers on Icechunk: hdr_arms.py <region> <var> <first> <last> <emulated|native>

Two codecs read the header (`header_switch`, `conditional`). Who prepends it:
- emulated: a Python wrapper around the released Icechunk store (PrefixStore).
- native  : Icechunk itself, from a prefix stored with each virtual reference
            (needs the Icechunk branch that adds `prefix` to VirtualChunkSpec).
"""

import dataclasses
import json
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import icechunk
import numpy as np
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from build import RESULTS, STORES, remote, HERE, LOADABLE, new_repo, open_repo
from compare_util import Counting, reset_source, source_stats
from header_codecs import ConditionalCodec, HeaderSwitchCodec, PrefixStore, case_key
from switch_codec import rle_cases
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.array_api import _concat_manifests
from virtualizarr.parsers import HDFParser
from workloads import Workloads

warnings.filterwarnings("ignore")
region, var, first, last, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
W = Workloads(region, var, first, last)
name = f"bench_{region}_{var}"
registry = ObjectStoreRegistry({"file://": LocalStore()})
with ThreadPoolExecutor(8) as pool:
    vdss = list(pool.map(lambda p: open_virtual_dataset(url=str(p), registry=registry, parser=HDFParser(), loadable_variables=LOADABLE), W.paths))
vdss = [remote(v) for v in vdss]
arrays = [v[var].data for v in vdss]
with __import__("virtualizarr").set_options(mismatched_concat="pieces"):
    base_vds = xr.concat(vdss, dim="time", coords="minimal", compat="override", data_vars="minimal")
eras = rle_cases([list(a.metadata.to_dict()["codecs"]) for a in arrays], [a.manifest.shape_chunk_grid[0] for a in arrays])
manifest = _concat_manifests([a.manifest for a in arrays], axis=0)
shape = (sum(a.shape[0] for a in arrays), *arrays[0].shape[1:])

# header_switch: one case per era, named by an 8-byte key
cases = [{"codecs": e["codecs"], "parameters": []} for e in eras]
switch_runs = [(e["chunks"], case_key(c)) for e, c in zip(eras, cases)]
# conditional: the longest era's bytes-to-bytes codecs, and a bitmask per era
longest = max((e["codecs"] for e in eras), key=len)
head, wrapped = longest[0], longest[1:]
cond_runs = []
for e in eras:
    assert e["codecs"][0] == head and all(c in wrapped for c in e["codecs"][1:])
    mask = sum(1 << i for i, c in enumerate(wrapped) if c in e["codecs"][1:])
    cond_runs.append((e["chunks"], mask.to_bytes(1, "little")))
ARMS = {
    "header switch": ((HeaderSwitchCodec.from_cases(cases),), switch_runs),
    "conditional": ((zarr.registry.get_codec_class(head["name"]).from_dict(head), ConditionalCodec(codecs_json=json.dumps(wrapped))), cond_runs),
}

report: dict = {"mode": mode, "icechunk": icechunk.__version__, "rows": [], "arrays": {}}
for arm, (codecs, runs) in ARMS.items():
    repo_name = f"{name}_{mode}_{arm.replace(' ', '_')}"
    t0 = time.time()
    metadata = dataclasses.replace(arrays[0].metadata, shape=shape, codecs=codecs)
    vds = base_vds.copy()
    vds[var] = xr.Variable(base_vds[var].dims, ManifestArray(metadata=metadata, chunkmanifest=manifest), attrs=base_vds[var].attrs, encoding=base_vds[var].encoding)
    repo = new_repo(repo_name)
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store)
    if mode == "native":  # rewrite the references with their prefix
        bounds = np.cumsum([n for n, _ in runs])
        specs = []
        for idx in np.ndindex(manifest._paths.shape):
            if manifest._paths[idx]:
                header = runs[int(np.searchsorted(bounds, idx[0], side="right"))][1]
                specs.append(icechunk.VirtualChunkSpec(index=list(idx), location=str(manifest._paths[idx]), offset=int(manifest._offsets[idx]),
                                                       length=int(manifest._lengths[idx]), prefix=header))
        session.store.set_virtual_refs(var, specs, validate_containers=True)
    session.commit(arm)
    path = STORES / repo_name
    build_seconds = round(time.time() - t0, 1)

    def open_array():
        store = open_repo(repo_name).readonly_session("main").store
        if mode == "emulated":
            store = PrefixStore(store, var, runs)
        return zarr.open_array(Counting(store), path=var, mode="r")

    for workload, sel in W.selections.items():
        want = W.truth(*sel)
        times = []
        for _ in range(2 if "every day" in workload else 5):
            Counting.gets = Counting.nbytes = 0; reset_source()
            t0 = time.perf_counter()
            got = open_array()[sel]
            times.append(time.perf_counter() - t0)
        report["rows"].append({"workload": workload, "approach": f"{mode}: {arm}", "shape": list(got.shape),
                               "equal_to_files": bool(np.array_equal(got, want)), "seconds_median": round(statistics.median(times), 3),
                               "seconds_min": round(min(times), 3), "store_gets": Counting.gets, "store_megabytes": round(Counting.nbytes / 1e6, 2), **source_stats()})
        print(json.dumps(report["rows"][-1]), flush=True)
    arr = open_array()
    report["arrays"][arm] = {
        "zarr_json_bytes": len(json.dumps(arr.metadata.to_dict(), default=str)), "codecs": [c["name"] for c in arr.metadata.to_dict()["codecs"]],
        "zarr_chunks": list(arr.chunks), "store_bytes": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
        "build_seconds": build_seconds, "header_bytes_per_chunk": len(runs[0][1]),
    }
(RESULTS / f"{name}_{mode}.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report["arrays"], indent=1))
