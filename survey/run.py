"""Make one array per variable from a few real granules of a collection, and check it.

    BENCH_HTTP=http://127.0.0.1:8765/ python run.py [collection ...]

For every variable: how the granules' layouts differ, what plain VirtualiZarr says, which
strategy the planner picks, and reads compared with the files (values, and the requests
and bytes that reach the source, against the same reads done granule by granule).
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import zarr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "bench"))

import switch_codec  # noqa: E402,F401
from build import DATA, HTTP, new_repo, open_repo  # noqa: E402
from compare_util import Counting, reset_source, source_stats  # noqa: E402
from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import LocalStore  # noqa: E402
from unify import Granule, plan, write  # noqa: E402
from virtualizarr import open_virtual_dataset  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402

warnings.filterwarnings("ignore")
assert HTTP, "set BENCH_HTTP and start bench/range_server.py"
zarr.config.set({"array.rectilinear_chunks": True, "array.sharding_coalesce_max_gap_bytes": 0})
PICKS = json.loads((Path(__file__).parent / "picks.json").read_text())
EXTRA = {
    "Daymet_PR": [f"daymet_v4_daily_pr_tmin_{y}.nc" for y in (2018, 2019, 2020, 2021)],
    "Daymet_NA": [f"daymet_v4_daily_na_dayl_{y}.nc" for y in (2019, 2020, 2021)],
}
# variable -> True when each granule is one step of a new leading axis (daily maps)
VARIABLES = {
    "MODISA_L3m_CHL": {"chlor_a": True},
    "OMPS_NPP_NMTO3_L3_DAILY": {"ColumnAmountO3": True},
    "GPM_2ADPR": {"FS/SLV/precipRateNearSurface": False, "FS/PRE/zFactorMeasured": False},
    "ATL08": {"gt1l/land_segments/dem_h": False, "gt1l/signal_photons/classed_pc_indx": False},
    "TEMPO_NO2_L2": {"support_data/gas_profile": False, "product/vertical_column_troposphere": False},
    "OCO2_L2_Lite_FP": {"xco2": False, "co2_profile_apriori": False},
    "Daymet_PR": {"tmin": False},
    "Daymet_NA": {"dayl": False},
}
registry = ObjectStoreRegistry({"file://": LocalStore()})
rng = np.random.default_rng(0)


def rename(p: str) -> str:
    return p.replace(f"file://{DATA}/", HTTP)


def survey(collection: str) -> dict:
    files = [DATA / f for f in (PICKS[collection]["files"] if collection in PICKS else EXTRA[collection])]
    report: dict = {"collection": collection, "granules": [f.name for f in files], "variables": {}}
    repo = new_repo(f"survey_{collection}")
    session = repo.writable_session("main")
    for var, stack in VARIABLES[collection].items():
        rec: dict = {"joined_along": "a new axis (one step per granule)" if stack else "the first axis"}
        report["variables"][var] = rec
        group, _, name = var.rpartition("/")
        safe = var.replace("/", "__")
        try:
            # Parse this variable alone: a group of variables without dimension scales
            # (GPM) does not form one consistent dataset.
            with h5py.File(files[0], "r") as h:
                others = [k for k, v in (h[group] if group else h).items() if k != name and isinstance(v, h5py.Dataset)]
            vdss = [open_virtual_dataset(url=str(f), registry=registry, parser=HDFParser(group=group or None, drop_variables=others),
                                         loadable_variables=[]) for f in files]
            arrays = [v[name].data for v in vdss]
        except Exception as err:
            rec["parse_error"] = f"{type(err).__name__}: {str(err)[:200]}"
            continue
        rec["per_granule"] = [{"shape": list(a.shape), "chunks": list(a.metadata.chunks),
                               "codecs": [c["name"] for c in a.metadata.to_dict()["codecs"]]} for a in arrays]
        try:
            dim = "granule" if stack else vdss[0][name].dims[0]
            xr.concat([v[[name]] for v in vdss], dim=dim, coords="minimal", compat="override", data_vars="all" if stack else "minimal")
            rec["plain_virtualizarr"] = "works"
        except Exception as err:
            rec["plain_virtualizarr"] = f"{type(err).__name__}: {str(err).splitlines()[0][:150]}"
        try:
            granules = [Granule.from_manifest_array(a, stack=stack, rename=rename) for a in arrays]
            p = plan(granules)
        except NotImplementedError as err:
            rec["planner"] = f"not handled: {err}"
            continue
        arr = write(session.store, safe, p, granules[0].dtype)
        for i, g in enumerate(granules):  # the same granules as separate arrays, for comparison
            write(session.store, f"_files/{safe}/{i}", plan([g]), g.dtype)
        rec.update(strategy=p.strategy, shape=list(p.shape), cases=len(p.cases), references=len(p.refs), source_chunks=p.source_chunks,
                   zarr_json_bytes=len(json.dumps(arr.metadata.to_dict(), default=str)), notes=p.notes)
        rec["_reads"] = (safe, stack, [g.shape[0] for g in granules], p)
    session.commit("survey", allow_empty=True)

    store = Counting(open_repo(f"survey_{collection}").readonly_session("main").store)
    handles = [h5py.File(f, "r") for f in files]
    for var, rec in report["variables"].items():
        if "_reads" not in rec:
            continue
        safe, stack, sizes, p = rec.pop("_reads")
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        shape = p.shape
        unified = zarr.open_array(store, path=safe, mode="r")
        parts = [zarr.open_array(store, path=f"_files/{safe}/{i}", mode="r") for i in range(len(sizes))]

        def per_granule(read, t: slice, rest: tuple) -> np.ndarray:
            out = []
            for i in range(len(sizes)):
                lo, hi = max(t.start, offsets[i]), min(t.stop, offsets[i + 1])
                if lo < hi:
                    out.append(read(i, slice(int(lo - offsets[i]), int(hi - offsets[i])), rest))
            return np.concatenate(out)

        def truth(t: slice, rest: tuple) -> np.ndarray:
            return per_granule(lambda i, s, r: (handles[i][var][()][None] if stack else handles[i][var])[(s, *r)] if stack
                               else handles[i][var][(s, *r)], t, rest)

        def window(max_len: int = 256) -> tuple:
            out = []
            for n in shape[1:]:
                w = min(n, max_len)
                start = int(rng.integers(0, n - w + 1))
                out.append(slice(start, start + w))
            return tuple(out)

        def point() -> tuple:
            return tuple(int(rng.integers(0, n)) for n in shape[1:])

        c0 = min(c[0] for c in p.notes["distinct_chunk_shapes"])
        selections = {}
        for k, b in enumerate(offsets[1:-1][:3]):
            b = int(b)
            selections[f"window across the boundary after granule {k + 1}"] = (slice(max(0, b - 2), min(shape[0], b + 2)), window())
        b = int(offsets[1])
        span = min(100 * c0, 20000)
        selections["series along the joined axis at one position"] = (slice(max(0, b - span), min(shape[0], b + span)), point())
        selections["whole first granule"] = (slice(0, sizes[0]), tuple(slice(0, n) for n in shape[1:]))
        rows = []
        for label, (t, rest) in selections.items():
            cells = (t.stop - t.start) * int(np.prod([(r.stop - r.start) if isinstance(r, slice) else 1 for r in rest]))
            if cells * unified.dtype.itemsize > 400e6:
                continue
            want = truth(t, rest)
            Counting.gets = 0
            reset_source()
            t0 = time.perf_counter()
            got = unified[(t, *rest)]
            seconds = time.perf_counter() - t0
            one = {"zarr_reads": Counting.gets, **source_stats()}
            reset_source()
            apart = per_granule(lambda i, s, r: parts[i][(s, *r)], t, rest)
            base = source_stats()
            rows.append({"read": label, "cells": cells, "equal_to_files": bool(np.array_equal(got, want, equal_nan=True)),
                         "per_granule_equal": bool(np.array_equal(apart, want, equal_nan=True)), "seconds": round(seconds, 3), **one,
                         "per_granule_source_requests": base["source_requests"], "per_granule_source_megabytes": base["source_megabytes"]})
        # scattered points
        pts = [(int(rng.integers(0, shape[0])), *point()) for _ in range(30)]
        want = np.array([truth(slice(q[0], q[0] + 1), q[1:])[0] for q in pts])
        reset_source()
        got = unified.vindex[tuple(np.array(c) for c in zip(*pts))]
        one = source_stats()
        rows.append({"read": "30 scattered points", "cells": 30, "equal_to_files": bool(np.array_equal(got, want, equal_nan=True)), **one})
        rec["reads"] = rows
        rec["all_equal"] = all(r["equal_to_files"] for r in rows)
        rec["source_requests"] = sum(r.get("source_requests", 0) for r in rows if "per_granule_source_requests" in r)
        rec["per_granule_source_requests"] = sum(r["per_granule_source_requests"] for r in rows if "per_granule_source_requests" in r)
        rec["source_megabytes"] = round(sum(r.get("source_megabytes", 0) for r in rows if "per_granule_source_requests" in r), 2)
        rec["per_granule_source_megabytes"] = round(sum(r["per_granule_source_megabytes"] for r in rows if "per_granule_source_requests" in r), 2)
    return report


if __name__ == "__main__":
    out = HERE / "results_http" / "survey"
    out.mkdir(parents=True, exist_ok=True)
    for collection in sys.argv[1:] or list(VARIABLES):
        try:
            rep = survey(collection)
        except Exception as err:
            rep = {"collection": collection, "error": f"{type(err).__name__}: {str(err)[:300]}"}
            import traceback

            traceback.print_exc()
        (out / f"{collection}.json").write_text(json.dumps(rep, indent=1, default=str))
        for var, rec in rep.get("variables", {}).items():
            brief = {k: rec.get(k) for k in ("strategy", "cases", "references", "source_chunks", "zarr_json_bytes", "all_equal",
                                             "source_requests", "per_granule_source_requests", "source_megabytes",
                                             "per_granule_source_megabytes", "plain_virtualizarr", "planner", "parse_error")}
            print(collection, var, json.dumps({k: v for k, v in brief.items() if v is not None})[:600], flush=True)
        if "error" in rep:
            print(collection, rep["error"], flush=True)
