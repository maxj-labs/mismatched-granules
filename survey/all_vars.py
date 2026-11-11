"""Every array variable of the sampled granules, joined into one array each.

    python all_vars.py [collection ...]

Uses the planner that now lives in VirtualiZarr (`plan_joined_array`). For each variable:
is a plain concat already possible, which strategy the planner picks, and whether reads
of the joined array equal the files.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import h5py
import icechunk
import numpy as np
import zarr
from zarr.codecs import SwitchCodec

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "bench"))

from build import DATA, new_repo  # noqa: E402
from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import LocalStore  # noqa: E402
from virtualizarr.manifests import ManifestArray  # noqa: E402
from virtualizarr.manifests.array_api import stack as _stack  # noqa: E402
from virtualizarr.manifests.utils import copy_and_replace_metadata  # noqa: E402


def stack(arrays, axis=0):
    """Give arrays a new leading axis. Their dimension names are dropped first: the
    files do not name the new axis."""
    bare = [ManifestArray(metadata=copy_and_replace_metadata(a.metadata, new_dimension_names=None), chunkmanifest=a.manifest) for a in arrays]
    return _stack(bare, axis=axis)

from virtualizarr.manifests.plan import plan_joined_array  # noqa: E402
from virtualizarr.manifests.utils import manifest_chunk_shape  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True})
PICKS = json.loads((Path(__file__).parent / "picks.json").read_text())
STACKED = {"MODISA_L3m_CHL", "OMPS_NPP_NMTO3_L3_DAILY"}  # one map per granule, no joined axis in the file
registry = ObjectStoreRegistry({"file://": LocalStore()})
rng = np.random.default_rng(0)


def walk(group, prefix=""):
    for name, arr in group.arrays.items():
        yield f"{prefix}{name}", arr
    for name, sub in group.groups.items():
        yield from walk(sub, f"{prefix}{name}/")


def survey(collection: str) -> dict:
    files = [DATA / f for f in PICKS[collection]["files"]]
    stacked = collection in STACKED
    t0 = time.time()
    trees = [dict(walk(HDFParser()(url=f"file://{f}", registry=registry)._group)) for f in files]
    handles = [h5py.File(f, "r") for f in files]
    names = sorted(set().union(*trees))
    repo = new_repo(f"allvars_{collection}")
    session = repo.writable_session("main")
    store = session.store
    rows = []
    for name in names:
        row: dict = {"variable": name}
        rows.append(row)
        present = [name in t for t in trees]
        row["present_in"] = f"{sum(present)} of {len(files)}"
        if sum(present) * 2 <= len(files):
            # e.g. provenance groups named after each granule's own input files
            row["outcome"] = "name is particular to a granule"
            continue
        have = [i for i, ok in enumerate(present) if ok]
        ref = trees[have[0]][name]
        if stacked:
            # A granule without the variable is a step that stores nothing: it reads
            # as the fill value.
            arrays = [t[name] if name in t else ref.with_fill_value_only(ref.metadata.fill_value) for t in trees]
            have = list(range(len(files)))
        else:
            # Along a ragged axis (photons, segments) a granule without the variable
            # has none of them: it adds nothing to the joined array.
            arrays = [trees[i][name] for i in have]
        row["absent_handled_as"] = None if all(present) else ("fill value" if stacked else "zero length")
        if arrays[0].ndim == 0 or any(a.dtype.kind not in "iufb" for a in arrays):
            row["outcome"] = "not a numeric array"
            continue
        if any(0 in a.shape for a in arrays):
            row["outcome"] = "empty in a granule"
            continue
        pieces = [stack([a], axis=0) for a in arrays] if stacked else arrays
        chunks = {tuple(manifest_chunk_shape(a.metadata)) for a in pieces}
        cut = sum(a.shape[0] % manifest_chunk_shape(a.metadata)[0] != 0 for a in pieces[:-1])
        missing = int(sum((np.asarray(a.manifest._paths) == "").sum() for a in pieces))
        fills = {np.asarray(a.metadata.fill_value).tobytes() for a in pieces}
        row.update(
            chunk_shapes=len(chunks), codec_chains=len({str(a.metadata.to_dict()["codecs"]) for a in pieces}),
            data_types=sorted({str(a.dtype) for a in pieces}), granules_cut=cut, unstored_chunks=missing, fill_values=len(fills),
        )
        row["plain_concat_possible"] = len(chunks) == 1 and row["codec_chains"] == 1 and len(row["data_types"]) == 1 and cut == 0 and (len(fills) == 1 or missing == 0)
        try:
            plan = plan_joined_array(pieces, axis=0)
        except Exception as err:
            row["outcome"] = f"not planned: {type(err).__name__}: {str(err)[:80]}"
            continue
        row.update(strategy=plan.strategy, cases=len(plan.cases), own_fill_cases=plan.notes["pieces_with_their_own_fill"])
        path = name.replace("/", "__")
        try:
            arr = zarr.create_array(store, name=path, shape=plan.shape, chunks=plan.chunk_grid, dtype=plan.dtype, fill_value=plan.fill_value,
                                    serializer=SwitchCodec(axis=0, cases=plan.cases), compressors=None, filters=None)
            failed = store.set_virtual_refs(path, [icechunk.VirtualChunkSpec(**r) for r in plan.refs], validate_containers=True)
            assert not failed
        except Exception as err:
            row["outcome"] = f"not written: {type(err).__name__}: {str(err)[:80]}"
            continue
        # reads against the files
        def truth(k, sel):
            i = have[k]
            if name not in handles[i]:  # an absent step: all fill
                return np.full((1, *ref.shape), plan.fill_value, dtype=plan.dtype)[sel]
            d = handles[i][name]
            return (d[()][None] if stacked else d)[sel].astype(plan.dtype)

        sizes = [p.shape[0] for p in pieces]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        try:
            if arr.nbytes <= 64e6:
                want = np.concatenate([truth(k, (slice(None),)) for k in range(len(pieces))])
                ok = np.array_equal(arr[...], want, equal_nan=True)
            else:
                ok = True
                rest = tuple(slice(int(s), int(s) + min(n, 64)) for n in plan.shape[1:] for s in [rng.integers(0, n - min(n, 64) + 1)])
                for k in range(1, len(pieces)):  # a window across each boundary
                    b = int(offsets[k])
                    lo, hi = max(b - 3, int(offsets[k - 1])), min(b + 3, int(offsets[k + 1]))
                    want = np.concatenate([truth(k - 1, (slice(lo - int(offsets[k - 1]), None), *rest)), truth(k, (slice(0, hi - b), *rest))])
                    ok &= np.array_equal(arr[(slice(lo, hi), *rest)], want, equal_nan=True)
                for _ in range(10):  # scattered cells
                    t = int(rng.integers(0, plan.shape[0]))
                    i = int(np.searchsorted(offsets, t, side="right") - 1)
                    cell = tuple(int(rng.integers(0, n)) for n in plan.shape[1:])
                    ok &= np.array_equal(arr[(t, *cell)], truth(i, (t - int(offsets[i]), *cell)), equal_nan=True)
            row["outcome"] = "equal to files" if ok else "DIFFERS FROM FILES"
        except Exception as err:
            row["outcome"] = f"read failed: {type(err).__name__}: {str(err)[:100]}"
    session.commit("all variables", allow_empty=True)
    summary = {
        "collection": collection, "granules": len(files), "variables": len(names), "seconds": round(time.time() - t0, 1),
        "outcomes": dict(Counter(r["outcome"].split(":")[0] for r in rows)),
        "joined": sum(r["outcome"] == "equal to files" for r in rows),
        "of_which_plain_concat_was_impossible": sum(r["outcome"] == "equal to files" and not r["plain_concat_possible"] for r in rows),
        "strategies": dict(Counter(r["strategy"] for r in rows if "strategy" in r)),
        "joined_despite_absent_granules": sum(r["outcome"] == "equal to files" and r.get("absent_handled_as") is not None for r in rows),
        "variables_with_unstored_chunks": sum(r.get("unstored_chunks", 0) > 0 for r in rows),
        "variables_given_their_own_fill": sum(r.get("own_fill_cases", 0) > 0 for r in rows),
        "variables_with_several_data_types": sum(len(r.get("data_types", [])) > 1 for r in rows),
        "max_cases": max((r.get("cases", 0) for r in rows), default=0),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    out = HERE / "results" / "all_vars"
    out.mkdir(parents=True, exist_ok=True)
    for collection in sys.argv[1:] or list(PICKS):
        rep = survey(collection)
        (out / f"{collection}.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep["summary"]), flush=True)
        for r in rep["rows"]:
            if r["outcome"] not in ("equal to files", "absent from some granules", "not a numeric array", "empty in a granule"):
                print("   ", r["variable"], "|", r["outcome"], flush=True)
