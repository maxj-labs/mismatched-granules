"""Which options can be stored without Icechunk? Tested with Kerchunk-style references.

    python without_icechunk.py

A Kerchunk reference is either inline bytes or one (url, offset, length). An option fits
when every chunk of its array is one of those. Where it fits, the array is read through
fsspec's reference filesystem and compared with the files.
"""

import json

import fsspec
import numpy as np
import zarr
from zarr.storage import FsspecStore

import arms
import cases as cases_module
import xcodecs  # noqa: F401
from common import RESULTS, NotExpressible, pieces_of, same

PLANNED = {
    "switch_position": lambda plan: zarr.codecs.SwitchCodec(axis=0, cases=plan.cases),
    "switch_key": lambda plan: zarr.codecs.SwitchCodec(select="key", key_bytes=plan.key_bytes, cases=plan.cases),
}
CASES = list(cases_module.build())


def kerchunk(plan, serializer) -> dict:
    """The planned array as a Kerchunk reference set, or NotExpressible."""
    refs: dict = {}
    for r in plan.refs:
        if r.get("prefix") or r.get("more_ranges"):
            raise NotExpressible("a chunk is a literal followed by one or more byte ranges; a Kerchunk reference is one or the other")
        refs["v/c/" + "/".join(str(i) for i in r["index"])] = [r["location"], r["offset"], r["length"]]
    holder = zarr.create_array(zarr.storage.MemoryStore(), name="v", shape=plan.shape, chunks=plan.chunk_grid, dtype=plan.dtype, fill_value=plan.fill_value,
                               serializer=serializer, compressors=None, filters=None)
    refs["v/zarr.json"] = json.dumps(holder.metadata.to_dict(), default=str)
    refs["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group"})
    return {"version": 1, "refs": refs}


if __name__ == "__main__":
    all_cases = cases_module.build()
    report: dict = {"reference_format": "Kerchunk JSON, read with fsspec's reference filesystem", "cells": {}}
    for name in CASES:
        case = all_cases[name]
        for arm, codec in PLANNED.items():
            cell: dict = {}
            report["cells"][f"{arm}|{name}"] = cell
            try:
                plan = arms._plan(pieces_of(case), select="key" if arm == "switch_key" else "position")
                refs = kerchunk(plan, codec(plan))
            except NotExpressible as err:
                cell.update(outcome="not expressible", why=str(err))
                continue
            fs = fsspec.filesystem("reference", fo=refs, remote_protocol="file", asynchronous=True)
            array = zarr.open_array(FsspecStore(fs, read_only=True), path="v", mode="r")
            cell["outcome"] = "equal" if same(case, array[...], case.truth) else "differs"
    (RESULTS / "without_icechunk.json").write_text(json.dumps(report, indent=1))
    for k, c in report["cells"].items():
        print(k.ljust(40), c["outcome"], c.get("why", "")[:110])
