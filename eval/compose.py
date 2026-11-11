"""Does an option survive an operation on the manifest that knows nothing about it?

    python compose.py

The operation: drop the first chunks along the joined axis (a chunk-aligned slice, or a
shift). Only what any manifest tool can do is done: the array's shape and chunk grid lose
those chunks, and the remaining references move up. The codec's configuration, and any
table the option keeps elsewhere, are left exactly as they were. If reads still equal
the files, the option composes with such operations as it stands.
"""

import json

import numpy as np

import arms
import cases as cases_module
from common import RESULTS, NotExpressible, pieces_of, same

TRIALS = {"codec_eras": 4, "padded_last_chunk": 2, "chunk_shape_change": 1, "layouts_alternate": 3, "packing_per_file": 5}
SKIP = {
    "array_concat": "the joined array has no chunks of its own; dropping part of it means editing the codec's list of pieces, and only whole pieces can go",
    "zarr_reference": "one chunk per piece: whole pieces can be dropped as chunks, part of a piece cannot",
    "materialize_odd_chunks": "an ordinary array: nothing beyond the manifest",
    "native_copy": "an ordinary array: nothing beyond the manifest",
    "separate_files": "not a stored array",
}

if __name__ == "__main__":
    all_cases = cases_module.build()
    report: dict = {"operation": "drop the first chunks along the joined axis; codec configuration untouched", "by_inspection": SKIP, "cells": {}}
    for name, k in TRIALS.items():
        case = all_cases[name]
        for arm, build in arms.ARMS.items():
            if arm in SKIP:
                continue
            arms.DROP_CHUNKS = 0
            try:
                whole = build(case, pieces_of(case))
            except NotExpressible:
                continue
            grid0 = whole.array.read_chunk_sizes[0] if hasattr(whole.array, "read_chunk_sizes") else None
            sizes = json.loads(json.dumps(whole.array.metadata.to_dict()["chunk_grid"], default=str))["configuration"]
            edges = sizes.get("chunk_shapes", [None])[0]
            arms.DROP_CHUNKS = k
            cell: dict = {}
            report["cells"][f"{arm}|{name}"] = cell
            try:
                built = build(case, pieces_of(case))
                dropped = whole.array.shape[0] - built.array.shape[0]
                want = case.truth[dropped:] if arm != "domain_convention" else None
                if arm == "domain_convention":
                    cell["outcome"] = "needs its own metadata rewritten"
                    cell["why"] = "the declared domain still lists the dropped chunks' ranges"
                    got = None
                    try:
                        got = built.read((slice(None),) * 3)
                    except Exception as err:
                        cell["why"] += f" ({type(err).__name__})"
                    continue
                got = built.read((slice(None),) * 3)
                cell["outcome"] = "composes as it stands" if same(case, got, want) else "needs its own metadata rewritten"
                if cell["outcome"] != "composes as it stands":
                    cell["why"] = "reads without error but the values are wrong" if np.asarray(got).shape == want.shape else "reads the wrong shape"
            except Exception as err:
                cell.update(outcome="needs its own metadata rewritten", why=f"{type(err).__name__}: {str(err)[:160]}")
            finally:
                arms.DROP_CHUNKS = 0
    (RESULTS / "compose.json").write_text(json.dumps(report, indent=1))
    names = list(TRIALS)
    print(" " * 24, *[n[:12].ljust(12) for n in names])
    for arm in arms.ARMS:
        row = [report["cells"].get(f"{arm}|{n}", {}).get("outcome", "") for n in names]
        if any(row):
            print(arm.ljust(24), *[{"composes as it stands": "ok", "needs its own metadata rewritten": "REWRITE", "": "."}[r].ljust(12) for r in row])
    for k, c in report["cells"].items():
        if "why" in c:
            print(k.ljust(40), c["why"][:150])
