"""Does an option survive an operation on the manifest that knows nothing about it?

    python compose.py

Three operations, each done the way a manifest tool would: the array's shape and chunk
grid lose the chunks, and the remaining references move up. Whatever else an option
keeps (a codec's configuration, a side array, a list of pieces, an attribute) is left
exactly as it was.

    drop first chunk      the first chunk along the joined axis goes
    drop first file       every chunk of the first file goes
    drop last chunk       the last chunk along the joined axis goes

Chunks are counted in each option's own grid, so every option is asked the same three
things whatever its chunks are. If reads still equal the files, the option composes with
that operation as it stands. Otherwise it either fails with an error or, worse, returns
wrong values.

Every option that stores a virtual array is run, with nothing decided in advance. Two
caveats on reading the results, both about emulations: the store-context option keeps
its table in a Python wrapper indexed by position, where a store that kept the context
with each reference would carry it along; and the grid option keeps its encoded sizes in
an attribute, where the proposal puts them in the chunk grid that such a tool rewrites.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
from zarr.core.chunk_grids import ChunkGrid

import arms
import cases as cases_module
import common
from common import RESULTS, NotExpressible, pieces_of, same

common.STORE_SUFFIX = "__composed"

TRIALS = ["codec_eras", "padded_last_chunk", "chunk_shape_change", "layouts_alternate", "packing_per_file", "fill_value_differs"]
ORDINARY = {"materialize_odd_chunks", "native_copy"}  # ordinary arrays: there is nothing beside the manifest to go stale
OPERATIONS = ["drop first chunk", "drop first file", "drop last chunk"]


def edges_of(array) -> list[int]:
    return [int(n) for n in ChunkGrid.from_metadata(array.metadata).chunk_sizes[0]]


def main() -> None:
    all_cases = cases_module.build()
    report: dict = {"operations": OPERATIONS, "ordinary_arrays": sorted(ORDINARY), "cells": {}}
    for name in TRIALS:
        case = all_cases[name]
        first_file = 4 if 0 in case.absent else h5py.File(case.files[0], "r")["v"].shape[0]
        for arm, build in arms.ARMS.items():
            if arm in ORDINARY or arm == "separate_files":
                continue
            arms.DROP_FIRST = arms.DROP_LAST = 0
            try:
                whole = build(case, pieces_of(case))
            except NotExpressible:
                continue
            edges = edges_of(whole.array)
            starts = np.concatenate([[0], np.cumsum(edges)])
            total = whole.array.shape[0] if arm != "domain_convention" else case.truth.shape[0]
            boundary = int(np.searchsorted(starts, first_file)) if first_file in starts else None
            for operation in OPERATIONS:
                cell: dict = {}
                if operation == "drop first chunk":
                    first, last = 1, 0
                elif operation == "drop first file":
                    if boundary is None or boundary == 0:
                        cell.update(outcome="not applicable", why="no chunk boundary falls at the end of the first file")
                        report["cells"][f"{arm}|{name}|{operation}"] = cell
                        continue
                    first, last = boundary, 0
                else:
                    first, last = 0, 1
                if first + last >= len(edges):
                    continue
                report["cells"][f"{arm}|{name}|{operation}"] = cell
                # what the remaining chunks hold, in the files' terms
                if arm == "domain_convention":  # its grid counts stored sizes; the data behind each chunk is what matters
                    logical = [b - a for a, b in whole.array.attrs["domain"]]
                    per_chunk = _logical_per_chunk(edges, whole.array.attrs["domain"])
                else:
                    per_chunk = [min(e, max(0, total - int(s))) for e, s in zip(edges, starts)]
                lo = sum(per_chunk[:first])
                hi = sum(per_chunk[: len(per_chunk) - last])
                want = case.truth[lo:hi]
                arms.DROP_FIRST, arms.DROP_LAST = first, last
                try:
                    built = build(case, pieces_of(case))
                    got = np.asarray(built.read((slice(None),) * 3))
                    if same(case, got, want):
                        cell["outcome"] = "composes"
                    else:
                        cell["outcome"] = "wrong values, no error"
                        cell["why"] = "the wrong shape" if got.shape != want.shape else f"{int((~np.isclose(got, want, equal_nan=True)).sum())} of {want.size} cells differ"
                except Exception as err:
                    cell.update(outcome="fails with an error", why=f"{type(err).__name__}: {str(err)[:140]}")
                finally:
                    arms.DROP_FIRST = arms.DROP_LAST = 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compose.json").write_text(json.dumps(report, indent=1))
    mark = {"composes": "ok", "wrong values, no error": "WRONG", "fails with an error": "error", "not applicable": "n/a"}
    for operation in OPERATIONS:
        print("\n" + operation)
        print(" " * 30, *[n[:12].ljust(12) for n in TRIALS])
        for arm in arms.ARMS:
            row = [report["cells"].get(f"{arm}|{n}|{operation}", {}).get("outcome") for n in TRIALS]
            if any(row):
                print(arm.ljust(30), *[(mark[r] if r else ".").ljust(12) for r in row])


def _logical_per_chunk(edges: list[int], domain: list[list[int]]) -> list[int]:
    """For an array stored with its padding: how many valid steps each stored chunk holds."""
    out, start = [], 0
    for e in edges:
        out.append(sum(max(0, min(b, start + e) - max(a, start)) for a, b in domain))
        start += e
    return out


if __name__ == "__main__":
    main()
