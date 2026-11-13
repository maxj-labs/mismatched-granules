"""Does an option survive an operation on the manifest that knows nothing about it?

    python compose.py

Three operations, each done the way a manifest tool would: the array's shape and chunk
grid lose the chunks, and the remaining references move up. Whatever else an option
keeps (a codec's configuration, a side array, a list of pieces, an attribute) is left
exactly as it was.

    drop first chunk      the first chunk along the joined axis goes
    drop first file       every chunk of the first file goes
    drop last chunk       the last chunk along the joined axis goes

and three that add a file, again as such a tool would: the array gains the file's chunks,
with the references a parser made for that file, in the right place; what the option keeps
for itself is as it was before the file arrived.

    append a file         the new file is the last
    insert a file         the new file goes in the middle
    prepend a file        the new file is the first (a backfill)

and one that replaces a file with a reprocessed one of the same length:

    replace a file        another file's chunks and references take a file's place

For the file that arrives, the references are the ones a writer for that design would
make for it: plain byte ranges by position, the same with a key in front for the keyed
designs. So "a tool that does not know the option" means a writer that adds or removes
references and rewrites nothing the array already has.

Each result notes how the arriving file relates to the array: its layout is already
there; it is new and only adds a case; or it changes how the files already there must be
described (a different fill value, data type or chunk shape), in which case the whole
array has to be planned again in any design.

Every result is judged on several reads, not one: the whole array, each file's part on
its own, and single cells. An option "composes" only if every read equals the files.
If any read returns wrong values without an error, that is what is recorded, even where
other reads fail loudly.

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
ADDITIONS = ["append a file", "insert a file", "prepend a file"]
REPLACEMENT = "replace a file"


def judge(case, read, want: np.ndarray, sizes: list[int]) -> dict:
    """Read the whole array, each part on its own, and single cells; compare each with `want`."""
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    selections = [(slice(None),) * 3]
    selections += [(slice(int(a), int(b)), slice(None), slice(None)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    rng = np.random.default_rng(2)
    selections += [(int(rng.integers(0, want.shape[0])), int(rng.integers(0, want.shape[1])), int(rng.integers(0, want.shape[2]))) for _ in range(6)]
    ok = wrong = errors = 0
    why = ""
    for sel in selections:
        try:
            got = np.asarray(read(sel))
        except Exception as err:
            errors += 1
            why = why or f"{type(err).__name__}: {str(err)[:140]}"
            continue
        if same(case, got, want[sel]):
            ok += 1
        else:
            wrong += 1
    out = {"reads": len(selections), "reads_correct": ok, "reads_wrong": wrong, "reads_failed": errors}
    if wrong:
        out.update(outcome="wrong values, no error", why=f"{wrong} of {len(selections)} reads returned wrong values" + (f"; {errors} failed" if errors else ""))
    elif errors:
        out.update(outcome="fails with an error", why=why)
    else:
        out["outcome"] = "composes"
    return out


def relation(pieces: list, own: list) -> str:
    """How the files that arrive relate to the array made from `own`."""
    full, before = arms._plan(pieces, "key"), arms._plan(own, "key")
    full_keys, before_keys = {c["key"] for c in full.cases}, {c["key"] for c in before.cases}
    if full.strategy != before.strategy or full.dtype != before.dtype or not before_keys <= full_keys:
        return "changes how existing files are described"
    return "new to the array" if full_keys - before_keys else "already in the array"



def edges_of(array) -> list[int]:
    return [int(n) for n in ChunkGrid.from_metadata(array.metadata).chunk_sizes[0]]


def main() -> None:
    all_cases = cases_module.build()
    report: dict = {"operations": OPERATIONS, "additions": ADDITIONS, "replacement": REPLACEMENT, "ordinary_arrays": sorted(ORDINARY), "cells": {}}
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
                    kept = [n for n in per_chunk[first : len(per_chunk) - last]]
                    cell.update(judge(case, built.read, want, kept))
                except Exception as err:
                    cell.update(outcome="fails with an error", why=f"{type(err).__name__}: {str(err)[:140]}")
                finally:
                    arms.DROP_FIRST = arms.DROP_LAST = 0
            # a file added later, with the option's own state left as it was before
            n_files = len(case.files)
            file_sizes = [4 if i in case.absent else h5py.File(f, "r")["v"].shape[0] for i, f in enumerate(case.files)]
            for operation in ADDITIONS:
                if operation == "insert a file" and n_files < 3:
                    continue
                omit = {"append a file": n_files - 1, "insert a file": n_files // 2, "prepend a file": 0}[operation]
                if omit in case.absent:
                    continue
                pieces = pieces_of(case)
                cell = {"layout": relation(pieces, [q for i, q in enumerate(pieces) if i != omit])}
                report["cells"][f"{arm}|{name}|{operation}"] = cell
                arms.OMIT = omit
                try:
                    built = build(case, pieces_of(case))
                    cell.update(judge(case, built.read, case.truth, file_sizes))
                except Exception as err:
                    cell.update(outcome="fails with an error", why=f"{type(err).__name__}: {str(err)[:140]}")
                finally:
                    arms.OMIT = None
            # a file replaced by another of the same length: the second file's chunks take the first's place
            if n_files >= 2 and not case.absent and file_sizes[0] == file_sizes[1]:
                original = pieces_of(case)
                changed = [original[1], *original[1:]]
                bounds = np.concatenate([[0], np.cumsum(file_sizes)])
                want = case.truth.copy()
                want[: file_sizes[0]] = case.truth[int(bounds[1]) : int(bounds[2])]
                after, before = arms._plan(changed, "key"), arms._plan(original, "key")
                # the file that arrives is a copy of one the array holds, so its layout is in the
                # table unless planning the new set describes the files differently
                fits = after.strategy == before.strategy and after.dtype == before.dtype and {c["key"] for c in after.cases} <= {c["key"] for c in before.cases}
                cell = {"layout": "already in the array" if fits else "changes how existing files are described"}
                cell["replacement_has_the_same_layout"] = bool(
                    arms._plan([original[0]], "key").cases[0]["key"] == arms._plan([original[1]], "key").cases[0]["key"])
                report["cells"][f"{arm}|{name}|{REPLACEMENT}"] = cell
                arms.OWN_PIECES = original
                try:
                    built = build(case, changed)
                    cell.update(judge(case, built.read, want, file_sizes))
                except Exception as err:
                    cell.update(outcome="fails with an error", why=f"{type(err).__name__}: {str(err)[:140]}")
                finally:
                    arms.OWN_PIECES = None
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compose.json").write_text(json.dumps(report, indent=1))
    mark = {"composes": "ok", "wrong values, no error": "WRONG", "fails with an error": "error", "not applicable": "n/a"}
    for operation in OPERATIONS + ADDITIONS + [REPLACEMENT]:
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
