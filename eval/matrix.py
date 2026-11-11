"""Every option against every case: can it describe the case, and do reads equal the files?

    python matrix.py [case ...]

Writes results/eval/matrix.json and prints a table.
"""

from __future__ import annotations

import json
import sys
import time
import traceback

import cases as cases_module
from arms import ARMS
from common import RESULTS, NotExpressible, pieces_of, same, selections

MARK = {"equal": "ok", "not expressible": "--", "differs": "WRONG", "error": "ERR"}


def run(names: list[str]) -> dict:
    all_cases = cases_module.build()
    report: dict = {"cases": {n: all_cases[n].what for n in names}, "arms": {a: (f.__doc__ or "").strip() for a, f in ARMS.items()}, "cells": {}}
    for name in names:
        case = all_cases[name]
        sels = selections(case)
        for arm, build in ARMS.items():
            cell: dict = {}
            report["cells"][f"{arm}|{name}"] = cell
            t0 = time.perf_counter()
            try:
                built = build(case, pieces_of(case))
            except NotExpressible as err:
                cell.update(outcome="not expressible", why=str(err))
                continue
            except Exception as err:
                cell.update(outcome="error", why=f"build: {type(err).__name__}: {str(err)[:300]}", trace=traceback.format_exc()[-1500:])
                continue
            cell["build_seconds"] = round(time.perf_counter() - t0, 3)
            cell.update(built.info, one_array=built.one_array)
            wrong = []
            try:
                t0 = time.perf_counter()
                for label, sel in sels.items():
                    if not same(case, built.read(sel), case.truth[sel]):
                        wrong.append(label)
                cell["read_seconds"] = round(time.perf_counter() - t0, 3)
                if built.array is not None:
                    cell["data_type"] = str(built.array.dtype)
                    cell["shape_seen_by_a_reader"] = list(built.array.shape)
            except Exception as err:
                cell.update(outcome="error", why=f"read: {type(err).__name__}: {str(err)[:300]}", trace=traceback.format_exc()[-1500:])
                continue
            cell["outcome"] = "differs" if wrong else "equal"
            if wrong:
                cell["why"] = "differs from the files in: " + ", ".join(wrong)
    return report


if __name__ == "__main__":
    names = sys.argv[1:] or list(cases_module.build())
    report = run(names)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "matrix.json"
    if sys.argv[1:] and out.exists():  # a partial run updates the cells it ran
        old = json.loads(out.read_text())
        old["cells"].update(report["cells"])
        old["cases"].update(report["cases"])
        report = old
    out.write_text(json.dumps(report, indent=1, default=str))
    width = max(len(a) for a in ARMS)
    print(" " * width, *[n[:9].ljust(9) for n in names])
    for arm in ARMS:
        print(arm.ljust(width), *[MARK[report["cells"][f"{arm}|{n}"]["outcome"]].ljust(9) for n in names])
    for key, cell in report["cells"].items():
        if cell["outcome"] in ("error", "differs") and key.split("|")[1] in names:
            print("\n", key, cell["why"])
            if "trace" in cell:
                print(cell["trace"][-700:])
