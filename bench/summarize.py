"""Collect the result files into results/RESULTS.md: summarize.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
R = HERE / "results"
out: list[str] = ["# Daymet: approaches compared", ""]


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


cmp_ = load("bench_pr_tmin_compare.json")
ref = load("bench_pr_tmin_1950_2025_refkv.json")
rows = cmp_["rows"] + ref["rows"]
approaches = ["concat", "switch", "refkv switch", "separate", "files"]
workloads = list(dict.fromkeys(r["workload"] for r in rows))
out += [f"## Puerto Rico tmin, {cmp_['years'][0]}-{cmp_['years'][1]} ({cmp_['build']['files']} files, shape {cmp_['shape']})", "",
        "Median seconds (store reads). Local disk, warm cache. Every result equals the files: "
        + str(all(r["equal_to_files"] for r in rows)), "",
        "| Read | " + " | ".join(approaches) + " |", "|---|" + "---:|" * len(approaches)]
for w in workloads:
    cells = []
    for a in approaches:
        r = next((r for r in rows if r["workload"] == w and r["approach"] == a), None)
        cells.append("n/a" if r is None else f"{r['seconds_median']:.3f}" + (f" ({r['store_gets']:,})" if r["store_gets"] is not None else ""))
    out.append(f"| {w} | " + " | ".join(cells) + " |")
arr = cmp_["arrays"]
out += ["", "| | concat | switch | refkv switch |", "|---|---:|---:|---:|",
        f"| zarr.json of the variable (bytes) | {arr['concat']['zarr_json_bytes']:,} | {arr['switch']['zarr_json_bytes']:,} | {ref['zarr_json_bytes']:,} |",
        f"| store size (bytes) | {arr['concat']['store_bytes']:,} | {arr['switch']['store_bytes']:,} | {ref['store_bytes']:,} |",
        f"| default dask chunks from xarray | {arr['concat']['default_dask_chunks']:,} | {arr['switch']['default_dask_chunks']:,} | {ref['default_dask_chunks']:,} |",
        f"| size of one default dask chunk (MB) | {arr['concat']['default_dask_chunk_megabytes']:,} | {arr['switch']['default_dask_chunk_megabytes']:,} | 0.3 |",
        f"| nodes in the store | {arr['concat']['nodes_in_store']} | {arr['switch']['nodes_in_store']} | n/a |",
        f"| build after parsing (s) | {cmp_['build']['concat']['build_seconds']} | {cmp_['build']['switch']['build_seconds']} | {ref['build_seconds']} |",
        ""]

na = load("bench_na_dayl_hybrid.json")
nb = load("bench_na_dayl_build.json")
r2 = load("bench_na_dayl_2020_2021_refkv.json")
r3 = load("bench_na_dayl_2019_2021_refkv.json")
out += ["## North America dayl, 2019-2021 (3 files, 4.3 GB)", "",
        f"- switch (Icechunk, one array): not feasible. {nb['switch']['why']}",
        f"- refkv switch, 2019-2021: not feasible. {r3['why']}",
        f"- refkv switch, 2020-2021 only: feasible, equal to files: {all(r['equal_to_files'] for r in r2['rows'])}",
        "", "| Read | concat, 3 pieces | hybrid, 2 pieces |", "|---|---:|---:|"]
for w in dict.fromkeys(r["workload"] for r in na["rows"]):
    cells = [next(r for r in na["rows"] if r["workload"] == w and r["approach"] == a) for a in ("concat, 3 pieces", "hybrid, 2 pieces")]
    out.append(f"| {w} | " + " | ".join(f"{c['seconds_median']:.3f} ({c['store_gets']:,}){'' if c['equal_to_files'] else ' NOT EQUAL'}" for c in cells) + " |")
out += ["", "## A reader without the new codecs (zarr-python 3.4.2)", ""]
out += [f"- `{k}`: {v}" for k, v in load("plain_reader.json").items() if k != "zarr"]
(R / "RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
