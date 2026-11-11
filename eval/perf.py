"""Every option on real granules, with the source behind an HTTP server that counts requests.

    python ../bench/range_server.py 8765 20 &          # 20 ms per request
    BENCH_HTTP=http://127.0.0.1:8765/ python perf.py pr tmin 2014 2025
    BENCH_HTTP=http://127.0.0.1:8765/ python perf.py na dayl 2019 2021

The same reads for every option: values against the files, time (median of 3, one run
for reads over a minute), and the requests and bytes that reach the source.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import icechunk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import common  # noqa: E402

HTTP = os.environ["BENCH_HTTP"]
common.PREFIX = HTTP
common.CHUNK_STORE = icechunk.http_store(opts={"allow_http": "true"})

import arms  # noqa: E402
from cases import Case  # noqa: E402
from common import RESULTS, NotExpressible, registry  # noqa: E402
from compare_util import reset_source, source_stats  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402
from workloads import DATA as DAYMET  # noqa: E402
from workloads import Workloads  # noqa: E402

ARMS = ["switch_position", "switch_key", "switch_case_id", "switch_key_parameters", "switch_index_prefix", "switch_sidecar",
        "switch_store_context", "conditional_codec", "array_concat", "zarr_reference"]


def main() -> None:
    region, var, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    only = sys.argv[5:] or ARMS
    paths = [DAYMET / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in range(first, last + 1)]
    with ThreadPoolExecutor(8) as pool:
        parsed = list(pool.map(lambda p: HDFParser()(url=f"file://{p}", registry=registry)._group.arrays[var], paths))
    pieces = [a.rename_paths(lambda p: p.replace(f"file://{DAYMET}/", HTTP)) for a in parsed]
    work = Workloads(region, var, first, last)
    case = Case(f"daymet_{region}_{var}", f"Daymet {region} {var} {first}-{last}")
    concurrency = os.environ.get("ZARR_CONCURRENCY")  # zarr-python's default is 10 chunks in flight
    if concurrency:
        import zarr

        zarr.config.set({"async.concurrency": int(concurrency)})
    out_path = RESULTS / (f"perf_{region}_{var}.json" if not concurrency else f"perf_{region}_{var}_concurrency_{concurrency}.json")
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    report.update(files=len(paths), years=[first, last], shape=[work.NT, work.NY, work.NX], delay="20 ms per source request")
    report.setdefault("arms", {})
    for arm in only:
        rec: dict = {"reads": {}}
        report["arms"][arm] = rec
        t0 = time.perf_counter()
        try:
            built = arms.ARMS[arm](case, list(pieces))
        except NotExpressible as err:
            rec.update(outcome="not expressible", why=str(err))
            print(arm, "not expressible:", err, flush=True)
            continue
        rec["build_seconds"] = round(time.perf_counter() - t0, 2)
        rec.update({k: v for k, v in built.info.items() if k in ("zarr_json_bytes", "manifest_bytes", "store_bytes", "strategy", "cases")})
        for label, sel in work.selections.items():
            want = work.truth(*sel)
            seconds: list[float] = []
            while len(seconds) < 3 and sum(seconds) < 60:
                array = built.reopen()  # a new session: nothing cached
                reset_source()
                t0 = time.perf_counter()
                got = array[sel]
                seconds.append(time.perf_counter() - t0)
                stats = source_stats()
            rec["reads"][label] = {"equal_to_files": bool(np.array_equal(got, want, equal_nan=True)), "seconds": round(float(np.median(seconds)), 3),
                                   "runs": len(seconds), **stats}
            print(arm, "|", label, "|", rec["reads"][label], flush=True)
        rec["outcome"] = "equal" if all(r["equal_to_files"] for r in rec["reads"].values()) else "differs"
        out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
