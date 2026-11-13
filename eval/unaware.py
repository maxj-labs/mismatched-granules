"""Open every store the matrix made with readers that have none of the proposed changes.

    python unaware.py

Two readers: released zarr-python with the modified Icechunk (does the Zarr side fail
safely?) and released zarr-python with released Icechunk (does the store side?).
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import cases as cases_module
from common import PREFIX, RESULTS, STORES

HERE = Path(__file__).resolve().parent
WHEEL = str(next((HERE.parent / "icechunk" / "target" / "wheels").glob("*.whl")))
KNOWS_CODECS = "zarr-python with the codecs, released Icechunk"
READERS = {
    "released zarr-python, modified Icechunk": ["--no-project", "--with", "zarr", "--with", WHEEL, "--with", "numpy"],
    "released zarr-python, released Icechunk": ["--no-project", "--with", "zarr", "--with", "icechunk", "--with", "numpy"],
    # the `demo` project: this workspace's zarr-python and VirtualiZarr branches with Icechunk from PyPI
    KNOWS_CODECS: ["--project", str(HERE.parent / "demo")],
}


def probe(job: tuple[str, str, str]) -> tuple[str, str, dict]:
    key, reader, truth = job
    arm, case = key.split("|")
    cmd = ["uv", "run", *READERS[reader], "python", str(HERE / "stock_read.py"), str(STORES / arm / case), truth, PREFIX]
    # a clean environment: nothing of the evaluation's own (forked) packages may leak in
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH") and not k.startswith("UV_")}
    if reader == KNOWS_CODECS:
        env["EVAL_KNOWS_CODECS"] = "1"
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/")
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    # no JSON line means the probe itself failed to start: that is a fault of the test, not a refusal
    return key, reader, json.loads(lines[-1]) if lines else {"outcome": "probe failed", "detail": (done.stderr.strip().splitlines() or ["no output"])[-1][:200]}


if __name__ == "__main__":
    matrix = json.loads((RESULTS / "matrix.json").read_text())
    all_cases = cases_module.build()
    truths = {}
    for name, case in all_cases.items():
        truths[name] = str(STORES / f"truth_{name}.npy")
        np.save(truths[name], case.truth)
    jobs = [(key, reader, truths[key.split("|")[1]]) for key, cell in matrix["cells"].items()
            if cell["outcome"] == "equal" and (STORES / key.split("|")[0] / key.split("|")[1]).exists() for reader in READERS]
    report: dict = {}
    with ThreadPoolExecutor(6) as pool:
        for key, reader, result in pool.map(probe, jobs):
            report.setdefault(key, {})[reader] = result
    (RESULTS / "unaware.json").write_text(json.dumps(report, indent=1))
    for reader in READERS:
        print("\n" + reader)
        for arm in matrix["arms"]:
            outcomes = [report[k][reader]["outcome"] for k in report if k.startswith(arm + "|")]
            if outcomes:
                print(f"  {arm:24s}", {o: outcomes.count(o) for o in sorted(set(outcomes))})
