"""TensorStore's stack driver as the reference design for array concatenation.

    uv run --no-project --with tensorstore --with numpy python tensorstore_probe.py [layers ...]

zarr-specs#288 and #399 point at it as the general form (layers with index transforms).
TensorStore cannot read virtual references, so this is not a virtualization test: each
layer is a small native Zarr v3 array. It answers what the design costs as layers grow:
the size of the spec that lists them, the time to open it, and the time to read.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorstore as ts

NY, NX = 24, 30


def probe(n: int) -> dict:
    root = Path(tempfile.mkdtemp(prefix="ts_stack_", dir=Path(__file__).resolve().parent / "stores"))
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(n, NY, NX)).astype("float32")
    t0 = time.perf_counter()
    layers = []
    for i in range(n):
        spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(root / f"a{i}")},
                "metadata": {"shape": [1, NY, NX], "data_type": "float32", "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, NY, NX]}}}}
        ts.open(spec, create=True).result()[...] = truth[i : i + 1]
        layers.append({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(root / f"a{i}")},
                       "transform": {"input_inclusive_min": [i, 0, 0], "input_exclusive_max": [i + 1, NY, NX], "output": [{"input_dimension": 0, "offset": -i}, {"input_dimension": 1}, {"input_dimension": 2}]}})
    write_seconds = time.perf_counter() - t0
    spec = {"driver": "stack", "dtype": "float32", "layers": layers}
    spec_bytes = len(json.dumps(spec))
    t0 = time.perf_counter()
    stack = ts.open(spec).result()
    open_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = stack[...].read().result()
    read_seconds = time.perf_counter() - t0
    picks = rng.integers(0, n, 200)
    t0 = time.perf_counter()
    some = [stack[int(i)].read().result() for i in picks]
    one = (time.perf_counter() - t0) / len(picks)
    ok = bool(np.array_equal(got, truth) and all(np.array_equal(s, truth[int(i)]) for s, i in zip(some, picks)))
    shutil.rmtree(root)
    return {"layers": n, "spec_bytes": spec_bytes, "bytes_per_layer": round(spec_bytes / n), "write_seconds": round(write_seconds, 2),
            "open_seconds": round(open_seconds, 3), "read_all_seconds": round(read_seconds, 2), "microseconds_per_layer": round(read_seconds / n * 1e6),
            "one_layer_milliseconds": round(one * 1e3, 2), "equal": ok}


if __name__ == "__main__":
    out = {"tensorstore": ts.__version__ if hasattr(ts, "__version__") else "?", "runs": []}
    for n in [int(a) for a in sys.argv[1:]] or [1000, 4000, 16000]:
        out["runs"].append(probe(n))
        print(json.dumps(out["runs"][-1]), flush=True)
    path = Path(__file__).resolve().parent.parent / "results" / "eval" / "tensorstore_stack.json"
    path.write_text(json.dumps(out, indent=1))
