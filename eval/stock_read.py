"""What a reader without any of the proposed changes makes of a store.

    python stock_read.py <repository path> <truth.npy> <virtual chunk prefix>

Run in an environment with released zarr-python (and either Icechunk). Prints JSON:
"refuses" (an error, the safe outcome), "reads correctly", or "WRONG VALUES" (silent).
"""

import json
import sys
import warnings

import icechunk
import numpy as np
import zarr

warnings.filterwarnings("ignore")
path, truth_path, prefix = sys.argv[1:4]
truth = np.load(truth_path)
out = {"zarr": zarr.__version__, "icechunk": icechunk.__version__}
import os

if os.environ.get("EVAL_KNOWS_CODECS"):  # a reader with the codecs but not the store change
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import xcodecs  # noqa: F401
else:
    assert "daymet-virtual" not in zarr.__file__, "this must run with released zarr-python"
out["icechunk_takes_parts"] = "prefix" in (icechunk.VirtualChunkSpec.__init__.__doc__ or "") or hasattr(icechunk.VirtualChunkSpec, "prefix")
try:
    zarr.config.set({"array.rectilinear_chunks": True})
except Exception:
    pass
try:
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path), authorize_virtual_chunk_access={prefix: None})
    array = zarr.open_array(repo.readonly_session("main").store, path="v", mode="r")
    got = array[...]
    if got.shape != truth.shape:
        out.update(outcome="WRONG VALUES", detail=f"reads without error as shape {got.shape}; the data's shape is {truth.shape}")
    elif np.allclose(got, truth, rtol=0, atol=1e-9, equal_nan=True):
        out.update(outcome="reads correctly")
    else:
        out.update(outcome="WRONG VALUES", detail=f"{int((~np.isclose(got, truth, rtol=0, atol=1e-9, equal_nan=True)).sum())} of {truth.size} cells differ, no error")
except BaseException as err:
    out.update(outcome="refuses", detail=f"{type(err).__name__}: {str(err).splitlines()[0][:200] if str(err) else ''}")
print(json.dumps(out))
