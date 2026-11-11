# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr==3.4.2", "icechunk==2.3.1", "xarray", "numpy"]
# ///
"""What released zarr-python, with none of the new codecs, does with each store."""
import json, sys, warnings
from pathlib import Path
import icechunk, zarr
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
prefix = f"file://{HERE / 'data'}/"
out = {"zarr": zarr.__version__}
for kind, paths in {"concat": ["tmin", "_pieces/tmin/0"], "switch": ["tmin"]}.items():
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(HERE / "stores" / f"bench_pr_tmin_{kind}")), authorize_virtual_chunk_access={prefix: None})
    store = repo.readonly_session("main").store
    for path in paths:
        try:
            arr = zarr.open_array(store, path=path, mode="r")
            value = arr[0, 100:102, 100:102]
            out[f"{kind}:{path}"] = f"read values {value.tolist()}"
        except Exception as err:
            out[f"{kind}:{path}"] = f"{type(err).__name__}: {str(err)[:140]}"
print(json.dumps(out, indent=1))
(HERE / "results" / "plain_reader.json").write_text(json.dumps(out, indent=1))
