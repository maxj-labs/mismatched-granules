"""Whole records, without downloading them: remote.py <collection>

Every granule of a record is virtualized through bench/range_server.py's relay, which
forwards byte-range requests to the archive, so only HDF5 metadata is read. The joined
array is then written to Icechunk and spot-checked: cells read through the joined array
are compared with the same cells read from the granule itself.

Start the relay first:  EARTHDATA_TOKEN=... uv run --with requests bench/range_server.py 8765 0
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import icechunk
import numpy as np
import requests
import zarr
from zarr.codecs import SwitchCodec

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "bench"))
sys.path.insert(0, str(Path(__file__).parent))

from obspec_utils.readers import BlockStoreReader  # noqa: E402
from obspec_utils.registry import ObjectStoreRegistry  # noqa: E402
from obstore.store import HTTPStore  # noqa: E402
from probe import RangeFile  # noqa: E402
from virtualizarr.manifests import ManifestArray  # noqa: E402
from virtualizarr.manifests.array_api import stack as _stack  # noqa: E402
from virtualizarr.manifests.utils import copy_and_replace_metadata  # noqa: E402


def stack(arrays, axis=0):
    """Give arrays a new leading axis. Their dimension names are dropped first: the
    files do not name the new axis."""
    bare = [ManifestArray(metadata=copy_and_replace_metadata(a.metadata, new_dimension_names=None), chunkmanifest=a.manifest) for a in arrays]
    return _stack(bare, axis=axis)

from virtualizarr.manifests.concat import concatenate_keeping_pieces  # noqa: E402
from virtualizarr.manifests.array_api import _concatenate_as_one_array  # noqa: E402
from virtualizarr.manifests.plan import plan_joined_array  # noqa: E402
from virtualizarr.manifests.utils import manifest_chunk_shape  # noqa: E402
from virtualizarr.parsers import HDFParser  # noqa: E402

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True})
PROXY = "http://127.0.0.1:8765/"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json?sort_key=start_date"
SPECS = {
    "Daymet_NA_dayl": dict(q="&short_name=Daymet_Daily_V4R1_2129&provider=ORNL_CLOUD", pattern="*daily_na_dayl_*", var="dayl", stack=False, block=1 << 15),
    "OCO2_L2_Lite_FP": dict(q="&short_name=OCO2_L2_Lite_FP&provider=GES_DISC&version=11.2r", var="co2_profile_apriori", stack=False, block=1 << 16, stride=8),
    "OMPS_NPP_NMTO3_L3_DAILY": dict(q="&short_name=OMPS_NPP_NMTO3_L3_DAILY&provider=GES_DISC", var="ColumnAmountO3", stack=True, block=1 << 16, stride=17),
    "MODISA_L3m_CHL": dict(q="&short_name=MODISA_L3m_CHL&provider=OB_CLOUD", pattern="*DAY.CHL.chlor_a.4km.nc", var="chlor_a", stack=True, block=1 << 20, stride=45),
}
rng = np.random.default_rng(0)


def granules(spec: dict) -> list[str]:
    q = CMR + spec["q"]
    if "pattern" in spec:
        q += "&options[readable_granule_name][pattern]=true&readable_granule_name=" + urllib.parse.quote(spec["pattern"])
    urls, after = [], None
    while True:
        r = requests.get(q + "&page_size=2000", headers={"CMR-Search-After": after} if after else {}, timeout=120)
        items = r.json()["items"]
        urls += [next(u["URL"] for u in i["umm"]["RelatedUrls"] if u["Type"] == "GET DATA" and u["URL"].startswith("https")) for i in items]
        after = r.headers.get("CMR-Search-After")
        if not items or not after:
            return urls


def relay(url: str) -> str:
    return PROXY + "remote/" + url[len("https://"):]


def stats() -> dict:
    return json.loads(urllib.request.urlopen(PROXY + "__stats").read())


def find(group, name):
    for k, v in group.arrays.items():
        if k == name:
            return v
    for sub in group.groups.values():
        if (got := find(sub, name)) is not None:
            return got
    return None


def parse_one(url: str, collection: str):
    """The variable of one granule, as a ManifestArray (or the error that stopped it)."""
    spec = SPECS[collection]
    registry = ObjectStoreRegistry({PROXY.rstrip("/"): HTTPStore.from_url(PROXY.rstrip("/"), client_options={"allow_http": True})})
    parser = HDFParser(reader_factory=partial(BlockStoreReader, block_size=spec["block"]))
    last: Exception | None = None
    for attempt in range(3):
        try:
            arr = find(parser(url=relay(url), registry=registry)._group, spec["var"])
            return stack([arr], axis=0) if spec["stack"] else arr
        except Exception as err:  # noqa: BLE001
            last = RuntimeError(f"{type(err).__name__}: {str(err)[:150]}")
            time.sleep(2 ** attempt)
    return last


def main(collection: str, limit: int | None = None) -> None:
    spec = SPECS[collection]
    urls = granules(spec)
    report: dict = {"collection": collection, "granules_in_record": len(urls), "variable": spec["var"]}
    if "stride" in spec:  # a regular sample of the record
        urls = urls[:: spec["stride"]]
    if limit:
        urls = urls[:limit]
    report["granules_used"] = len(urls)
    urllib.request.urlopen(PROXY + "__reset").read()
    t0 = time.time()
    # h5py lets one thread at a time read HDF5, so the granules are parsed in processes
    with ProcessPoolExecutor(16) as pool:
        parsed = list(pool.map(parse_one, urls, [collection] * len(urls), chunksize=1))
    failed = [(u, str(a)[:120]) for u, a in zip(urls, parsed) if isinstance(a, Exception) or a is None]
    keep = [(u, a) for u, a in zip(urls, parsed) if not (isinstance(a, Exception) or a is None)]
    urls, arrays = [u for u, _ in keep], [a for _, a in keep]
    s = stats()
    report.update(parse_seconds=round(time.time() - t0, 1), parse_failures=len(failed), first_failures=failed[:3],
                  metadata_requests=s["requests"], metadata_megabytes=round(s["bytes"] / 1e6, 1))
    layouts: dict = {}
    for a in arrays:
        key = json.dumps([list(manifest_chunk_shape(a.metadata)), [c["name"] if isinstance(c, dict) else str(c) for c in a.metadata.to_dict()["codecs"]], str(a.dtype)])
        layouts[key] = layouts.get(key, 0) + 1
    report["layouts"] = {k: v for k, v in sorted(layouts.items(), key=lambda kv: -kv[1])[:8]}
    report["distinct_layouts"] = len(layouts)

    t0 = time.time()
    joined = concatenate_keeping_pieces(arrays, 0, merge=_concatenate_as_one_array)
    pieces = list(getattr(joined, "pieces", [joined]))
    plan = plan_joined_array(pieces, axis=0)
    report.update(pieces=len(pieces), strategy=plan.strategy, cases=len(plan.cases), references=len(plan.refs), source_chunks=plan.source_chunks,
                  shape=list(plan.shape), plan_seconds=round(time.time() - t0, 1), notes={k: str(v)[:200] for k, v in plan.notes.items()})

    path = HERE / "stores_http" / f"remote_{collection}"
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(url_prefix=PROXY, store=icechunk.http_store(opts={"allow_http": "true"})))
    # a local repository's defaults split every fetch into small parts; use the object-store ones
    config.storage = icechunk.StorageSettings(concurrency=icechunk.StorageConcurrencySettings(max_concurrent_requests_for_object=18, ideal_concurrent_request_size=12 << 20))
    repo = icechunk.Repository.create(storage=icechunk.local_filesystem_storage(str(path)), config=config, authorize_virtual_chunk_access={PROXY: None})
    session = repo.writable_session("main")
    t0 = time.time()
    arr = zarr.create_array(session.store, name="v", shape=plan.shape, chunks=plan.chunk_grid, dtype=plan.dtype, fill_value=plan.fill_value,
                            serializer=SwitchCodec(axis=0, cases=plan.cases), compressors=None, filters=None)
    bad = session.store.set_virtual_refs("v", [icechunk.VirtualChunkSpec(**r) for r in plan.refs], validate_containers=True)
    session.commit("whole record")
    report.update(write_seconds=round(time.time() - t0, 1), refs_refused=bad and len(bad), zarr_json_bytes=len(json.dumps(arr.metadata.to_dict(), default=str)),
                  store_megabytes=round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2))

    # spot checks against the granules themselves
    sizes = [a.shape[0] for a in arrays]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    unified = zarr.open_array(repo.readonly_session("main").store, path="v", mode="r")
    sess = requests.Session()

    def open_granule(i: int):
        size = int(sess.head(relay(urls[i]), timeout=120).headers["Content-Length"])
        return h5py.File(RangeFile(sess, relay(urls[i]), size, block=1 << 18), "r")

    # granules to check: the ends, both sides of every change of layout, and a few at random
    change = [i for i in range(1, len(arrays)) if manifest_chunk_shape(arrays[i].metadata) != manifest_chunk_shape(arrays[i - 1].metadata)
              or str(arrays[i].metadata.to_dict()["codecs"]) != str(arrays[i - 1].metadata.to_dict()["codecs"])]
    report["layout_changes_at_granule"] = change[:20]
    check = sorted({0, len(arrays) - 1, *[j for i in change[:4] for j in (i - 1, i)], *[int(x) for x in rng.integers(0, len(arrays), 4)]})
    urllib.request.urlopen(PROXY + "__reset").read()
    checks = []
    for i in check:
        with open_granule(i) as h:
            d = find_h5(h, spec["var"])
            local_shape = d.shape
            for _ in range(3):
                cell = tuple(int(rng.integers(0, n)) for n in local_shape)
                want = d[cell]
                where = (int(offsets[i]), *cell) if spec["stack"] else (int(offsets[i]) + cell[0], *cell[1:])
                got = unified[where]
                checks.append(bool(np.array_equal(np.asarray(got), np.asarray(want).astype(plan.dtype), equal_nan=True)))
    s = stats()
    report.update(granules_checked=len(check), cells_checked=len(checks), cells_equal=int(sum(checks)), check_megabytes=round(s["bytes"] / 1e6, 1))
    out = HERE / "results_http" / "remote"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{collection}.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


def find_h5(h, name):
    if name in h:
        return h[name]
    hit = []
    h.visititems(lambda k, v: hit.append(v) if isinstance(v, h5py.Dataset) and k.split("/")[-1] == name else None)
    return hit[0]


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
