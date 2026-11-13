"""Shared pieces of the evaluation: repositories, parsed files, comparison with the files."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import icechunk
import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import HDFParser
from zarr.core.metadata.v3 import ArrayV3Metadata

from cases import DATA, HERE, Case

warnings.filterwarnings("ignore")
zarr.config.set({"array.rectilinear_chunks": True})
STORES = HERE / "stores"
RESULTS = HERE.parent / "results" / "eval"
PREFIX = f"file://{DATA}/"
STORE_SUFFIX = ""  # compose.py writes its altered arrays beside the matrix's, not over them
CHUNK_STORE = None  # perf.py points both at an HTTP server that counts requests
registry = ObjectStoreRegistry({"file://": LocalStore()})


class NotExpressible(Exception):
    """The option has no way to describe this case. The message says why."""


@dataclass
class Built:
    """What an option made of a case."""

    read: Callable[[tuple], np.ndarray]
    array: Any = None  # the zarr array a user would open, when there is one
    one_array: bool = True  # a reader sees one ordinary array
    info: dict = field(default_factory=dict)
    store_path: Path | None = None
    reopen: Callable[[], Any] | None = None  # opens the array afresh, as a new reader would


def new_repo(arm: str, case: str) -> tuple[icechunk.Repository, Path]:
    path = STORES / (arm + STORE_SUFFIX) / case
    shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = icechunk.RepositoryConfig.default()
    config.storage = icechunk.StorageSettings(
        concurrency=icechunk.StorageConcurrencySettings(max_concurrent_requests_for_object=18, ideal_concurrent_request_size=12 << 20)
    )
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(url_prefix=PREFIX, store=CHUNK_STORE or icechunk.local_filesystem_store(PREFIX)))
    repo = icechunk.Repository.create(
        storage=icechunk.local_filesystem_storage(str(path)), config=config, authorize_virtual_chunk_access={PREFIX: None}
    )
    return repo, path


def open_store(path: Path) -> Any:
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)), authorize_virtual_chunk_access={PREFIX: None})
    return repo.readonly_session("main").store


def store_bytes(path: Path) -> dict:
    def total(sub: str) -> int:
        return sum(f.stat().st_size for f in (path / sub).rglob("*") if f.is_file())

    # `metadata_bytes`: the snapshots, which hold every array's zarr.json, as stored (compressed).
    # `virtual_bytes`: everything the store keeps that is not chunk data.
    native = total("chunks")
    return {"manifest_bytes": total("manifests"), "metadata_bytes": total("snapshots"), "native_chunk_bytes": native,
            "store_bytes": total(""), "virtual_bytes": total("") - native}


def pieces_of(case: Case) -> list[ManifestArray]:
    """Each file's `v` as a ManifestArray, ready to be joined along axis 0.

    - A file without the variable is a piece that stores nothing.
    - A packed file becomes a float64 array whose codecs undo the packing, so that
      per-file packing is a difference between codec chains like any other.
    """
    arrays: list[ManifestArray | None] = []
    for i, f in enumerate(case.files):
        if i in case.absent:
            arrays.append(None)
            continue
        ma = HDFParser()(url=f"file://{f}", registry=registry)._group.arrays["v"]
        if case.packed:
            with h5py.File(f, "r") as h:
                scale, offset = (float(np.ravel(h["v"].attrs[k])[0]) for k in ("scale_factor", "add_offset"))
            md = json.loads(json.dumps(ma.metadata.to_dict(), default=str))
            stored = md["data_type"]
            md["data_type"], md["fill_value"] = "float64", "NaN"
            md["codecs"] = [
                {"name": "scale_offset", "configuration": {"offset": offset, "scale": 1.0 / scale}},
                {"name": "cast_value", "configuration": {
                    "data_type": stored, "rounding": "nearest-even", "scalar_map": {"encode": [["NaN", int(ma.metadata.fill_value)]]}}},
                *md["codecs"],
            ]
            ma = ManifestArray(metadata=ArrayV3Metadata.from_dict(md), chunkmanifest=ma.manifest)
        arrays.append(ma)
    ref = next(a for a in arrays if a is not None)
    return [ref.with_fill_value_only(ref.metadata.fill_value) if a is None else a for a in arrays]


def selections(case: Case) -> dict[str, tuple]:
    """Reads every option must answer: everything, a window across each seam, single cells."""
    shape = case.truth.shape
    sizes = [4 if i in case.absent else h5py.File(f, "r")["v"].shape[0] for i, f in enumerate(case.files)]
    seams = np.cumsum(sizes)[:-1]
    out: dict[str, tuple] = {"everything": (slice(None),) * 3}
    for k, b in enumerate(seams[:3]):
        out[f"window across seam {k + 1}"] = (slice(max(0, int(b) - 3), min(shape[0], int(b) + 3)), slice(5, 17), slice(7, 23))
    rng = np.random.default_rng(1)
    for k in range(5):
        out[f"cell {k + 1}"] = (int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1])), int(rng.integers(0, shape[2])))
    out["one row along the joined axis"] = (slice(None), 11, 13)
    return out


def same(case: Case, got: np.ndarray, want: np.ndarray) -> bool:
    got = np.asarray(got)
    if got.shape != np.asarray(want).shape:
        return False
    if case.packed:  # unpacking is floating-point arithmetic: equal to rounding
        return bool(np.allclose(got, want, rtol=0, atol=1e-9, equal_nan=True))
    return bool(np.array_equal(got.astype(want.dtype, copy=False) if got.dtype != np.asarray(want).dtype else got, want, equal_nan=True))
