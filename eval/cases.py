"""The ways granules of one record disagree, as small netCDF4 files.

Each case is a set of files whose variable `v` should read as one array along `time`.
Cases 1 to 5 are the ones listed in zarr-specs#399; the rest came up in the NASA survey
or as worst cases for a particular design.

    python cases.py            # writes eval/data/<case>/f<i>.nc and prints a summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NY, NX = 24, 30


@dataclass
class Case:
    name: str
    what: str
    files: list[Path] = field(default_factory=list)
    # The values a reader of the joined array must get: every file's values as the
    # file's own reader presents them (packing undone where the case is about packing).
    truth: np.ndarray | None = None
    packed: bool = False  # v is stored packed (scale_factor / add_offset)
    absent: list[int] = field(default_factory=list)  # files without the variable


def _field(rng: np.random.Generator, t0: int, n: int) -> np.ndarray:
    """A smooth field that continues from file to file, so a seam decoded wrongly shows."""
    t = np.arange(t0, t0 + n)[:, None, None]
    y = np.arange(NY)[None, :, None]
    x = np.arange(NX)[None, None, :]
    return (10 * np.sin(t / 7) + y * 0.5 + np.cos(x / 5) * 3 + rng.normal(scale=0.1, size=(n, NY, NX))).astype("float32")


def _write(path: Path, data: np.ndarray | None, *, chunks, dtype="f4", fill=None, scale=None, offset=None,
           unwritten_from: int | None = None, **compression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if unwritten_from is not None:
        # netCDF4 writes the fill value into every chunk; h5py leaves chunks unallocated
        with h5py.File(path, "w") as h:
            h.create_dataset("time", data=np.arange(data.shape[0], dtype="f8"))
            v = h.create_dataset("v", shape=data.shape, dtype=dtype, chunks=chunks, fillvalue=fill, compression="gzip", compression_opts=1)
            v[:unwritten_from] = data[:unwritten_from]
        return
    with netCDF4.Dataset(path, "w") as ds:
        ds.set_fill_off()
        n = data.shape[0] if data is not None else 4
        ds.createDimension("time", n)
        ds.createDimension("y", NY)
        ds.createDimension("x", NX)
        ds.createVariable("time", "f8", ("time",))[:] = np.arange(n)
        if data is None:
            return
        v = ds.createVariable("v", dtype, ("time", "y", "x"), chunksizes=chunks, fill_value=fill, **compression)
        v.set_auto_maskandscale(False)
        if scale is not None:
            v.scale_factor, v.add_offset = scale, offset
            data = np.round((data - offset) / scale)
        stop = n if unwritten_from is None else unwritten_from
        v[:stop] = data[:stop].astype(dtype)


def _raw(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["v"][...]


def build(write: bool = False) -> dict[str, Case]:
    """The cases. Files already on disk are kept, so that stores made from them stay
    valid (a virtual reference remembers when its file was written); `write` makes them anew."""
    rng = np.random.default_rng(0)
    cases: dict[str, Case] = {}

    def make(name: str, what: str, specs: list[dict], packed: bool = False) -> Case:
        case = Case(name, what, packed=packed)
        t0, parts = 0, []
        for i, spec in enumerate(specs):
            spec = dict(spec)
            n = spec.pop("n")
            path = DATA / name / f"f{i}.nc"
            fresh = write or not path.exists()
            if spec.pop("absent", False):
                if fresh:
                    _write(path, None, chunks=None)
                case.absent.append(i)
                parts.append(None)
            else:
                data = spec.pop("data", None)
                if fresh:
                    _write(path, data if data is not None else _field(rng, t0, n), **spec)
                raw = _raw(path)
                if "scale" in spec:
                    raw = raw.astype("float64") * spec["scale"] + spec["offset"]
                parts.append(raw)
            case.files.append(path)
            t0 += n
        ref = next(p for p in parts if p is not None)
        with h5py.File(next(f for i, f in enumerate(case.files) if i not in case.absent), "r") as h:
            fill = h["v"].fillvalue  # an absent file reads as the fill value the others declare
        dtype = np.result_type(*[p.dtype for p in parts if p is not None])
        case.truth = np.concatenate([np.full((4, NY, NX), fill, dtype) if p is None else p.astype(dtype) for p in parts])
        cases[name] = case
        return case

    z = dict(zlib=True, complevel=1, shuffle=False)
    zs = dict(zlib=True, complevel=1, shuffle=True)
    whole = (1, NY, NX)

    make("codec_eras", "#399 case 1: gzip, then shuffle + gzip, then uncompressed", [
        dict(n=6, chunks=whole, **z), dict(n=6, chunks=whole, **zs), dict(n=6, chunks=whole)])
    make("file_lengths", "#399 case 2: monthly files of 31, 28 and 31 days, each one chunk", [
        dict(n=31, chunks=(31, NY, NX), **z), dict(n=28, chunks=(28, NY, NX), **z), dict(n=31, chunks=(31, NY, NX), **z)])
    make("padded_last_chunk", "#399 case 3: 10 steps in chunks of 4, the last chunk stored whole", [
        dict(n=10, chunks=(4, NY, NX), **z), dict(n=10, chunks=(4, NY, NX), **z), dict(n=10, chunks=(4, NY, NX), **z)])
    make("packing_per_file", "#399 case 4: int16 with a scale_factor and add_offset of each file's own", [
        dict(n=6, chunks=whole, dtype="i2", scale=0.01, offset=0.0, **z),
        dict(n=6, chunks=whole, dtype="i2", scale=0.02, offset=5.0, **z),
        dict(n=6, chunks=whole, dtype="i2", scale=0.005, offset=-3.0, **z)], packed=True)
    make("data_types", "#399 case 5: int16 files followed by float32 files", [
        dict(n=6, chunks=whole, dtype="i2", data=np.round(_field(rng, 0, 6)), **z),
        dict(n=6, chunks=whole, dtype="f4", **z)])
    make("chunk_shape_change", "chunk shape changes off the joined axis (Daymet North America in 2020)", [
        dict(n=8, chunks=(1, 12, 15), **z), dict(n=8, chunks=(4, 8, 10), **zs)])
    make("fill_value_differs", "files leave chunks unwritten and declare different fill values", [
        dict(n=8, chunks=(2, NY, NX), fill=-9999.0, unwritten_from=6, **z),
        dict(n=8, chunks=(2, NY, NX), fill=-1.0, unwritten_from=4, **z)])
    make("variable_absent", "one file does not have the variable", [
        dict(n=4, chunks=whole, **z), dict(n=4, absent=True), dict(n=4, chunks=whole, **z)])
    make("layouts_alternate", "two layouts that alternate file by file, 12 files", [
        dict(n=2, chunks=whole, **(z if i % 2 == 0 else zs)) for i in range(12)])
    make("packing_per_chunk", "every chunk packed with its own scale and offset (as GRIB packs messages), 60 files", [
        dict(n=1, chunks=whole, dtype="i2", scale=0.01 + 0.0005 * i, offset=float(i % 7) - 3, **z) for i in range(60)], packed=True)
    return cases


if __name__ == "__main__":
    for case in build(write=True).values():
        print(f"{case.name:20s} {len(case.files):3d} files  {case.truth.shape}  {case.truth.dtype}  {case.what}")
