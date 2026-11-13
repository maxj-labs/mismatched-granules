"""Codecs and store wrappers that exist only to evaluate proposals not in the branches.

- `switch_x`: the switch codec's case table, with the chunk's case found in the other
  ways proposed in zarr-specs#399:
    index_prefix    the store puts the chunk's grid index in front of its bytes (design 2.1)
    sidecar         the codec reads the case of every chunk from another array (design 2.2)
    context         the store answers `get_chunk_context(key)` (design 3, "Zarr v3.2")
    key_parameters  a key in front of the chunk, followed by numbers that fill in the
                    case's open parameters (refkv PR #73's design, on Icechunk)
- `zarr_reference`: each chunk is a reference to another array (zarr-specs#288, jbms).
- encoded chunk shapes declared by the chunk grid (zarr-extensions PR #75), emulated.
- `DomainView`: an array stored with its padding, plus a declared valid domain
  (zarr-extensions#74, d-v-b).
"""

from __future__ import annotations

import json
import posixpath
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import zarr
from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.codecs.switch import SwitchCodec, _after_key, _partial_decoder, _pipeline
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.registry import register_codec
from zarr.storage import WrapperStore

# ---------------------------------------------------------------- switch_x


def _fill_parameters(codecs_json: str, values: tuple[float, ...]) -> str:
    def fill(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {"parameter"}:
                return values[node["parameter"]]
            return {k: fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        return node

    return json.dumps(fill(json.loads(codecs_json)), sort_keys=True)


@dataclass(frozen=True)
class SwitchXCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """`config_json`: {"select", "cases", "axis", "ndim", "sidecar", "key_bytes"}.

    Cases are listed once each. `index_prefix` adds `runs`, [case, chunks] pairs along
    `axis`; the other modes name a case by its place in the list, or by `key`.
    """

    config_json: str
    is_fixed_size = False
    needs_chunk_coords = True  # used by `sidecar` only

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(config_json=json.dumps(data["configuration"], sort_keys=True))

    def to_dict(self) -> dict:
        return {"name": "switch_x", "configuration": json.loads(self.config_json)}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    def validate(self, *, shape: Any, dtype: Any, chunk_grid: Any) -> None:
        """Where the table is a list of runs in the metadata, it must cover the array's
        chunks, as the position mode of the real codec checks. (A table kept in a side
        array or by the store cannot be checked here: its length is not in the metadata.)"""
        config = json.loads(self.config_json)
        if config["select"] != "index_prefix":
            return
        from zarr.core.chunk_grids import ChunkGrid

        sizes = getattr(chunk_grid, "chunk_shapes", None) or chunk_grid.chunk_shape
        chunks = ChunkGrid.from_sizes(shape, sizes).grid_shape[config["axis"]]
        covered = sum(n for _, n in config["runs"])
        if covered != chunks:
            raise ValueError(f"The table covers {covered} chunks along axis {config['axis']}, but the array has {chunks}.")

    @property
    def _state(self) -> dict:
        state = self.__dict__.get("_cached")
        if state is None:
            config = json.loads(self.config_json)
            cases = config["cases"]
            # the real codec resolves a case (stored shape, fill value, chain) by its index
            inner = SwitchCodec(
                select="key", key_bytes=4,
                cases=[{"key": i.to_bytes(4, "big").hex(), **{k: v for k, v in c.items() if k not in ("chunks", "key", "parameters")}}
                       for i, c in enumerate(cases)],
            )
            # `runs`: [case, chunks] pairs along the axis, each case listed once in `cases`
            ends, run_case, total = [], [], 0
            for i, n in config.get("runs", []):
                total += n
                ends.append(total)
                run_case.append(i)
            state = {
                "config": config, "inner": inner, "ends": ends, "run_case": run_case, "numbers": None,
                "by_key": {bytes.fromhex(c["key"]): i for i, c in enumerate(cases) if "key" in c},
                "parameters": [c.get("parameters", 0) for c in cases],
                "table": None,
            }
            object.__setattr__(self, "_cached", state)
        return state

    async def _find(self, byte_getter: Any, chunk_spec: ArraySpec) -> tuple[int, int, tuple[float, ...]] | None:
        """The chunk's case, how many leading bytes are not chunk data, and its parameters."""
        state = self._state
        config, proto = state["config"], chunk_spec.prototype
        select = config["select"]
        if select == "index_prefix":
            n = 8 * config["ndim"]
            head = await byte_getter.get(proto, RangeByteRequest(0, n))
            if head is None:
                return None
            coords = np.frombuffer(head.to_bytes(), dtype="<u8")
            return state["run_case"][bisect_right(state["ends"], int(coords[config["axis"]]))], n, ()
        if select == "sidecar":
            if state["table"] is None:
                array_path = byte_getter.path.rsplit("/c/", 1)[0]
                sidecar = await zarr.api.asynchronous.open_array(store=byte_getter.store, path=array_path + config["sidecar"], mode="r")
                state["table"] = np.asarray(await sidecar.getitem(Ellipsis))
                if config.get("sidecar_parameters"):
                    numbers = await zarr.api.asynchronous.open_array(store=byte_getter.store, path=array_path + config["sidecar_parameters"], mode="r")
                    state["numbers"] = np.asarray(await numbers.getitem(Ellipsis))
            assert chunk_spec.chunk_coords is not None
            i = chunk_spec.chunk_coords[config["axis"]]
            index = int(state["table"][i])
            n = state["parameters"][index]
            return index, 0, tuple(state["numbers"][i, :n].tolist()) if n else ()
        if select == "context":
            context = await byte_getter.store.get_chunk_context(byte_getter.path)
            index = int.from_bytes(context[:4], "little")
            return index, 0, tuple(np.frombuffer(context[4:], dtype="<f8").tolist()[: state["parameters"][index]])
        if select == "key_parameters":
            k = config.get("key_bytes", 8)
            head = await byte_getter.get(proto, RangeByteRequest(0, k))
            if head is None:
                return None
            index = state["by_key"][head.to_bytes()]
            n = state["parameters"][index]
            values: tuple[float, ...] = ()
            if n:
                more = await byte_getter.get(proto, RangeByteRequest(k, k + 8 * n))
                values = tuple(np.frombuffer(more.to_bytes(), dtype="<f8").tolist())
            return index, k + 8 * n, values
        raise ValueError(select)

    async def _decode_partial_single(self, byte_getter: Any, selection: Any, chunk_spec: ArraySpec) -> NDBuffer | None:
        found = await self._find(byte_getter, chunk_spec)
        if found is None:
            return None
        index, skip, values = found
        spec, codecs_json, own_fill = self._state["inner"]._resolve(index, chunk_spec)
        if values:
            codecs_json = _fill_parameters(codecs_json, values)
        source = _after_key(byte_getter, skip) if skip else byte_getter
        inner = _partial_decoder(codecs_json, spec)
        if inner is not None:
            out = await inner._decode_partial_single(source, selection, spec)
            if out is None and own_fill:
                return SwitchCodec._filled(selection, chunk_spec, spec.fill_value)
            return out
        chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)
        if chunk_bytes is not None and skip:  # drop the header here: a chunk may be its header alone
            chunk_bytes = chunk_spec.prototype.buffer.from_bytes(chunk_bytes.to_bytes()[skip:])
        if chunk_bytes is None or len(chunk_bytes) == 0:
            return SwitchCodec._filled(selection, chunk_spec, spec.fill_value) if own_fill else None
        (decoded,) = await _pipeline(codecs_json, spec).decode([(chunk_bytes, spec)])
        if spec.shape != chunk_spec.shape:
            decoded = decoded[tuple(slice(0, n) for n in chunk_spec.shape)]
        return decoded[selection]

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError("switch_x reads through the store")

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError("switch_x is read-only")


register_codec("switch_x", SwitchXCodec)


def _slice_range(data: bytes, byte_range: Any) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix :]
    raise TypeError(byte_range)


class IndexPrefixStore(WrapperStore[Any]):
    """A store that puts each chunk's grid index (uint64 little-endian per axis) in front
    of its bytes, for the arrays named in `paths`. Nothing is stored for it: the index
    comes from the key. A chunk that is not stored reads as its index alone.

    This is what "Icechunk prepends chunk index info on read" would do, done in Python.
    """

    def __init__(self, store: Any, paths: tuple[str, ...] = ()) -> None:
        super().__init__(store)
        self._paths = tuple(paths)

    def _with_store(self, store: Any) -> Self:
        return type(self)(store, self._paths)

    def _header(self, key: str) -> bytes | None:
        for p in self._paths:
            marker = f"{p}/c/"
            if key.startswith(marker):
                return np.array([int(i) for i in key[len(marker) :].split("/")], dtype="<u8").tobytes()
        return None

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        header = self._header(key)
        if header is None:
            return await self._store.get(key, prototype, byte_range)
        h = len(header)
        if isinstance(byte_range, RangeByteRequest) and byte_range.end <= h:
            return prototype.buffer.from_bytes(header[byte_range.start : byte_range.end])
        if isinstance(byte_range, RangeByteRequest) and byte_range.start >= h:
            return await self._store.get(key, prototype, RangeByteRequest(byte_range.start - h, byte_range.end - h))
        if isinstance(byte_range, OffsetByteRequest) and byte_range.offset >= h:
            body = await self._store.get(key, prototype, OffsetByteRequest(byte_range.offset - h) if byte_range.offset > h else None)
            return body if body is not None else prototype.buffer.from_bytes(b"")
        if isinstance(byte_range, SuffixByteRequest):
            return await self._store.get(key, prototype, byte_range)
        body = await self._store.get(key, prototype)
        return prototype.buffer.from_bytes(_slice_range(header + (b"" if body is None else body.to_bytes()), byte_range))

    get_ranges = Store.get_ranges  # many ranges at once go through `get` above

    async def get_partial_values(self, prototype: Any, key_ranges: Any) -> Any:
        return [await self.get(k, prototype, r) for k, r in key_ranges]


class ContextStore(WrapperStore[Any]):
    """A store with the extra method design 3 asks for: `get_chunk_context(key)`.

    The context of every chunk of an array is held here in memory, loaded from a table
    array in the store when the wrapper is made. A store that kept it in its manifest
    would answer from there.
    """

    def __init__(self, store: Any, tables: dict | None = None) -> None:
        super().__init__(store)
        self._tables = tables or {}

    def _with_store(self, store: Any) -> Self:
        return type(self)(store, self._tables)

    @classmethod
    def load(cls, store: Any, path: str, axis: int, sidecar: str, parameters: str | None = None) -> Self:
        ids = np.asarray(zarr.open_array(store, path=path + sidecar, mode="r")[...])
        values = np.asarray(zarr.open_array(store, path=path + parameters, mode="r")[...]) if parameters else None
        return cls(store, {path: (axis, ids, values)})

    async def get_chunk_context(self, key: str) -> bytes:
        path, _, coords = key.rpartition("/c/")
        axis, ids, values = self._tables[path]
        i = int(coords.split("/")[axis])
        extra = b"" if values is None else np.asarray(values[i], dtype="<f8").tobytes()
        return int(ids[i]).to_bytes(4, "little") + extra


# ---------------------------------------------------------------- zarr_reference


@dataclass(frozen=True)
class ZarrReferenceCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Each chunk's bytes are JSON `{"path": <another array, relative>}`; the chunk reads
    as that array. One chunk per piece on a rectilinear grid."""

    is_fixed_size = False

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls()

    def to_dict(self) -> dict:
        return {"name": "zarr_reference", "configuration": {}}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    async def _decode_partial_single(self, byte_getter: Any, selection: Any, chunk_spec: ArraySpec) -> NDBuffer | None:
        raw = await byte_getter.get(prototype=chunk_spec.prototype)
        if raw is None:
            return None
        reference = json.loads(raw.to_bytes())
        array_path = byte_getter.path.rsplit("/c/", 1)[0]
        target = posixpath.normpath(posixpath.join(array_path, reference["path"]))
        opened = self.__dict__.setdefault("_opened", {})
        if target not in opened:
            opened[target] = await zarr.api.asynchronous.open_array(store=byte_getter.store, path=target, mode="r")
        # read the smallest box of the other array that holds the selection, then pick from it
        box, inside = [], []
        for sel in selection:
            if isinstance(sel, slice):
                box.append(sel)
                inside.append(slice(None))
            elif isinstance(sel, (int, np.integer)):
                box.append(slice(int(sel), int(sel) + 1))
                inside.append(0)
            else:
                index = np.asarray(sel)
                box.append(slice(int(index.min()), int(index.max()) + 1))
                inside.append(index - int(index.min()))
        data = np.asarray(await opened[target].getitem(tuple(box)))
        return chunk_spec.prototype.nd_buffer.from_numpy_array(data)[tuple(inside)]

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError("zarr_reference reads through the store")

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError("zarr_reference is read-only")


register_codec("zarr_reference", ZarrReferenceCodec)

# ---------------------------------------------------------------- encoded chunk shapes in the grid

_ENCODED_ATTR = "encoded_chunk_shapes"


def enable_grid_encoded_chunk_shapes() -> None:
    """Make zarr-python's rectilinear grid decode each chunk at a declared encoded size
    and keep only its logical extent, when the array declares one.

    zarr-extensions PR #75 puts the declaration in the chunk grid's configuration; here
    it is read from the array attribute `encoded_chunk_shapes` (a list per axis, or
    null), so that zarr-python's metadata classes need no change.
    """
    from zarr.core import chunk_grids

    if getattr(chunk_grids.ChunkGrid, "_eval_patched", False):
        return
    original = chunk_grids.ChunkGrid.from_metadata.__func__

    class EncodedVaryingDimension(chunk_grids.VaryingDimension):
        def chunk_size(self, chunk_ix: int) -> int:
            return self.__dict__["encoded"][chunk_ix]

    def from_metadata(cls: Any, metadata: Any) -> Any:
        grid = original(cls, metadata)
        encoded = getattr(metadata, "attributes", {}).get(_ENCODED_ATTR)
        if not encoded:
            return grid
        dims = []
        for dim, sizes in zip(grid.dimensions, encoded):
            if sizes is None:
                dims.append(dim)
                continue
            new = EncodedVaryingDimension(dim.edges, dim.extent)
            object.__setattr__(new, "encoded", tuple(sizes))
            dims.append(new)
        return cls(dimensions=tuple(dims))

    chunk_grids.ChunkGrid.from_metadata = classmethod(from_metadata)
    chunk_grids.ChunkGrid._eval_patched = True


# ---------------------------------------------------------------- a declared valid domain


class DomainView:
    """Reads an array stored with its padding through the valid ranges it declares in
    the attribute `domain` (a list of [start, stop) along axis 0)."""

    def __init__(self, array: zarr.Array) -> None:
        self.array = array
        self.ranges = [tuple(r) for r in array.attrs["domain"]]
        sizes = [b - a for a, b in self.ranges]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.shape = (int(self.offsets[-1]), *array.shape[1:])

    def __getitem__(self, selection: tuple) -> np.ndarray:
        first, rest = selection[0], tuple(selection[1:])
        if not isinstance(first, slice):
            i = int(first)
            k = int(np.searchsorted(self.offsets, i, side="right") - 1)
            return self.array[(self.ranges[k][0] + i - int(self.offsets[k]), *rest)]
        start, stop, _ = first.indices(self.shape[0])
        parts = []
        for k, (a, _) in enumerate(self.ranges):
            lo, hi = max(start, int(self.offsets[k])), min(stop, int(self.offsets[k + 1]))
            if lo < hi:
                parts.append(self.array[(slice(a + lo - int(self.offsets[k]), a + hi - int(self.offsets[k])), *rest)])
        return np.concatenate(parts)
