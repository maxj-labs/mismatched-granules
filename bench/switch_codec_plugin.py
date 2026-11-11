"""A per-chunk `switch` codec, as a plugin: no change to zarr-python or to the store.

The array is an ordinary Zarr array with one manifest of chunks. Its only codec holds
several codec chains ("cases"), each covering a run of chunks along one axis, and picks
the chain from the chunk's grid coordinates. That is design 2 of zarr-specs#399 with the
per-chunk context run-length encoded in `zarr.json`, so nothing is stored per chunk.

A case may also give an `encoded_chunk_shape`: the shape the stored chunk decodes to when
it is larger than the chunk's place in the grid (HDF5 stores a file's last chunk whole).
The codec then crops it, which is what zarr-extensions PR #75 asks the chunk grid to say.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.core.array_spec import ArraySpec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.common import parse_named_configuration
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import ByteGetter
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.indexing import SelectorTuple


@lru_cache(maxsize=256)
def _pipeline(codecs_json: str, spec: ArraySpec) -> BatchedCodecPipeline:
    codecs = parse_codecs(json.loads(codecs_json))
    return BatchedCodecPipeline.from_codecs(codecs).evolve_from_array_spec(spec)


@lru_cache(maxsize=256)
def _partial_decoder(codecs_json: str, spec: ArraySpec) -> Any:
    codecs = parse_codecs(json.loads(codecs_json))
    if len(codecs) == 1 and isinstance(codecs[0], ArrayBytesCodecPartialDecodeMixin):
        return codecs[0].evolve_from_array_spec(spec)
    return None


@dataclass(frozen=True)
class SwitchCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Decode each chunk with the codec chain of the run of chunks it belongs to.

    Parameters
    ----------
    axis
        The axis along which the runs lie.
    cases
        The runs, in order. Each is ``{"chunks": n, "codecs": [...]}``, with ``n`` the
        number of chunks along `axis` that the run covers, and optionally
        ``"encoded_chunk_shape"``.
    """

    is_fixed_size = False

    axis: int
    cases_json: str

    def __init__(self, *, axis: int, cases: Any) -> None:
        cases = json.loads(cases) if isinstance(cases, str) else list(cases)
        for case in cases:
            if not {"chunks", "codecs"} <= set(case) <= {"chunks", "codecs", "encoded_chunk_shape", "fill_value"}:
                raise ValueError(f"A case has 'chunks', 'codecs' and optionally 'encoded_chunk_shape' and 'fill_value'. Got {case!r}.")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "cases_json", json.dumps(cases, sort_keys=True))

    @property
    def cases(self) -> list[dict[str, Any]]:
        return json.loads(self.cases_json)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _, configuration = parse_named_configuration(data, "switch")
        return cls(axis=configuration["axis"], cases=configuration["cases"])

    def to_dict(self) -> dict[str, Any]:
        return {"name": "switch", "configuration": {"axis": self.axis, "cases": self.cases}}

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError

    def _case(self, chunk_key: str, ndim: int) -> dict[str, Any]:
        # The codec is not told which chunk it decodes, so it reads the grid coordinates
        # back from the chunk's key (the `default` encoding with the `/` separator).
        index = int(chunk_key.split("/")[-ndim:][self.axis])
        start = 0
        for case in self.cases:
            start += case["chunks"]
            if index < start:
                return case
        raise IndexError(f"Chunk {chunk_key!r} lies beyond the last case.")

    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        case = self._case(byte_getter.path, len(chunk_spec.shape))  # type: ignore[attr-defined]
        # an entry of `null` means "as large as the chunk's place in the grid"
        encoded_shape = tuple(
            n if e is None else e
            for e, n in zip(case.get("encoded_chunk_shape") or chunk_spec.shape, chunk_spec.shape, strict=True)
        )
        # A case may carry its own fill value: what its granules read where they store
        # nothing (HDF5's fill value, which need not be the array's).
        own_fill = "fill_value" in case
        fill = chunk_spec.dtype.from_json_scalar(case["fill_value"], zarr_format=3) if own_fill else chunk_spec.fill_value
        spec = ArraySpec(
            shape=encoded_shape,
            dtype=chunk_spec.dtype,
            fill_value=fill,
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )
        inner = _partial_decoder(json.dumps(case["codecs"], sort_keys=True), spec)
        if inner is not None:
            # The chain is one codec that reads parts of a chunk by itself (sharding):
            # hand it the selection, so it fetches only the inner chunks it needs. The
            # chunk's cell is the leading corner of the stored shape, so the selection
            # means the same thing in both. It fills missing inner chunks from `spec`.
            out = await inner._decode_partial_single(byte_getter, selection, spec)
            return self._filled(selection, chunk_spec, fill) if out is None and own_fill else out
        chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)
        if chunk_bytes is None:
            # Returning None makes the reader use the array's fill value.
            return self._filled(selection, chunk_spec, fill) if own_fill else None
        pipeline = _pipeline(json.dumps(case["codecs"], sort_keys=True), spec)
        (decoded,) = await pipeline.decode([(chunk_bytes, spec)])
        assert decoded is not None
        if encoded_shape != chunk_spec.shape:
            decoded = decoded[tuple(slice(0, n) for n in chunk_spec.shape)]
        return decoded[selection]

    @staticmethod
    def _filled(selection: SelectorTuple, chunk_spec: ArraySpec, fill: Any) -> NDBuffer:
        """What a selection of a chunk that is not stored reads as: all `fill`."""
        from zarr.core.chunk_grids import ChunkGrid
        from zarr.core.indexing import get_indexer

        shape = chunk_spec.shape
        indexer = get_indexer(selection, shape=shape, chunk_grid=ChunkGrid.from_sizes(shape, list(shape)))
        out = chunk_spec.prototype.nd_buffer.create(
            shape=indexer.shape, dtype=chunk_spec.dtype.to_native_dtype(), order=chunk_spec.order, fill_value=fill
        )
        return out.reshape(indexer.sel_shape) if hasattr(indexer, "sel_shape") else out

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError("The switch codec needs the chunk's key, so it must be the array's only codec.")

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError("This sketch of the switch codec is read-only.")


register_codec("switch", SwitchCodec)


def rle_cases(codecs_per_file: list[list[dict[str, Any]]], chunks_per_file: list[int]) -> list[dict[str, Any]]:
    """Run-length encode per-file codec chains into cases."""
    cases: list[dict[str, Any]] = []
    for codecs, n in zip(codecs_per_file, chunks_per_file, strict=True):
        if cases and cases[-1]["codecs"] == codecs:
            cases[-1]["chunks"] += n
        else:
            cases.append({"chunks": n, "codecs": codecs})
    return cases


__all__ = ["SwitchCodec", "rle_cases", "np"]
