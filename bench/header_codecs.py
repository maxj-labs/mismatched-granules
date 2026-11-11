"""Codecs that read a chunk's encoding from a header at the start of its bytes, and a
store wrapper that puts the header there.

- `header_switch`: the `switch` codec of d-v-b/refkv PR #73 (see vendor/), under another
  name so it can sit beside the position-based `switch` of switch_codec.py. A chunk is an
  8-byte key naming one of the codec chains in `zarr.json`, then the bytes that chain decodes.
- `conditional`: the zarr-extensions `conditional` codec. A chunk is a bitmask, one bit per
  wrapped bytes-to-bytes codec, then the bytes; a codec whose bit is 0 is skipped.
- `PrefixStore`: stands in for a store that keeps per-chunk context in its manifest and
  prepends it on read (NASA-IMPACT/veda-odd#498).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Self

from zarr.abc.codec import ArrayBytesCodec, BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_pipeline_class, register_codec
from zarr.storage import WrapperStore

KEY_BYTES = 8


def case_key(case: dict) -> bytes:
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(canonical).digest()[:KEY_BYTES]


@lru_cache(maxsize=256)
def _chain(codecs_json: str, spec: ArraySpec):
    return get_pipeline_class().from_codecs(parse_codecs(json.loads(codecs_json))).evolve_from_array_spec(spec)


@dataclass(frozen=True)
class HeaderSwitchCodec(ArrayBytesCodec):
    cases: tuple = ()  # (key hex, codecs as JSON)
    is_fixed_size = False

    @classmethod
    def from_cases(cls, cases: list[dict]) -> Self:
        return cls(cases=tuple((case_key(c).hex(), json.dumps(c["codecs"])) for c in cases))

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(cases=tuple((k, json.dumps(c["codecs"])) for k, c in data["configuration"]["cases"].items()))

    def to_dict(self) -> dict:
        return {"name": "header_switch", "configuration": {"cases": {
            k: {"codecs": json.loads(c), "parameters": []} for k, c in self.cases}}}

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raw = chunk_bytes.to_bytes()
        codecs = dict(self.cases)[raw[:KEY_BYTES].hex()]
        (out,) = await _chain(codecs, chunk_spec).decode(
            [(chunk_spec.prototype.buffer.from_bytes(raw[KEY_BYTES:]), chunk_spec)])
        return out

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        raise NotImplementedError("read-only sketch")

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class ConditionalCodec(BytesBytesCodec):
    codecs_json: str = "[]"
    header_bits: int = 8
    is_fixed_size = False

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        c = data["configuration"]
        return cls(codecs_json=json.dumps(c["codecs"]), header_bits=c.get("header_bits", 8))

    def to_dict(self) -> dict:
        return {"name": "conditional", "configuration": {"codecs": json.loads(self.codecs_json), "header_bits": self.header_bits}}

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        raw = chunk_bytes.to_bytes()
        n = self.header_bits // 8
        mask = int.from_bytes(raw[:n], "little")
        data = chunk_spec.prototype.buffer.from_bytes(raw[n:])
        codecs = parse_codecs(json.loads(self.codecs_json))
        for i in reversed(range(len(codecs))):
            if mask >> i & 1:
                (data,) = await codecs[i].decode([(data, chunk_spec)])
        return data

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        raise NotImplementedError("read-only sketch")

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("header_switch", HeaderSwitchCodec)
register_codec("conditional", ConditionalCodec)


class PrefixStore(WrapperStore):
    """Prepends each chunk's header on read, from a run-length table along axis 0.

    `runs` is `[(number of chunks along axis 0, header bytes), ...]` for the array at `path`.
    """

    def __init__(self, store: Any, path: str = "", runs: list[tuple[int, bytes]] | None = None) -> None:
        super().__init__(store)
        self._path, self._runs = path, runs or []
        self._bounds, total = [], 0
        for n, _ in self._runs:
            total += n
            self._bounds.append(total)

    def _with_store(self, store: Any) -> Self:
        return type(self)(store, self._path, self._runs)

    async def get(self, key, prototype, byte_range=None):
        out = await self._store.get(key, prototype, byte_range)
        marker = f"{self._path}/c/"
        if out is None or not key.startswith(marker):
            return out
        index = int(key[len(marker):].split("/", 1)[0])
        for bound, (_, header) in zip(self._bounds, self._runs):
            if index < bound:
                return prototype.buffer.from_bytes(header + out.to_bytes())
        raise IndexError(key)
