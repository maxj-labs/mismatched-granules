"""Plan one Zarr array from per-granule layouts that do not match, and write it to Icechunk.

Two strategies, chosen from the layouts:

- "switch": the granules agree on their chunk shape off the joined axis. Every source
  chunk stays one Zarr chunk on a rectilinear grid; a position-based `switch` codec picks
  the codec chain per run of granules and crops chunks that are stored larger than their
  place in the grid. Needs no change to Icechunk.
- "shards": the chunk shapes disagree. Each Zarr chunk is a virtual shard (a literal
  `sharding_indexed` index followed by the source chunks it holds), sized to the least
  common multiple of the layouts where that is small, and to the whole extent otherwise.
  Needs virtual references made of several parts (the Icechunk branch).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MISSING = 2**64 - 1


@dataclass
class Granule:
    """One granule's copy of the variable, as VirtualiZarr parsed it."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    codecs: list[dict]
    dtype: np.dtype
    fill_value: Any
    paths: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    fill_json: Any = None  # the fill value as Zarr writes it in JSON
    sentinel: Any = None  # the value that marks "no data" in this granule (CF `_FillValue`), as JSON

    @property
    def stores_everything(self) -> bool:
        return bool((self.paths != "").all())

    @classmethod
    def from_manifest_array(cls, ma: Any, *, stack: bool, rename=lambda p: p, sentinel: Any = None) -> "Granule":
        md, m = ma.metadata, ma.manifest
        shape, chunks = tuple(md.shape), tuple(md.chunks)
        paths, offsets, lengths = np.asarray(m._paths), np.asarray(m._offsets), np.asarray(m._lengths)
        if not shape:
            raise NotImplementedError("scalar variable")
        if stack:  # a new leading axis of length 1
            shape, chunks = (1, *shape), (1, *chunks)
            paths, offsets, lengths = paths[None], offsets[None], lengths[None]
        vrename = np.vectorize(lambda p: rename(p) if p else "", otypes=[object])
        fill_json = md.data_type.to_json_scalar(md.fill_value, zarr_format=3)
        return cls(shape, chunks, list(md.to_dict()["codecs"]), ma.dtype, md.fill_value, vrename(paths), offsets, lengths,
                   fill_json=fill_json, sentinel=sentinel)


def edges(n: int, c: int) -> list[int]:
    return [c] * (n // c) + ([n % c] if n % c else [])


@dataclass
class Plan:
    strategy: str
    shape: tuple[int, ...]
    chunk_grid: list[list[int]]
    cases: list[dict]
    refs: list[dict] = field(default_factory=list)  # index, location, offset, length, prefix, more_ranges
    source_chunks: int = 0
    notes: dict = field(default_factory=dict)
    fill_json: Any = None
    sentinel: Any = None


def _most_common(values: list) -> Any:
    keys = [json.dumps(v) for v in values]
    return json.loads(max(set(keys), key=keys.count))


def plan(granules: list[Granule], *, per_granule_fill: bool = True, remap_sentinels: bool = True) -> Plan:
    """
    per_granule_fill
        Give a run of granules its own fill value when it differs from the array's and a
        granule of the run leaves chunks unstored. (Where every chunk is stored, the fill
        value is never read, so the case table does not grow.)
    remap_sentinels
        Where a granule marks "no data" with another value than the array does, rewrite
        that value on read with a `cast_value` codec at the head of its chain.
    """
    g0 = granules[0]
    ndim = len(g0.shape)
    if any(g.dtype != g0.dtype for g in granules):
        raise NotImplementedError(f"data types differ: {sorted({str(g.dtype) for g in granules})}")
    if any(g.shape[1:] != g0.shape[1:] for g in granules):
        raise NotImplementedError(f"extents differ off the joined axis: {sorted({g.shape[1:] for g in granules})}")
    shape = (sum(g.shape[0] for g in granules), *g0.shape[1:])
    notes = {
        "distinct_chunk_shapes": sorted({g.chunks for g in granules}),
        "distinct_codec_chains": len({str(g.codecs) for g in granules}),
        "granules_ending_in_a_cut_chunk": sum(g.shape[0] % g.chunks[0] != 0 for g in granules),
        "fill_values": sorted({repr(g.fill_value) for g in granules}),
        "missing_source_chunks": int(sum((g.paths == "").sum() for g in granules)),
    }
    cases: list[dict] = []
    array_fill = _most_common([g.fill_json for g in granules])
    sentinels = [g.sentinel for g in granules if g.sentinel is not None]
    array_sentinel = _most_common(sentinels) if sentinels else None
    notes["granules_with_their_own_fill"] = 0
    notes["granules_with_a_remapped_sentinel"] = 0

    def extras(g: Granule) -> tuple[dict, list[dict]]:
        """A granule's case-level fill value, and the codecs to put before its chain."""
        case: dict = {}
        if per_granule_fill and json.dumps(g.fill_json) != json.dumps(array_fill) and not g.stores_everything:
            case["fill_value"] = g.fill_json
            notes["granules_with_their_own_fill"] += 1
        head: list[dict] = []
        if remap_sentinels and g.sentinel is not None and json.dumps(g.sentinel) != json.dumps(array_sentinel):
            head.append({"name": "cast_value", "configuration": {
                "data_type": str(g.dtype), "scalar_map": {"decode": [[g.sentinel, array_sentinel]], "encode": [[array_sentinel, g.sentinel]]}}})
            notes["granules_with_a_remapped_sentinel"] += 1
        return case, head

    def add_case(n: int, case: dict) -> None:
        if cases and {k: v for k, v in cases[-1].items() if k != "chunks"} == case:
            cases[-1]["chunks"] += n
        else:
            cases.append({"chunks": n, **case})

    if all(g.chunks[1:] == g0.chunks[1:] for g in granules):
        grid = [[], *[edges(shape[d], g0.chunks[d]) for d in range(1, ndim)]]
        out = Plan("switch", shape, grid, cases, notes=notes, fill_json=array_fill, sentinel=array_sentinel)
        base = 0
        for g in granules:
            along = edges(g.shape[0], g.chunks[0])
            whole = g.shape[0] % g.chunks[0] == 0
            own, head = extras(g)
            add_case(len(along), {"codecs": head + g.codecs, "encoded_chunk_shape": [None if whole else g.chunks[0], *g.chunks[1:]], **own})
            grid[0] += along
            for idx in np.ndindex(g.paths.shape):
                if g.paths[idx]:
                    out.refs.append({"index": [base + idx[0], *idx[1:]], "location": g.paths[idx],
                                     "offset": int(g.offsets[idx]), "length": int(g.lengths[idx])})
            out.source_chunks += int((g.paths != "").sum())
            base += len(along)
        return out

    # shards
    lcm0 = math.lcm(*[g.chunks[0] for g in granules])
    shard0 = lcm0 if lcm0 <= min(g.shape[0] for g in granules) else None  # None: one shard per granule along the axis
    shard = [0] * ndim
    for d in range(1, ndim):
        lcm = math.lcm(*[g.chunks[d] for g in granules])
        shard[d] = lcm if lcm <= shape[d] else shape[d]
    grid = [[], *[edges(shape[d], shard[d]) for d in range(1, ndim)]]
    out = Plan("shards", shape, grid, cases, notes=notes, fill_json=array_fill, sentinel=array_sentinel)
    notes["shard_shape"] = [shard0 or "one per granule", *shard[1:]]
    base = 0
    for g in granules:
        length = [shard0 or g.shape[0], *shard[1:]]
        encoded = [math.ceil(length[d] / g.chunks[d]) * g.chunks[d] for d in range(ndim)]
        per_shard = tuple(encoded[d] // g.chunks[d] for d in range(ndim))
        along = edges(g.shape[0], length[0])
        n_shards = (len(along), *[len(grid[d]) for d in range(1, ndim)])
        index_nbytes = 16 * int(np.prod(per_shard))
        own, head = extras(g)
        sharding = {"name": "sharding_indexed", "configuration": {
            "chunk_shape": list(g.chunks), "codecs": head + g.codecs,
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_location": "start"}}
        add_case(len(along), {"codecs": [sharding], "encoded_chunk_shape": encoded, **own})
        grid[0] += along
        for s in np.ndindex(n_shards):
            index = np.full((*per_shard, 2), MISSING, dtype="<u8")
            parts, pos, location = [], index_nbytes, None
            for inner in np.ndindex(per_shard):
                src = tuple(s[d] * per_shard[d] + inner[d] for d in range(ndim))
                if all(x < n for x, n in zip(src, g.paths.shape)) and g.paths[src]:
                    n = int(g.lengths[src])
                    index[inner] = (pos, n)
                    parts.append((int(g.offsets[src]), n))
                    location = g.paths[src]
                    pos += n
            if parts:
                out.refs.append({"index": [base + s[0], *s[1:]], "location": location, "offset": parts[0][0], "length": parts[0][1],
                                 "prefix": index.tobytes(), "more_ranges": [(None, o, n) for o, n in parts[1:]] or None})
                out.source_chunks += len(parts)
        base += len(along)
    return out


def write(store: Any, path: str, p: Plan, dtype: np.dtype, fill_value: Any = None) -> Any:
    """Create the array at `path` in an Icechunk store and set its virtual references.

    The array's fill value is the plan's (the most common among the granules) unless
    `fill_value` is given.
    """
    if fill_value is None:
        fill_value = np.dtype(dtype).type(p.fill_json) if not isinstance(p.fill_json, str) else np.dtype(dtype).type(float(p.fill_json))
    import icechunk
    import zarr

    from switch_codec import SwitchCodec

    arr = zarr.create_array(
        store, name=path, shape=p.shape, chunks=p.chunk_grid, dtype=dtype, fill_value=fill_value,
        serializer=SwitchCodec(axis=0, cases=p.cases), compressors=None, filters=None,
    )
    failed = store.set_virtual_refs(path, [icechunk.VirtualChunkSpec(**r) for r in p.refs], validate_containers=True)
    if failed:
        raise RuntimeError(f"{len(failed)} references were refused")
    return arr
