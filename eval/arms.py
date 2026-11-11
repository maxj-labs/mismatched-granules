"""Every option, as a function that turns a case's files into something readable.

An option raises NotExpressible, with the reason, when it has no way to describe a case.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import icechunk
import numpy as np
import xarray as xr
import zarr
from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.concat import ConcatenatedManifestArray
from virtualizarr.manifests.plan import JoinedArrayPlan, plan_joined_array
from virtualizarr.writers.icechunk import _icechunk_takes_parts, _reported_chunks, _write_literals, write_virtual_variable_to_icechunk
from zarr.codecs import ArrayConcatCodec, SwitchCodec
from zarr.core.metadata.v3 import ArrayV3Metadata

import xcodecs
from cases import Case
from common import Built, NotExpressible, new_repo, open_store, store_bytes
from xcodecs import ContextStore, DomainView, IndexPrefixStore, SwitchXCodec, ZarrReferenceCodec

assert _icechunk_takes_parts(), "the evaluation needs the Icechunk branch (references with a prefix and several ranges)"
DIMS = ("time", "y", "x")


def _plan(pieces: list[ManifestArray], select: str = "position") -> JoinedArrayPlan:
    try:
        return plan_joined_array(pieces, axis=0, select=select)
    except NotImplementedError as err:
        raise NotExpressible(str(err)) from None


# compose.py sets this to drop leading chunks from what is written, as an operation on
# the manifest that knows nothing about the codec would.
DROP_CHUNKS = 0


def _refs(store: Any, name: str, refs: list[dict]) -> None:
    if DROP_CHUNKS and name == "v":
        refs = [{**r, "index": [r["index"][0] - DROP_CHUNKS, *r["index"][1:]]} for r in refs if r["index"][0] >= DROP_CHUNKS]
    failed = store.set_virtual_refs(name, [icechunk.VirtualChunkSpec(**r) for r in refs], validate_containers=True)
    assert not failed, failed


def _dropped(shape: tuple, grid: list) -> tuple[tuple, list]:
    if not DROP_CHUNKS:
        return tuple(shape), grid
    return (shape[0] - sum(grid[0][:DROP_CHUNKS]), *shape[1:]), [grid[0][DROP_CHUNKS:], *grid[1:]]


def _literals(store: Any, pieces: list[ManifestArray], header: Callable[[int], bytes] | None = None) -> None:
    """Chunks that store nothing in a file whose fill value is not the array's: written as
    the bytes that name their case and nothing else. `header` replaces the planner's key."""
    literals = _plan(pieces, select="key").literals
    if header is not None:
        literals = [{"index": lit["index"], "data": header(lit["index"][0]) + lit["data"][8:]} for lit in literals]
    literals = [{**lit, "index": [lit["index"][0] - DROP_CHUNKS, *lit["index"][1:]]} for lit in literals if lit["index"][0] >= DROP_CHUNKS]
    _write_literals(store, "v", literals)


def _create(store: Any, name: str, plan: JoinedArrayPlan, serializer: Any, **kwargs: Any) -> zarr.Array:
    shape, grid = _dropped(plan.shape, plan.chunk_grid)
    return zarr.create_array(
        store, name=name, shape=shape, chunks=grid, dtype=plan.dtype, fill_value=plan.fill_value,
        serializer=serializer, compressors=None, filters=None, dimension_names=DIMS, **kwargs,
    )


def _finish(arm: str, case: Case, repo: Any, path: Any, wrap: Callable[[Any], Any] | None = None, **info: Any) -> Built:
    def reopen() -> zarr.Array:
        store = open_store(path)
        return zarr.open_array(wrap(store) if wrap else store, path="v", mode="r")

    array = reopen()
    meta = json.dumps(array.metadata.to_dict(), default=str)
    return Built(read=lambda sel: array[sel], array=array, store_path=path, reopen=reopen,
                 info={"zarr_json_bytes": len(meta), **store_bytes(path), **info})


def _per_chunk_case(plan: JoinedArrayPlan) -> tuple[list[dict], list[int]]:
    """The plan's run-length table as a list of distinct cases and, for every chunk along
    the joined axis, the case it uses."""
    distinct: list[dict] = []
    seen: dict[str, int] = {}
    ids: list[int] = []
    for case in plan.cases:
        body = {k: v for k, v in case.items() if k != "chunks"}
        key = json.dumps(body, sort_keys=True, default=str)
        if key not in seen:
            seen[key] = len(distinct)
            distinct.append(body)
        ids += [seen[key]] * case["chunks"]
    return distinct, ids


# ------------------------------------------------------------------ the switch codec, by how the case is found


def switch_position(case: Case, pieces: list[ManifestArray]) -> Built:
    """A case table in zarr.json, run-length encoded along the joined axis; the codec is told the chunk's position."""
    plan = _plan(pieces)
    repo, path = new_repo("switch_position", case.name)
    session = repo.writable_session("main")
    _create(session.store, "v", plan, SwitchCodec(axis=0, cases=plan.cases))
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    return _finish("switch_position", case, repo, path, strategy=plan.strategy, cases=len(plan.cases))


def switch_key(case: Case, pieces: list[ManifestArray]) -> Built:
    """Every reference carries an 8-byte key, the hash of its case; the codec reads the key."""
    plan = _plan(pieces, select="key")
    repo, path = new_repo("switch_key", case.name)
    session = repo.writable_session("main")
    _create(session.store, "v", plan, SwitchCodec(select="key", key_bytes=plan.key_bytes, cases=plan.cases))
    _refs(session.store, "v", plan.refs)
    _literals(session.store, pieces)
    session.commit("x")
    return _finish("switch_key", case, repo, path, strategy=plan.strategy, cases=len(plan.cases))


def switch_case_id(case: Case, pieces: list[ManifestArray]) -> Built:
    """Every reference carries a 1-byte case number (the per-chunk variant id of veda-odd#498)."""
    plan = _plan(pieces)
    distinct, ids = _per_chunk_case(plan)
    if len(distinct) > 256:
        raise NotExpressible("more than 256 cases do not fit a 1-byte case number")
    cases = [{"key": f"{i:02x}", **c} for i, c in enumerate(distinct)]
    refs = [{**r, "prefix": bytes([ids[r["index"][0]]]) + r.get("prefix", b"")} for r in plan.refs]
    repo, path = new_repo("switch_case_id", case.name)
    session = repo.writable_session("main")
    _create(session.store, "v", plan, SwitchCodec(select="key", key_bytes=1, cases=cases))
    _refs(session.store, "v", refs)
    _literals(session.store, pieces, lambda chunk: bytes([ids[chunk]]))
    session.commit("x")
    return _finish("switch_case_id", case, repo, path, strategy=plan.strategy, cases=len(cases))


def _templates(distinct: list[dict]) -> tuple[list[dict], list[int], list[tuple[float, ...]]]:
    """Cases that differ only in the numbers of a `scale_offset` codec become one case
    with open parameters. Returns the templates, and per input case its template and values."""
    templates: list[dict] = []
    seen: dict[str, int] = {}
    which: list[int] = []
    values: list[tuple[float, ...]] = []
    for c in distinct:
        c = json.loads(json.dumps(c))
        vals: list[float] = []
        for codec in c["codecs"]:
            if codec.get("name") == "scale_offset":
                codec.setdefault("configuration", {})
                for k, default in (("offset", 0.0), ("scale", 1.0)):
                    vals.append(float(codec["configuration"].get(k, default)))
                    codec["configuration"][k] = {"parameter": len(vals) - 1}
        if vals:
            c["parameters"] = len(vals)
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen[key] = len(templates)
            templates.append(c)
        which.append(seen[key])
        values.append(tuple(vals))
    return templates, which, values


def switch_key_parameters(case: Case, pieces: list[ManifestArray]) -> Built:
    """A key, then the numbers that fill in the case's open parameters (refkv PR #73's design, on Icechunk)."""
    plan = _plan(pieces)
    distinct, ids = _per_chunk_case(plan)
    templates, which, values = _templates(distinct)
    keys = [hashlib.sha256(json.dumps(t, sort_keys=True).encode()).digest()[:8] for t in templates]
    cases = [{"key": k.hex(), **t} for k, t in zip(keys, templates)]

    def header(chunk: int) -> bytes:
        d = ids[chunk]
        return keys[which[d]] + np.asarray(values[d], dtype="<f8").tobytes()

    refs = [{**r, "prefix": header(r["index"][0]) + r.get("prefix", b"")} for r in plan.refs]
    repo, path = new_repo("switch_key_parameters", case.name)
    session = repo.writable_session("main")
    _create(session.store, "v", plan, SwitchXCodec(config_json=json.dumps({"select": "key_parameters", "key_bytes": 8, "cases": cases})))
    _refs(session.store, "v", refs)
    _literals(session.store, pieces, header)
    session.commit("x")
    return _finish("switch_key_parameters", case, repo, path, strategy=plan.strategy, cases=len(cases))


def switch_index_prefix(case: Case, pieces: list[ManifestArray]) -> Built:
    """The store puts the chunk's grid index in front of its bytes (zarr-specs#399 design 2.1); the table is as for position."""
    plan = _plan(pieces)
    repo, path = new_repo("switch_index_prefix", case.name)
    session = repo.writable_session("main")
    config = {"select": "index_prefix", "axis": 0, "ndim": 3, "cases": plan.cases}
    _create(session.store, "v", plan, SwitchXCodec(config_json=json.dumps(config)))
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    return _finish("switch_index_prefix", case, repo, path, wrap=lambda s: IndexPrefixStore(s, ("v",)),
                   strategy=plan.strategy, cases=len(plan.cases), needs_store_wrapper=True)


def _write_sidecar(store: Any, plan: JoinedArrayPlan) -> tuple[list[dict], bool]:
    distinct, ids = _per_chunk_case(plan)
    templates, which, values = _templates(distinct)
    ids = ids[DROP_CHUNKS:]  # held by the store with each reference, so a manifest operation carries it along
    table = zarr.create_array(store, name="v__cases", shape=(len(ids),), chunks=(max(1, len(ids)),), dtype="uint32", fill_value=0)
    table[:] = np.array([which[d] for d in ids], dtype="uint32")
    width = max((len(v) for v in values), default=0)
    if width:
        params = zarr.create_array(store, name="v__parameters", shape=(len(ids), width), chunks=(max(1, len(ids)), width), dtype="float64", fill_value=0)
        params[:] = np.array([list(values[d]) + [0.0] * (width - len(values[d])) for d in ids])
    return templates, bool(width)


def switch_sidecar(case: Case, pieces: list[ManifestArray]) -> Built:
    """The codec itself reads each chunk's case from another array in the store (design 2.2, ChunkContextFetcherCodec)."""
    plan = _plan(pieces)
    repo, path = new_repo("switch_sidecar", case.name)
    session = repo.writable_session("main")
    distinct, ids = _per_chunk_case(plan)
    # parameters would need a second sidecar read by the codec; cases are kept whole here
    table = zarr.create_array(session.store, name="v__cases", shape=(len(ids),), chunks=(len(ids),), dtype="uint32", fill_value=0)
    table[:] = np.array(ids, dtype="uint32")
    config = {"select": "sidecar", "axis": 0, "sidecar": "__cases", "cases": distinct}
    _create(session.store, "v", plan, SwitchXCodec(config_json=json.dumps(config)))
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    return _finish("switch_sidecar", case, repo, path, strategy=plan.strategy, cases=len(distinct), extra_arrays=1)


def switch_store_context(case: Case, pieces: list[ManifestArray]) -> Built:
    """The store has a new method that returns a chunk's context (design 3, "Zarr v3.2"); here a wrapper holds the table."""
    plan = _plan(pieces)
    repo, path = new_repo("switch_store_context", case.name)
    session = repo.writable_session("main")
    templates, has_parameters = _write_sidecar(session.store, plan)
    config = {"select": "context", "cases": templates}
    _create(session.store, "v", plan, SwitchXCodec(config_json=json.dumps(config)))
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    return _finish("switch_store_context", case, repo, path,
                   wrap=lambda s: ContextStore.load(s, "v", 0, "__cases", "__parameters" if has_parameters else None),
                   strategy=plan.strategy, cases=len(templates), needs_store_wrapper=True)


def conditional_codec(case: Case, pieces: list[ManifestArray]) -> Built:
    """zarr-extensions `conditional`: a bytes-to-bytes codec whose 1-byte header says which of its codecs to apply."""
    from header_codecs import ConditionalCodec  # noqa: F401  (registers the codec)

    plan = _plan(pieces)
    if plan.strategy != "switch":
        raise NotExpressible("the chunk shape changes, and a bytes-to-bytes codec cannot change what a chunk decodes to")
    distinct, ids = _per_chunk_case(plan)
    heads = {json.dumps([c for c in d["codecs"] if c["name"] in ("bytes", "cast_value", "scale_offset")], sort_keys=True) for d in distinct}
    if len(heads) > 1:
        raise NotExpressible("the files differ before the bytes stage (data type or packing), which a bytes-to-bytes codec cannot vary")
    for d in distinct:
        if "fill_value" in d:
            raise NotExpressible("a bytes-to-bytes codec cannot give a run of chunks its own fill value")
        if d.get("encoded_chunk_shape") and d["encoded_chunk_shape"][0] is not None:
            raise NotExpressible("chunks are stored larger than their place in the grid, and a bytes-to-bytes codec cannot crop")
    head = json.loads(next(iter(heads)))
    tails = [[c for c in d["codecs"] if c not in head] for d in distinct]
    union: list[dict] = []
    for t in sorted(tails, key=len, reverse=True):  # longest first, so a shared order is kept when there is one
        for c in t:
            if c not in union:
                union.append(c)
    for t in tails:  # the conditional codec applies its codecs in their listed order
        if [c for c in union if c in t] != t:
            raise NotExpressible("the files apply the same filters in different orders")
    masks = [sum(1 << union.index(c) for c in t) for t in tails]
    refs = [{**r, "prefix": bytes([masks[ids[r["index"][0]]]])} for r in plan.refs]
    repo, path = new_repo("conditional_codec", case.name)
    session = repo.writable_session("main")
    arr = _create(session.store, "v", plan, "auto")
    meta = json.loads(json.dumps(arr.metadata.to_dict(), default=str))
    meta["codecs"] = [*head, {"name": "conditional", "configuration": {"codecs": union, "header_bits": 8}}]
    zarr.core.sync.sync(session.store.set("v/zarr.json", zarr.core.buffer.default_buffer_prototype().buffer.from_bytes(json.dumps(meta).encode())))
    _refs(session.store, "v", refs)
    session.commit("x")
    return _finish("conditional_codec", case, repo, path, cases=len(set(masks)))


# ------------------------------------------------------------------ array concatenation


def _same_dtype(pieces: list[ManifestArray]) -> list[ManifestArray]:
    """Pieces cast to one data type by a codec in front of their own chain, as the planner does inside a case."""
    dtype = np.result_type(*[p.dtype for p in pieces])
    out = []
    for p in pieces:
        if p.dtype != dtype:
            md = json.loads(json.dumps(p.metadata.to_dict(), default=str))
            stored = md["data_type"]
            md["data_type"], md["fill_value"] = str(dtype), "NaN" if dtype.kind == "f" else 0
            cast: dict = {"data_type": stored}
            if dtype.kind == "f" and p.dtype.kind in "iu":
                cast["scalar_map"] = {"encode": [["NaN", int(p.metadata.fill_value)]]}
            md["codecs"] = [{"name": "cast_value", "configuration": cast}, *md["codecs"]]
            p = ManifestArray(metadata=ArrayV3Metadata.from_dict(md), chunkmanifest=p.manifest)
        out.append(p)
    return out


def _write_pieces(session: Any, pieces: list[ManifestArray]) -> zarr.Group:
    root = zarr.group(session.store)
    group = root.require_group("_pieces/v")
    for i, piece in enumerate(pieces):
        write_virtual_variable_to_icechunk(store=session.store, group=group, name=str(i), var=xr.Variable(DIMS, piece))
    return root


def array_concat(case: Case, pieces: list[ManifestArray]) -> Built:
    """Each file's array is kept; a codec on a further array lists them and reads them (zarr-specs#288)."""
    pieces = _same_dtype(pieces)
    repo, path = new_repo("array_concat", case.name)
    session = repo.writable_session("main")
    root = _write_pieces(session, pieces)
    cma = ConcatenatedManifestArray(pieces, axis=0)
    root.require_array(
        name="v", shape=cma.shape, chunks=_reported_chunks(cma), dtype=pieces[0].dtype, filters=None, compressors=None,
        serializer=ArrayConcatCodec(axis=0, pieces=[{"path": f"../_pieces/v/{i}", "size": p.shape[0]} for i, p in enumerate(pieces)]),
        dimension_names=DIMS, fill_value=pieces[0].metadata.fill_value,
    )
    session.commit("x")
    return _finish("array_concat", case, repo, path, extra_arrays=len(pieces))


def zarr_reference(case: Case, pieces: list[ManifestArray]) -> Built:
    """One chunk per file on a rectilinear grid; the chunk's bytes are a reference to that file's array (jbms, zarr-specs#288)."""
    pieces = _same_dtype(pieces)
    repo, path = new_repo("zarr_reference", case.name)
    session = repo.writable_session("main")
    _write_pieces(session, pieces)
    shape = (sum(p.shape[0] for p in pieces), *pieces[0].shape[1:])
    zarr.create_array(session.store, name="v", shape=shape, chunks=[[p.shape[0] for p in pieces], [shape[1]], [shape[2]]], dtype=pieces[0].dtype,
                      fill_value=pieces[0].metadata.fill_value, serializer=ZarrReferenceCodec(), compressors=None, filters=None, dimension_names=DIMS)
    proto = zarr.core.buffer.default_buffer_prototype()
    for i in range(len(pieces)):
        zarr.core.sync.sync(session.store.set(f"v/c/{i}/0/0", proto.buffer.from_bytes(json.dumps({"path": f"../_pieces/v/{i}"}).encode())))
    session.commit("x")
    return _finish("zarr_reference", case, repo, path, extra_arrays=len(pieces))


# ------------------------------------------------------------------ changes to the grid, or a convention beside it


def _only_padding(plan: JoinedArrayPlan, what: str) -> dict:
    """The one case of a plan whose files differ at most in how their last chunk is stored."""
    if plan.strategy != "switch":
        raise NotExpressible(f"the chunk shape changes; {what} keeps one codec chain and one inner chunk shape")
    distinct, _ = _per_chunk_case(plan)
    chains = {json.dumps(d["codecs"], sort_keys=True) for d in distinct}
    if len(chains) > 1:
        raise NotExpressible(f"the files' codec chains differ; {what} keeps one chain for the array")
    if any("fill_value" in d for d in distinct):
        raise NotExpressible(f"the files' fill values differ; {what} keeps one fill value")
    return distinct[0]


def _plain_array(session: Any, plan: JoinedArrayPlan, chain: list[dict], grid: list[list[int]], shape: tuple, attributes: dict) -> None:
    """An array with an ordinary codec chain, written as metadata."""
    shape, grid = _dropped(shape, grid)
    arr = zarr.create_array(session.store, name="v", shape=shape, chunks=grid, dtype=plan.dtype, fill_value=plan.fill_value, dimension_names=DIMS,
                            attributes=attributes)
    meta = json.loads(json.dumps(arr.metadata.to_dict(), default=str))
    meta["codecs"] = chain
    zarr.core.sync.sync(session.store.set("v/zarr.json", zarr.core.buffer.default_buffer_prototype().buffer.from_bytes(json.dumps(meta).encode())))


def _encoded_sizes(plan: JoinedArrayPlan) -> list[int]:
    out: list[int] = []
    for c in plan.cases:
        e = (c.get("encoded_chunk_shape") or [None])[0]
        out += [e] * c["chunks"]
    return [n if e is None else e for e, n in zip(out, plan.chunk_grid[0])]


def grid_encoded_shapes(case: Case, pieces: list[ManifestArray]) -> Built:
    """The chunk grid states each chunk's stored size beside its logical size (zarr-extensions PR #75)."""
    plan = _plan(pieces)
    one = _only_padding(plan, "a chunk grid with encoded chunk shapes")
    xcodecs.enable_grid_encoded_chunk_shapes()
    repo, path = new_repo("grid_encoded_shapes", case.name)
    session = repo.writable_session("main")
    encoded = _encoded_sizes(plan)
    padded = encoded != plan.chunk_grid[0]
    _plain_array(session, plan, one["codecs"], plan.chunk_grid, plan.shape, {"encoded_chunk_shapes": [encoded, None, None]} if padded else {})
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    return _finish("grid_encoded_shapes", case, repo, path, emulated="declared in an attribute, read by a patched grid")


def domain_convention(case: Case, pieces: list[ManifestArray]) -> Built:
    """The array is stored with its padding; an attribute declares which ranges are data (d-v-b, zarr-extensions#74)."""
    plan = _plan(pieces)
    one = _only_padding(plan, "a declared domain")
    repo, path = new_repo("domain_convention", case.name)
    session = repo.writable_session("main")
    encoded = _encoded_sizes(plan)
    starts = np.concatenate([[0], np.cumsum(encoded)])
    domain = [[int(starts[i]), int(starts[i]) + n] for i, n in enumerate(plan.chunk_grid[0])]
    merged: list[list[int]] = []
    for a, b in domain:
        if merged and merged[-1][1] == a:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    shape = (int(starts[-1]), *plan.shape[1:])
    _plain_array(session, plan, one["codecs"], [encoded, plan.chunk_grid[1], plan.chunk_grid[2]], shape, {"domain": merged})
    _refs(session.store, "v", plan.refs)
    session.commit("x")
    built = _finish("domain_convention", case, repo, path, stored_shape=list(shape))
    view = DomainView(built.array)
    built.read = lambda sel: view[sel]
    built.one_array = shape == tuple(plan.shape)  # an ordinary reader sees the padding as data
    return built


# ------------------------------------------------------------------ copying


def materialize_odd_chunks(case: Case, pieces: list[ManifestArray]) -> Built:
    """The array takes the commonest layout; every chunk that does not fit it is decoded and stored again (VirtualiZarr#1111)."""
    dtype = np.result_type(*[p.dtype for p in pieces])
    fills = [dtype.type(p.metadata.fill_value) for p in pieces]
    fill = max(fills, key=lambda f: sum(f.tobytes() == g.tobytes() for g in fills))
    if dtype.kind == "f" and any(p.dtype.kind in "iu" for p in pieces):
        fill = dtype.type("nan")
    singles = []
    for p in pieces:
        if not bool((np.asarray(p.manifest._paths) != "").any()):
            singles.append(None)  # stores nothing: reads as the array's fill value
            continue
        singles.append(plan_joined_array([p], axis=0, dtype=dtype, fill_value=fill))

    def signature(plan: JoinedArrayPlan) -> str:
        c = plan.cases[0]
        return json.dumps([c["codecs"], plan.chunk_grid[1], plan.chunk_grid[2], c.get("fill_value"), plan.strategy], sort_keys=True, default=str)

    weight: dict[str, int] = {}
    for plan in singles:
        if plan is not None:
            weight[signature(plan)] = weight.get(signature(plan), 0) + plan.shape[0]
    # A chunk that is stored again is encoded with the array's chain, so that chain must
    # hold every file's values exactly: not one that packs or narrows them.
    lossless = {s: w for s, w in weight.items() if not any(c["name"] in ("cast_value", "scale_offset") for c in json.loads(s)[0])}
    if not lossless:
        raise NotExpressible("every file packs or narrows its values in its own way, so no file's chain can hold the others' values: all would be copied")
    major = max(lossless, key=lambda s: lossless[s])
    lead = next(p for p in singles if p is not None and signature(p) == major)
    if "fill_value" in lead.cases[0]:  # the array takes the fill value of the files it keeps virtual
        fill = dtype.type(lead.cases[0]["fill_value"])
    if lead.strategy != "switch":
        raise NotExpressible("the commonest layout is itself a virtual shard")
    chain = lead.cases[0]["codecs"]
    step = lead.chunk_grid[0][0]
    grid0: list[int] = []
    refs: list[dict] = []
    native: list[tuple[int, int]] = []  # [start, stop) along the joined axis, to be stored again
    base = t0 = 0
    for p, plan in zip(pieces, singles):
        n = p.shape[0]
        if plan is None:
            edges = [step] * (n // step) + ([n % step] if n % step else [])
        elif signature(plan) == major:
            edges = list(plan.chunk_grid[0])
            encoded = _encoded_sizes(plan)
            offsets = np.concatenate([[0], np.cumsum(edges)])
            for r in plan.refs:
                i = r["index"][0]
                if encoded[i] == edges[i]:
                    refs.append({**r, "index": [i + base, *r["index"][1:]]})
            for i, (e, size) in enumerate(zip(encoded, edges)):
                if e != size:
                    native.append((t0 + int(offsets[i]), t0 + int(offsets[i + 1])))
        else:
            edges = [step] * (n // step) + ([n % step] if n % step else [])
            native.append((t0, t0 + n))
        grid0 += edges
        base += len(edges)
        t0 += n
    shape = (t0, *pieces[0].shape[1:])
    repo, path = new_repo("materialize_odd_chunks", case.name)
    session = repo.writable_session("main")
    holder = JoinedArrayPlan("switch", 0, shape, dtype, fill, [grid0, lead.chunk_grid[1], lead.chunk_grid[2]], [])
    _plain_array(session, holder, chain, holder.chunk_grid, shape, {})
    _refs(session.store, "v", refs)
    arr = zarr.open_array(session.store, path="v", mode="r+")
    copied = 0
    for a, b in native:
        arr[a:b] = case.truth[a:b]  # what reading the files gives
        copied += b - a
    session.commit("x")
    return _finish("materialize_odd_chunks", case, repo, path, copied_fraction=round(copied / shape[0], 4))


def native_copy(case: Case, pieces: list[ManifestArray]) -> Built:
    """Not virtual: every value is read from the files and written as a new Zarr array."""
    repo, path = new_repo("native_copy", case.name)
    session = repo.writable_session("main")
    arr = zarr.create_array(session.store, name="v", shape=case.truth.shape, chunks=(max(1, min(8, case.truth.shape[0])), *case.truth.shape[1:]),
                            dtype=case.truth.dtype, fill_value=np.nan if case.truth.dtype.kind == "f" else 0, dimension_names=DIMS)
    arr[...] = case.truth
    session.commit("x")
    return _finish("native_copy", case, repo, path, copied_fraction=1.0)


def separate_files(case: Case, pieces: list[ManifestArray]) -> Built:
    """Not one array: xarray opens every file and joins them in memory (`open_mfdataset`)."""
    present = list(case.files)
    try:
        ds = xr.open_mfdataset(present, combine="nested", concat_dim="time", engine="h5netcdf", mask_and_scale=case.packed, decode_times=False)
    except ValueError:  # files written without netCDF dimensions
        ds = xr.open_mfdataset(present, combine="nested", concat_dim="phony_dim_0", engine="h5netcdf", mask_and_scale=False, decode_times=False,
                               phony_dims="sort", drop_variables=["time"])

    def read(sel: tuple) -> np.ndarray:
        return ds["v"][sel].values

    return Built(read=read, array=None, one_array=False, info={})


ARMS: dict[str, Callable[[Case, list[ManifestArray]], Built]] = {f.__name__: f for f in (
    switch_position, switch_key, switch_case_id, switch_key_parameters, switch_index_prefix, switch_sidecar, switch_store_context,
    conditional_codec, array_concat, zarr_reference, grid_encoded_shapes, domain_convention, materialize_odd_chunks, native_copy, separate_files,
)}
