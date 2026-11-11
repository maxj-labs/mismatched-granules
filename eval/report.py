"""Build the evaluation page from the result files.

    python report.py        # writes ../evaluation/index.html (page content, as the brief is)

Every number and every mark on the page is read from results/eval/*.json; the text
around them is written here.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "eval"
OUT = HERE.parent / "evaluation" / "index.html"


def load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


MATRIX, UNAWARE, COMPOSE, SCALING = load("matrix.json"), load("unaware.json"), load("compose.json"), load("scaling.json")
KERCHUNK, TS = load("without_icechunk.json"), load("tensorstore_stack.json")
PERF = {"pr": load("perf_pr_tmin.json"), "na": load("perf_na_dayl.json")}
REFKV = json.loads(p.read_text()) if (p := HERE.parent / "results_http" / "bench_pr_tmin_2014_2025_refkv.json").exists() else {}

LABEL = {
    "switch_position": "Switch codec · case by position",
    "switch_key": "Switch codec · case by key",
    "switch_case_id": "Switch codec · 1-byte case number",
    "switch_key_parameters": "Switch codec · key + parameters",
    "switch_index_prefix": "Switch codec · store prepends the chunk index",
    "switch_sidecar": "Switch codec · codec reads a side array",
    "switch_store_context": "Switch codec · store answers a context call",
    "conditional_codec": "<span class=\"mono\">conditional</span> codec",
    "array_concat": "Array concatenation codec",
    "zarr_reference": "Reference-per-chunk codec",
    "grid_encoded_shapes": "Encoded chunk shapes in the grid",
    "domain_convention": "Declared valid domain",
    "materialize_odd_chunks": "Copy the chunks that do not fit",
    "native_copy": "Copy everything to native Zarr",
    "separate_files": "No store: <span class=\"mono\">open_mfdataset</span>",
}
SOURCE = {
    "switch_position": ("zarr-specs#399 design 2 with a run-length table (TomNicholas, ghidalgo3); chunk coordinates in <span class=\"mono\">ArraySpec</span> (astrofrog)", "Built: zarr-python and VirtualiZarr branches"),
    "switch_key": ("zarr-specs#399: inline bytes in front of the chunk, read by the codec (d-v-b, jbms)", "Built: all three branches"),
    "switch_case_id": ("NASA-IMPACT/veda-odd#498: a per-chunk variant in the manifest", "Built: the key mode with 1-byte keys"),
    "switch_key_parameters": ("refkv PR #73 (d-v-b): cases as templates, numbers carried per chunk", "Built here on Icechunk as an evaluation codec; also run as released on refkv"),
    "switch_index_prefix": ("zarr-specs#399 design 2.1 as first written; veda-odd#498 step 3", "Built: the Icechunk branch prepends the index for arrays that ask; the codec is an evaluation codec"),
    "switch_sidecar": ("zarr-specs#399 design 2.2, <span class=\"mono\">ChunkContextFetcherCodec</span>", "Built as an evaluation codec that reads a second array"),
    "switch_store_context": ("zarr-specs#399 design 3, &ldquo;Zarr v3.2&rdquo;: <span class=\"mono\">Store.get_context</span>", "Emulated: a store wrapper answers the call; the codec asks it directly, not through <span class=\"mono\">ArraySpec</span>"),
    "conditional_codec": ("zarr-extensions <span class=\"mono\">conditional</span> (mkitti); HDF5's per-chunk filter mask", "Built: my reading of the merged spec, 1-byte header"),
    "array_concat": ("zarr-specs#288 (TomNicholas, 2024); zarr-specs#399 design 1", "Built: zarr-python and VirtualiZarr branches"),
    "zarr_reference": ("zarr-specs#288 (jbms): each chunk of a rectilinear grid is a reference to an array", "Built as an evaluation codec"),
    "grid_encoded_shapes": ("zarr-extensions#74 and PR #75 (TomNicholas)", "Emulated: declared in an attribute, read by a patched chunk grid"),
    "domain_convention": ("zarr-extensions#74 (d-v-b): store the padding, declare the valid domain", "Built as a reader-side helper"),
    "materialize_odd_chunks": ("VirtualiZarr#1111 (selmanozleyen): store the awkward chunks again as native chunks", "Built, generalised to any chunk that does not fit the commonest layout"),
    "native_copy": ("The alternative to virtualizing at all", "Built"),
    "separate_files": ("What users do today", "Run"),
}
CASE_LABEL = {
    "codec_eras": "Codec eras", "file_lengths": "File lengths", "padded_last_chunk": "Padded last chunk", "packing_per_file": "Packing per file",
    "data_types": "Data types", "chunk_shape_change": "Chunk shape change", "fill_value_differs": "Fill value differs", "variable_absent": "Variable absent",
    "layouts_alternate": "Layouts alternate", "packing_per_chunk": "Packing per chunk",
}
ARMS = list(LABEL)
CASES = list(CASE_LABEL)
e = html.escape


def size(n: float | None) -> str:
    if n is None:
        return ""
    for unit, scale in (("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if n < 100 * scale else f"{n / scale:.0f} {unit}"
    return f"{n:.0f} B"


def cell(arm: str, case: str) -> dict:
    return MATRIX["cells"].get(f"{arm}|{case}", {})


def count(arm: str) -> int:
    return sum(cell(arm, c).get("outcome") == "equal" for c in CASES)


# ------------------------------------------------------------------ tables


def options_table() -> str:
    rows = "".join(f"<tr><td>{LABEL[a]}</td><td>{SOURCE[a][0]}</td><td>{SOURCE[a][1]}</td></tr>" for a in ARMS)
    return f"""<div class="scroll"><table><thead><tr><th>Option</th><th>Where it was proposed</th><th>How it was tested here</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def matrix_table() -> str:
    head = "".join(f'<th class="c" title="{e(MATRIX["cases"][c])}">{CASE_LABEL[c]}</th>' for c in CASES)
    body = ""
    for a in ARMS:
        tds = ""
        for c in CASES:
            x = cell(a, c)
            o = x.get("outcome")
            if o == "equal":
                extra = ""
                if x.get("copied_fraction") not in (None, 0, 0.0) and a != "native_copy":
                    extra = f'<span class="sub">{x["copied_fraction"] * 100:.0f}% copied</span>'
                tds += f'<td class="c"><span class="yes" title="every read equals the files">&#10003;</span>{extra}</td>'
            elif o == "not expressible":
                tds += f'<td class="c"><span class="dash" title="{e(x.get("why", ""))}">&ndash;</span></td>'
            else:
                tds += f'<td class="c"><span class="no" title="{e(x.get("why", ""))}">&#10007;</span></td>'
        body += f'<tr><td>{LABEL[a]}</td>{tds}<td class="n">{count(a)} of {len(CASES)}</td></tr>'
    return f"""<div class="scroll"><table class="grid"><thead><tr><th>Option</th>{head}<th class="n">Described</th></tr></thead><tbody>{body}</tbody></table></div>"""


def reasons_list() -> str:
    """Why an option cannot describe a case, grouped."""
    seen: dict[tuple[str, str], list[str]] = {}
    for a in ARMS:
        for c in CASES:
            x = cell(a, c)
            if x.get("outcome") == "not expressible":
                seen.setdefault((a, x["why"]), []).append(CASE_LABEL[c].lower())
            elif x.get("outcome") in ("differs", "error"):
                why = ("xarray fills the file that lacks the variable with NaN, where the stores return the fill value the other files declare"
                       if a == "separate_files" else "reads did not equal the files: " + x.get("why", "")[:120])
                seen.setdefault((a, why), []).append(CASE_LABEL[c].lower())
    items = "".join(f"<li><b>{re.sub('<[^>]+>', '', LABEL[a])}</b>, {', '.join(cs)}: {e(why)}.</li>" for (a, why), cs in seen.items())
    return f"<ul>{items}</ul>"


def unaware_summary(arm: str) -> tuple[str, str]:
    """(what released zarr-python does, what a reader with the codecs but released Icechunk does)"""
    out = []
    for reader in ("released zarr-python, released Icechunk", "zarr-python with the codecs, released Icechunk"):
        outcomes = [v[reader]["outcome"] for k, v in UNAWARE.items() if k.startswith(arm + "|") and reader in v]
        if not outcomes:
            out.append("")
        elif "WRONG VALUES" in outcomes:
            out.append(f'<span class="no">wrong values, no error</span> ({outcomes.count("WRONG VALUES")} of {len(outcomes)})')
        elif all(o == "refuses" for o in outcomes):
            out.append('<span class="yes">refuses</span>')
        elif all(o == "reads correctly" for o in outcomes):
            out.append("reads correctly")
        else:
            out.append(f'reads {outcomes.count("reads correctly")}, <span class="yes">refuses</span> {outcomes.count("refuses")}')
    if arm == "switch_store_context":
        out[1] = '<span class="sub0">not a stored format: it needs new behaviour from the store</span>'
    return out[0], out[1]


def compose_summary(arm: str) -> str:
    if arm in COMPOSE.get("by_inspection", {}):
        return f'<span class="sub0">{e(COMPOSE["by_inspection"][arm])} &dagger;</span>'
    cells = [v for k, v in COMPOSE.get("cells", {}).items() if k.startswith(arm + "|")]
    if not cells:
        return ""
    good = sum(v["outcome"] == "composes as it stands" for v in cells)
    if good == len(cells):
        return f'<span class="yes">yes</span> ({good} of {len(cells)})'
    silent = sum("values are wrong" in v.get("why", "") for v in cells)
    tail = f"; {silent} read wrong values without an error" if silent else ""
    return f'<span class="no">no</span>: correct in {good} of {len(cells)}{tail}'


def scale(arm: str, kind: str, field: str, n: int = 16000) -> float | None:
    return SCALING.get(f"{kind}|{n}", {}).get(arm, {}).get(field)


def requirements_table() -> str:
    body = ""
    for a in ARMS[:-1]:
        zarr_side, store_side = unaware_summary(a)
        body += (f"<tr><td>{LABEL[a]}</td><td class=\"n\">{count(a)} of {len(CASES)}</td><td>{zarr_side}</td><td>{store_side}</td>"
                 f"<td>{compose_summary(a)}</td></tr>")
    return f"""<div class="scroll"><table><thead><tr><th>Option</th><th class="n">Cases described</th><th>Released zarr-python</th>
<th>Codecs known, released Icechunk</th><th>Leading chunks dropped, nothing else rewritten</th></tr></thead><tbody>{body}</tbody></table></div>"""


def scaling_table() -> str:
    arms = [a for a in ARMS if scale(a, "alternate", "outcome") or scale(a, "packed", "outcome")]
    body = ""
    for a in arms:
        tds = ""
        for kind in ("alternate", "packed"):
            if scale(a, kind, "outcome") != "equal":
                tds += '<td class="n dashcell" colspan="4">cannot describe it</td>'
                continue
            meta = scale(a, kind, "zarr_json_bytes")
            tds += (f'<td class="n{" hot" if meta > 100_000 else ""}">{size(meta)}</td><td class="n">{size(scale(a, kind, "manifest_bytes"))}</td>'
                    f'<td class="n">{scale(a, kind, "build_seconds"):.0f} s</td><td class="n">{scale(a, kind, "microseconds_per_chunk") / 1000:.2f} ms</td>')
        body += f"<tr><td>{LABEL[a]}</td>{tds}</tr>"
    return f"""<div class="scroll"><table><thead>
<tr><th></th><th class="n" colspan="4">Two layouts alternating chunk by chunk</th><th class="n" colspan="4">Every chunk packed its own way</th></tr>
<tr><th>Option</th><th class="n">zarr.json</th><th class="n">Manifests</th><th class="n">Build</th><th class="n">Read, per chunk</th>
<th class="n">zarr.json</th><th class="n">Manifests</th><th class="n">Build</th><th class="n">Read, per chunk</th></tr></thead><tbody>{body}</tbody></table></div>"""


def growth_note() -> str:
    def series(arm: str, kind: str) -> str:
        return " → ".join(size(scale(arm, kind, "zarr_json_bytes", n)) for n in (1000, 4000, 16000))

    return (f"Growth of <code>zarr.json</code> from 1,000 to 4,000 to 16,000 chunks, every chunk packed its own way: by position {series('switch_position', 'packed')}; "
            f"by key {series('switch_key', 'packed')}; key + parameters {series('switch_key_parameters', 'packed')}; store context {series('switch_store_context', 'packed')}.")


def perf_table(which: str) -> str:
    perf = PERF[which]
    if not perf:
        return "<p class=\"note\">Not run.</p>"
    reads = list(next(v["reads"] for v in perf["arms"].values() if v.get("reads")))
    short = [r.split(":")[0] + ":" + r.split(":")[1].split(",")[0] for r in reads]
    head = "".join(f'<th class="n" title="{e(r)}">{e(s)}</th>' for r, s in zip(reads, short))
    body = ""
    base = perf["arms"].get("switch_position", {}).get("reads", {})
    for a, v in perf["arms"].items():
        if v.get("outcome") == "not expressible":
            body += f'<tr><td>{LABEL[a]}</td><td class="dashcell" colspan="{len(reads) + 3}">cannot describe it: {e(v["why"])}</td></tr>'
            continue
        tds = ""
        same_requests = True
        for r in reads:
            x = v["reads"][r]
            ok = "" if x["equal_to_files"] else ' <span class="no">&#10007;</span>'
            tds += f'<td class="n">{x["seconds"]:.2f} s{ok}</td>'
            same_requests &= x.get("source_requests") == base.get(r, {}).get("source_requests")
        req = '<span class="yes">same</span>' if same_requests else "/".join(str(v["reads"][r].get("source_requests")) for r in reads)
        body += f'<tr><td>{LABEL[a]}</td>{tds}<td class="n">{req}</td><td class="n">{size(v.get("manifest_bytes"))}</td><td class="n">{v.get("build_seconds", 0):.1f} s</td></tr>'
    requests = " · ".join(f"{base[r].get('source_requests'):,}" for r in reads) if base else ""
    return f"""<div class="scroll"><table><thead><tr><th>Option</th>{head}<th class="n">Source requests</th><th class="n">Manifests</th><th class="n">Build</th></tr></thead>
<tbody>{body}</tbody></table></div>
<p class="note">Requests reaching the source for the reads, in column order, by position: {requests}. &ldquo;Same&rdquo; means the same count for every read.</p>"""


def kerchunk_sentence() -> str:
    cells = KERCHUNK.get("cells", {})
    pos = [v["outcome"] for k, v in cells.items() if k.startswith("switch_position|")]
    key = [v["outcome"] for k, v in cells.items() if k.startswith("switch_key|")]
    return (f"As Kerchunk references read through fsspec, position selection read correctly in {pos.count('equal')} of {len(pos)} cases "
            f"(the exception is the chunk-shape change, which needs virtual shards); key selection in {key.count('equal')} of {len(key)}, "
            "because a Kerchunk reference is either inline bytes or one byte range, never a literal followed by a range.")


def tensorstore_sentence() -> str:
    runs = TS.get("runs", [])
    if not runs:
        return ""
    last = runs[-1]
    return (f"TensorStore's stack driver, with {last['layers']:,} one-chunk native arrays as layers: a {size(last['spec_bytes'])} spec "
            f"({last['bytes_per_layer']} bytes per layer), {last['open_seconds']:.2f} s to open, {last['microseconds_per_layer'] / 1000:.2f} ms per layer to read everything.")


if __name__ == "__main__":
    from page import render  # the page's text lives beside this file

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(globals()))
    print("wrote", OUT)
