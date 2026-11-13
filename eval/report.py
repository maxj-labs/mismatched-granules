"""Build the evaluation page from the result files.

    python report.py        # writes ../evaluation/index.html (page content, as the brief is)

Every number and every mark on the page is read from results/eval/*.json and
results_http/*.json; the text around them is in page.py.
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
PERF = {"pr": load("perf_pr_tmin.json"), "na": load("perf_na_dayl.json"), "pr120": load("perf_pr_tmin_concurrency_120.json")}
REFKV = json.loads(p.read_text()) if (p := HERE.parent / "results_http" / "bench_pr_tmin_2014_2025_refkv.json").exists() else {}

MONO = '<span class="mono">{}</span>'
LABEL = {
    "switch_position": "Layout table · found by position",
    "switch_key": "Layout table · 8-byte key in each reference",
    "switch_case_id": "Layout table · 1-byte number in each reference",
    "switch_key_parameters": "Layout table · key and numbers in each reference",
    "switch_index_prefix": "Layout table · store prepends the chunk index",
    "switch_sidecar": "Layout table · codec reads side arrays",
    "switch_store_context": "Layout table · store answers a context call",
    "conditional_codec": MONO.format("conditional") + " codec",
    "conditional_with_grid_shapes": MONO.format("conditional") + " + encoded chunk shapes",
    "array_concat": "Array concatenation codec",
    "zarr_reference": "Reference-per-chunk codec",
    "grid_encoded_shapes": "Encoded chunk shapes in the grid",
    "domain_convention": "Declared valid domain",
    "materialize_odd_chunks": "Copy the chunks that do not fit",
    "native_copy": "Copy everything to native Zarr",
    "separate_files": "No store: " + MONO.format("open_mfdataset"),
}
SOURCE = {
    "switch_position": ("zarr-specs#399 design 2 with a run-length table (TomNicholas, ghidalgo3); chunk coordinates in " + MONO.format("ArraySpec") + " (astrofrog)", "Built: zarr-python and VirtualiZarr branches"),
    "switch_key": ("zarr-specs#399: inline bytes in front of the chunk, read by the codec (d-v-b, jbms). The key as a hash of the layout's content is refkv PR #73's (d-v-b)", "Built: all three branches"),
    "switch_case_id": ("NASA-IMPACT/veda-odd#498: a per-chunk variant in the manifest", "Built as an evaluation codec: the byte numbers the layout instead of hashing it, which the real codec refuses"),
    "switch_key_parameters": ("refkv PR #73 (d-v-b): layouts as templates, numbers carried per chunk", "Built: all three branches, with typed numbers after the key; refkv itself run on Daymet"),
    "switch_index_prefix": ("zarr-specs#399 design 2.1 as first written; veda-odd#498 step 3", "Built: an Icechunk branch prepends the index for arrays that ask; evaluation codec"),
    "switch_sidecar": ("zarr-specs#399 design 2.2, " + MONO.format("ChunkContextFetcherCodec"), "Built as an evaluation codec that reads two more arrays"),
    "switch_store_context": ("zarr-specs#399 design 3, &ldquo;Zarr v3.2&rdquo;: " + MONO.format("Store.get_context"), "Emulated: a Python wrapper answers the call from a table indexed by position"),
    "conditional_codec": ("zarr-extensions " + MONO.format("conditional") + " (mkitti); HDF5's per-chunk filter mask", "Built: my reading of the merged spec, 1-byte header"),
    "conditional_with_grid_shapes": ("The two narrow proposals used together", "Built from the two rows beside it"),
    "array_concat": ("zarr-specs#288 (TomNicholas, 2024); zarr-specs#399 design 1", "Built: zarr-python and VirtualiZarr branches"),
    "zarr_reference": ("zarr-specs#288 (jbms): each chunk of a rectilinear grid is a reference to an array", "Built as an evaluation codec"),
    "grid_encoded_shapes": ("zarr-extensions#74 and PR #75 (TomNicholas)", "Emulated: declared in an attribute, read by a patched chunk grid"),
    "domain_convention": ("zarr-extensions#74 (d-v-b): store the padding, declare the valid domain", "Built as a reader-side helper"),
    "materialize_odd_chunks": ("VirtualiZarr#1111 (selmanozleyen): store the awkward chunks again as native chunks", "Built, for any chunk that does not fit the commonest layout"),
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
VIRTUAL = [a for a in ARMS if a not in ("native_copy", "separate_files")]
e = html.escape


def plain(arm: str) -> str:
    return re.sub("<[^>]+>", "", LABEL[arm])


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


# ------------------------------------------------------------------ the options, and the matrix


def options_table() -> str:
    rows = "".join(f"<tr><td>{LABEL[a]}</td><td>{SOURCE[a][0]}</td><td>{SOURCE[a][1]}</td></tr>" for a in ARMS)
    return f"""<div class="scroll"><table><thead><tr><th>Option</th><th>Where it was proposed</th><th>How it was tested here</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def matrix_table() -> str:
    head = "".join(f'<th class="c" title="{e(MATRIX["cases"][c])}">{CASE_LABEL[c]}</th>' for c in CASES)
    body = ""
    table = [a for a in ARMS if a.startswith("switch_")]
    together = len({tuple(cell(a, c).get("outcome") for c in CASES) for a in table}) == 1 and not any(cell(a, c).get("copied_fraction") for a in table for c in CASES)
    for a in ARMS:
        if together and a in table[1:]:
            continue
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
        name = f"Layout table · all {len(table)} variants" if together and a == table[0] else LABEL[a]
        body += f'<tr><td>{name}</td>{tds}<td class="n">{count(a)} of {len(CASES)}</td></tr>'
    return f"""<div class="scroll"><table class="grid"><thead><tr><th>Option</th>{head}<th class="n">Passed</th></tr></thead><tbody>{body}</tbody></table></div>"""


def reasons_list() -> str:
    """Why an option cannot describe a case, grouped."""
    seen: dict[tuple[str, str], list[str]] = {}
    for a in ARMS:
        for c in CASES:
            x = cell(a, c)
            if x.get("outcome") == "not expressible":
                seen.setdefault((a, x["why"]), []).append(CASE_LABEL[c].lower())
            elif x.get("outcome") in ("differs", "error"):
                why = ("xarray fills the file that lacks the variable with NaN; the stores return the fill value the other files declare, which is the convention this test compares against"
                       if a == "separate_files" else "reads did not equal the files: " + x.get("why", "")[:120])
                seen.setdefault((a, why), []).append(CASE_LABEL[c].lower())
    failing = {a for a, _ in seen}
    assert failing == {"conditional_codec", "conditional_with_grid_shapes", "grid_encoded_shapes", "domain_convention", "materialize_odd_chunks", "separate_files"}, failing
    return """<ul>
<li><b>The conditional codec</b> works on bytes. It can switch compressors, but it cannot crop a padded chunk, and it cannot change the data type, packing, chunk shape or fill value. Adding encoded chunk shapes fixes the cropping only.</li>
<li><b>Encoded chunk shapes and the declared valid domain</b> deal with padding only. The array still has one set of codecs, one inner chunk shape and one fill value.</li>
<li><b>Copying the chunks that do not fit</b> has nothing to keep when every file or chunk packs its values differently: all of the data would be copied.</li>
<li><b>open_mfdataset</b> fills a file that lacks the variable with NaN. The stores return the fill value the other files declare, which is what this test expects.</li>
</ul>"""


# ------------------------------------------------------------------ readers without the feature; without Icechunk


def reader_outcomes(arm: str, reader: str) -> list[str]:
    return [v[reader]["outcome"] for k, v in UNAWARE.items() if k.startswith(arm + "|") and reader in v]


def reader_cell(arm: str, reader: str) -> str:
    outcomes = reader_outcomes(arm, reader)
    if not outcomes:
        return ""
    n = len(outcomes)
    parts = []
    for outcome, cls, some, every in (("WRONG VALUES", "no", "wrong with no error", "wrong values with no error"), ("reads correctly", "", "read correctly", "reads correctly"),
                                      ("refuses", "yes", "refused", "refuses"), ("probe failed", "dash", "probe failed", "probe failed")):
        k = outcomes.count(outcome)
        if k:
            text = f"{k} {some}" if k < n else every
            parts.append(f'<span class="{cls}">{text}</span>' if cls else text)
    return ", ".join(parts)


def readers_table() -> str:
    body = ""
    for a in VIRTUAL:
        body += (f"<tr><td>{LABEL[a]}</td><td class=\"n\">{count(a)}</td>"
                 f"<td>{reader_cell(a, 'released zarr-python, released Icechunk')}</td>"
                 f"<td>{reader_cell(a, 'zarr-python with the codecs, released Icechunk')}</td></tr>")
    return f"""<div class="scroll"><table><thead><tr><th>Option</th><th class="n">Stores opened</th><th>Released zarr-python, released Icechunk</th>
<th>zarr-python with the new codecs, released Icechunk</th></tr></thead><tbody>{body}</tbody></table></div>"""


# ------------------------------------------------------------------ composition


ADDING = ("append a file", "insert a file", "prepend a file")
DROPPING = ("drop first chunk", "drop first file", "drop last chunk")
REPLACING = ("replace a file",)
COMPOSE_COLUMNS = [  # (heading, operations, how the arriving file relates to the array)
    ("Drop chunks from the front or the end", DROPPING, None),
    ("Add a file in a layout the array has", ADDING, "already in the array"),
    ("Add a file in a new layout", ADDING, "new to the array"),
    ("Replace a file, in a layout the array has", REPLACING, "already in the array"),
    ("Add or replace a file that changes how the others are described", ADDING + REPLACING, "changes how existing files are described"),
]


def compose_counts(arm: str, operations: tuple, layout: str | None = None) -> dict:
    out = {"composes": 0, "wrong values, no error": 0, "fails with an error": 0}
    for k, v in COMPOSE.get("cells", {}).items():
        a, _, op = k.split("|")
        if a == arm and op in operations and v["outcome"] in out and (layout is None or v.get("layout") == layout):
            out[v["outcome"]] += 1
    return out


def compose_n(arms: list[str], operations: tuple, layout: str | None, outcome: str | None = None) -> int:
    """Trials over several options: all of them, or those with one outcome."""
    return sum(sum(compose_counts(a, operations, layout).values()) if outcome is None else compose_counts(a, operations, layout)[outcome] for a in arms)


def compose_cell(arm: str, operations: tuple, layout: str | None = None) -> str:
    c = compose_counts(arm, operations, layout)
    n = sum(c.values())
    if not n:
        return ""
    if c["composes"] == n:
        return f'<span class="yes">&#10003; {"all " if n > 1 else ""}{n} correct</span>'
    if c["fails with an error"] == n:
        return f'{"all " if n > 1 else ""}{n} refused'
    parts = []
    if c["wrong values, no error"]:
        parts.append(f'<span class="no">&#10007; {c["wrong values, no error"]} wrong</span>')
    if c["composes"]:
        parts.append(f'{c["composes"]} correct')
    if c["fails with an error"]:
        parts.append(f'{c["fails with an error"]} refused')
    return ", ".join(parts)


def compose_table() -> str:
    head = "".join(f"<th>{h}</th>" for h, _, _ in COMPOSE_COLUMNS)
    body = ""
    for a in VIRTUAL:
        if a in COMPOSE.get("ordinary_arrays", []):
            body += f'<tr><td>{LABEL[a]}</td><td colspan="{len(COMPOSE_COLUMNS)}" class="sub0">an ordinary Zarr array, with nothing extra to keep in step with the references (not run)</td></tr>'
        elif any(k.startswith(a + "|") for k in COMPOSE.get("cells", {})):
            body += f"<tr><td>{LABEL[a]}</td>" + "".join(f"<td>{compose_cell(a, ops, layout)}</td>" for _, ops, layout in COMPOSE_COLUMNS) + "</tr>"
    return f"""<div class="scroll"><table><thead><tr><th>Option</th>{head}</tr></thead><tbody>{body}</tbody></table></div>"""


def compose_total(arm: str, operations: tuple | None = None, layout: str | None = None) -> dict:
    operations = operations or tuple(COMPOSE.get("operations", []))
    return compose_counts(arm, operations, layout)


# ------------------------------------------------------------------ growth


def scale(arm: str, kind: str, field: str, n: int = 16000):
    return SCALING.get(f"{kind}|{n}", {}).get(arm, {}).get(field)


def best(kind: str, field: str, n: int = 16000) -> float:
    """The smallest value among the options that keep the array virtual."""
    return min(v[field] for a, v in SCALING[f"{kind}|{n}"].items() if a != "native_copy" and v.get("outcome") == "equal")


def scaling_table() -> str:
    arms = [a for a in ARMS if scale(a, "alternate", "outcome") or scale(a, "packed", "outcome")]
    fields = [("zarr_json_bytes", size), ("virtual_bytes", size), ("open_seconds", lambda v: f"{v:.2f} s"), ("build_seconds", lambda v: f"{v:.0f} s"),
              ("microseconds_per_chunk", lambda v: f"{v / 1000:.2f} ms")]
    body = ""
    for a in arms:
        tds = ""
        for kind in ("alternate", "packed"):
            if scale(a, kind, "outcome") != "equal":
                tds += f'<td class="n dashcell" colspan="{len(fields)}">cannot describe it</td>'
                continue
            for field, fmt in fields:
                v = scale(a, kind, field)
                hot = a != "native_copy" and v >= 3 * best(kind, field) and not (field == "open_seconds" and v < 0.1)
                tds += f'<td class="n{" hot" if hot else ""}">{fmt(v)}</td>'
        body += f"<tr><td>{LABEL[a]}</td>{tds}</tr>"
    cols = '<th class="n">zarr.json</th><th class="n">All metadata, stored</th><th class="n">Open</th><th class="n">Write</th><th class="n">Read, per chunk</th>'
    return f"""<div class="scroll"><table><thead>
<tr><th></th><th class="n" colspan="{len(fields)}">Two layouts alternating chunk by chunk</th><th class="n" colspan="{len(fields)}">Every chunk packed its own way</th></tr>
<tr><th>Option</th>{cols}{cols}</tr></thead><tbody>{body}</tbody></table></div>"""


def spread() -> float:
    """The largest spread between repeats of the same read, as a fraction of the median."""
    return max((v.get("read_all_spread", 0) for run in SCALING.values() for v in run.values()), default=0)


def growth(arm: str, kind: str, field: str) -> str:
    return " → ".join(size(scale(arm, kind, field, n)) for n in (1000, 4000, 16000))


def per_chunk(arm: str, kind: str, n: int = 16000) -> float:
    return scale(arm, kind, "microseconds_per_chunk", n) / 1000


# ------------------------------------------------------------------ real granules


def perf_reads(which: str) -> list[str]:
    return list(next(v["reads"] for v in PERF[which]["arms"].values() if v.get("reads")))


def perf_table(which: str, whole_record_series: bool = False) -> str:
    perf = PERF[which]
    if not perf:
        return "<p class=\"note\">Not run.</p>"
    reads = perf_reads(which)
    short = []
    for r in reads:
        s = r.split(":")[0] + ":" + r.split(":")[1].split(",")[0]
        if whole_record_series and "10 years" in r:
            s = "series: every day at one cell (repeat)"
        short.append(s)
    head = "".join(f'<th class="n" title="{e(r)}">{e(s)}</th>' for r, s in zip(reads, short))
    body = ""
    base = perf["arms"].get("switch_position", {}).get("reads", {})
    for a, v in perf["arms"].items():
        if v.get("outcome") == "not expressible":
            body += f'<tr><td>{LABEL[a]}</td><td class="dashcell" colspan="{len(reads) + 2}">cannot describe it: {e(v["why"])}</td></tr>'
            continue
        if not v.get("reads"):
            continue
        tds = ""
        same_requests = True
        for r in reads:
            x = v["reads"][r]
            ok = "" if x["equal_to_files"] else ' <span class="no">&#10007;</span>'
            tds += f'<td class="n">{x["seconds"]:.2f} s{ok}</td>'
            same_requests &= x.get("source_requests") == base.get(r, {}).get("source_requests")
        req = '<span class="yes">same</span>' if same_requests else " · ".join(f"{v['reads'][r].get('source_requests'):,}" for r in reads)
        body += f'<tr><td>{LABEL[a]}</td>{tds}<td class="n">{req}</td><td class="n">{size(v.get("manifest_bytes"))}</td></tr>'
    requests = " · ".join(f"{base[r].get('source_requests'):,}" for r in reads) if base else ""
    runs = sorted({x["runs"] for v in perf["arms"].values() for x in v.get("reads", {}).values()})
    return f"""<div class="scroll"><table><thead><tr><th>Option</th>{head}<th class="n">Source requests</th><th class="n">Manifests</th></tr></thead>
<tbody>{body}</tbody></table></div>
<p class="note">Source requests for the four reads, left to right: {requests}. &ldquo;Same&rdquo; means the option made exactly that many for every read. Times are the median of {' or '.join(str(r) for r in runs)} runs.</p>"""


def seconds(which: str, arm: str, read: str = "series: every day at one cell") -> float:
    return PERF[which]["arms"][arm]["reads"][read]["seconds"]


def requests(which: str, arm: str, read: str = "series: every day at one cell") -> int:
    return PERF[which]["arms"][arm]["reads"][read]["source_requests"]


def manifest(which: str, arm: str) -> int:
    return PERF[which]["arms"][arm]["manifest_bytes"]


def tensorstore_sentence() -> str:
    runs = TS.get("runs", [])
    if not runs:
        return ""
    last = runs[-1]
    return (f"TensorStore's stack driver, given {last['layers']:,} one-chunk native arrays as layers, opened in {last['open_seconds']:.2f} s and read everything "
            f"at {last['microseconds_per_layer'] / 1000:.2f} ms per layer, from a {size(last['spec_bytes'])} spec.")


if __name__ == "__main__":
    from page import render  # the page's text lives beside this file

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(globals()))
    print("wrote", OUT)
