"""Build the storybook page from the evaluation's result files.

    python storybook.py        # writes ../storybook/index.html (page content)

Two ways in: pick an implementation and read what follows from it, or pick a use case
and see which implementations serve it. Every verdict is computed here from
results/eval/*.json; the words around the verdicts are written in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "eval"
OUT = HERE.parent / "storybook" / "index.html"
BRIEF = HERE.parent / "brief" / "index.html"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


MATRIX, UNAWARE, COMPOSE, SCALING, KERCHUNK = (load(n) for n in ("matrix.json", "unaware.json", "compose.json", "scaling.json", "without_icechunk.json"))
PERF = {"pr": load("perf_pr_tmin.json"), "na": load("perf_na_dayl.json")}

# name, group, how it works, what it takes, how it was tested
OPTIONS = {
    "switch_key_parameters": ("Layout table · key and numbers in each reference", "A table of layouts",
        "The array's codec holds a short table of layouts: the ways a chunk in this array can be stored. Every chunk reference begins with a key naming its layout, followed by any numbers that are that chunk's own (a scale and offset, say).",
        ["A new codec in every Zarr reader", "A new kind of Icechunk reference: a literal followed by byte ranges"], "Built in the zarr-python, Icechunk and VirtualiZarr branches, with typed numbers after the key. It is refkv PR #73's design, and refkv itself was also run."),
    "switch_key": ("Layout table · 8-byte key in each reference", "A table of layouts",
        "The array's codec holds a short table of layouts: the ways a chunk in this array can be stored. Every chunk reference begins with an 8-byte key, the start of a hash of its layout's content. Shorter keys are allowed and collide sooner.",
        ["A new codec in every Zarr reader", "A new kind of Icechunk reference: a literal followed by byte ranges"], "Built in the zarr-python, Icechunk and VirtualiZarr branches. The content-hash key is refkv PR #73's."),
    "switch_case_id": ("Layout table · 1-byte number in each reference", "A table of layouts",
        "A one-byte key that numbers the layouts in the order they first appear, instead of hashing them. It is here as a caution: the codec in the branch refuses a key that is not a hash of its layout.",
        ["A new codec in every Zarr reader", "A new kind of Icechunk reference: a literal followed by byte ranges"], "Built as an evaluation codec. This is the per-chunk variant of NASA-IMPACT/veda-odd#498, read literally."),
    "switch_position": ("Layout table · found by position", "A table of layouts",
        "The array's codec holds a short table of layouts, and zarr.json also lists which layout each run of chunks along the joined axis uses. Nothing is stored per chunk; the codec is told which chunk it is decoding.",
        ["A new codec in every Zarr reader", "Zarr readers must tell a codec the chunk's coordinates", "Icechunk change only when the chunk shape changes (virtual shards)"], "Built in the zarr-python and VirtualiZarr branches."),
    "switch_store_context": ("Layout table · store answers a context call", "A table of layouts",
        "The store gains a method that returns a chunk's context (its layout and numbers) beside its bytes. This is the &ldquo;Zarr v3.2&rdquo; design of zarr-specs#399.",
        ["A new codec in every Zarr reader", "A new method on the Zarr store interface, and a spec change", "Somewhere in the Icechunk manifest to keep the context"], "Emulated: a Python wrapper around the store answers the call from a table indexed by position."),
    "switch_sidecar": ("Layout table · codec reads side arrays", "A table of layouts",
        "Two more arrays in the store list every chunk's layout and its numbers. The codec reads them itself.",
        ["A new codec that does its own I/O, in every Zarr reader", "Zarr readers must tell a codec the chunk's coordinates", "Icechunk change only when the chunk shape changes"], "Built as an evaluation codec."),
    "switch_index_prefix": ("Layout table · store prepends the chunk index", "A table of layouts",
        "On every read the store puts the chunk's grid index in front of its bytes, and the codec looks the layout up from that.",
        ["A new codec in every Zarr reader", "Icechunk must prepend the index for these arrays and no others"], "Built: an Icechunk branch prepends the index for arrays whose codec asks."),
    "conditional_codec": ("conditional codec", "A narrower codec",
        "zarr-extensions' conditional codec: a one-byte header in front of each chunk says which of a list of compressors to apply.",
        ["The conditional codec in every Zarr reader", "A new kind of Icechunk reference: a literal followed by a byte range"], "Built from my reading of the merged spec."),
    "conditional_with_grid_shapes": ("conditional + encoded chunk shapes", "A narrower codec",
        "Two narrow proposals used together: conditional picks the compressors, and the chunk grid states how large each chunk is stored so padded chunks can be cropped.",
        ["The conditional codec and a change to the rectilinear chunk grid, in every Zarr reader", "A new kind of Icechunk reference: a literal followed by a byte range"], "Built from the two proposals; the grid part is emulated."),
    "array_concat": ("Array concatenation codec", "Keep each file's array",
        "Each file stays an array of its own. One more array lists them in its codec and reads through to them.",
        ["A codec that opens other arrays, in every Zarr reader", "Nothing from the store"], "Built in the zarr-python and VirtualiZarr branches."),
    "zarr_reference": ("Reference-per-chunk codec", "Keep each file's array",
        "Each file stays an array of its own. The joined array has one chunk per file, and that chunk's bytes are a reference to the file's array.",
        ["A codec that opens other arrays, in every Zarr reader", "Nothing from the store"], "Built as an evaluation codec."),
    "grid_encoded_shapes": ("Encoded chunk shapes in the grid", "Change the grid, not the codec",
        "The chunk grid states how large each chunk is stored, beside how much of it counts.",
        ["A change to the rectilinear chunk grid, in every Zarr reader"], "Emulated: declared in an attribute and read by a patched grid."),
    "domain_convention": ("Declared valid domain", "Change the grid, not the codec",
        "The array is stored with the files' padding in it. An attribute says which ranges are data.",
        ["A convention every reader must follow to see the right array"], "Built as a reader-side helper."),
    "materialize_odd_chunks": ("Copy the chunks that do not fit", "Copy data",
        "The array takes the commonest layout. Every chunk that does not fit it is decoded and written again as an ordinary chunk.",
        ["Nothing new: released zarr-python and Icechunk read it"], "Built, for any chunk that does not fit the commonest layout."),
    "native_copy": ("Copy everything to native Zarr", "Copy data",
        "Not virtual at all: read every value and write a new array.",
        ["Nothing new", "Storage and compute for a full second copy"], "Built."),
}
CASES = {
    "codec_eras": ("The compression changed partway through the record", "Daymet Puerto Rico added a shuffle filter in 2020. zarr-specs#399 cites GOES-16, with four eras."),
    "file_lengths": ("Files have different lengths", "Monthly files of 31, 28 and 31 days, each stored as one chunk."),
    "padded_last_chunk": ("Each file's last chunk is stored whole", "HDF5 and netCDF4 pad the final chunk. Daymet North America from 2020: 365 days in chunks of 10."),
    "chunk_shape_change": ("The chunk shape changed", "Daymet North America went from 1 &times; 1000 &times; 1000 to 10 &times; 300 &times; 300 in 2020."),
    "data_types": ("The data type changed", "int16 files followed by float32 files. Raised in zarr-specs#399."),
    "packing_per_file": ("Every file has its own scale and offset", "zarr-specs#399 cites BOM BARPA, where each file is packed with its own scale_factor and add_offset."),
    "packing_per_chunk": ("Every chunk has its own scale and offset", "How GRIB packs each message. The worst case for anything that lists layouts in metadata."),
    "layouts_alternate": ("Two layouts alternate from file to file", "A record where reprocessed and original granules are interleaved. Constructed for this test."),
    "fill_value_differs": ("Files declare different fill values and leave chunks unwritten", "Constructed for this test. No sampled NASA granule had it."),
    "variable_absent": ("A file does not have the variable", "Seen in the sampled NASA granules, where some lack a variable the others carry."),
}
NEEDS = {
    "need_released_readers": ("It must read with software people already have", "Released zarr-python and released Icechunk, with nothing new installed."),
    "need_manifest_ops": ("We insert, backfill, replace or drop files after the store is built", "Files are added, replaced or dropped by editing the stored references, with a tool that updates nothing else the option keeps (a table, a list of pieces, a side array)."),
    "need_no_icechunk": ("It must work without Icechunk", "Stored as Kerchunk references, which are either inline bytes or one byte range."),
    "need_many_granules": ("The record is tens of thousands of small granules", "16,000 one-chunk pieces, in the two worst cases: layouts that alternate, and every chunk packed its own way."),
}
PLAIN_REFS = {  # by inspection, where no Kerchunk run exists
    "switch_sidecar": (None, "its references are plain byte ranges plus two more arrays, except where the chunk shape changes, which needs a literal followed by ranges"),
    "array_concat": (True, "each file's array uses plain byte ranges"),
    "zarr_reference": (True, "plain byte ranges, plus one small inline chunk per file"),
    "grid_encoded_shapes": (True, "its references are plain byte ranges"),
    "domain_convention": (True, "its references are plain byte ranges"),
    "switch_case_id": (False, "every reference is a literal followed by a byte range"),
    "switch_key_parameters": (False, "every reference is a literal followed by a byte range"),
    "conditional_codec": (False, "every reference is a literal followed by a byte range"),
    "conditional_with_grid_shapes": (False, "every reference is a literal followed by a byte range"),
    "switch_index_prefix": (False, "the store itself must prepend the index"),
    "switch_store_context": (False, "the store itself must answer the context call"),
    "materialize_odd_chunks": (None, "the copied chunks would have to sit inline in the reference file, which is practical only while they are few and small"),
    "native_copy": (False, "it is a store of native chunks"),
}


def size(n: float) -> str:
    for unit, scale in (("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if n < 100 * scale else f"{n / scale:.0f} {unit}"
    return f"{n:.0f} B"


FIELDS = {  # what is compared between options in the worst cases, and how it is said
    "virtual_bytes": lambda v, b: f"{size(v)} stored beside the chunk data, {v / b:.0f}&times; the smallest",
    "build_seconds": lambda v, b: f"{v:.0f} s to write, {v / b:.0f}&times; the quickest",
    "microseconds_per_chunk": lambda v, b: f"{v / 1000:.1f} ms per chunk to read everything, {v / b:.0f}&times; the quickest",
    "open_seconds": lambda v, b: f"{v:.2f} s to open, because zarr.json is {{doc}}",
}
RULE = "at least three times the best option that keeps the array virtual, in the same worst case"


def build() -> dict:
    data: dict = {"options": {}, "cases": {}, "needs": {}, "rule": RULE}
    best = {kind: {f: min(v[f] for a, v in SCALING[f"{kind}|16000"].items() if a != "native_copy" and v.get("outcome") == "equal") for f in FIELDS}
            for kind in ("alternate", "packed")}
    for arm, (name, group, how, takes, tested) in OPTIONS.items():
        o: dict = {"name": name, "group": group, "how": how, "takes": takes, "tested": tested, "cases": {}, "costs": [], "warnings": [], "good": []}
        for case in CASES:
            cell = MATRIX["cells"][f"{arm}|{case}"]
            entry = {"ok": cell["outcome"] == "equal", "why": cell.get("why", "")}
            copied = cell.get("copied_fraction")
            if entry["ok"] and copied:
                entry["cost"] = f"{copied * 100:.0f}% of the data is stored a second time"
                entry["heavy"] = True  # any copying is a cost; the page gives the fraction
            o["cases"][case] = entry
        # growth with the number of chunks, against the best option in the same worst case
        scale = {}
        for kind, label in (("alternate", "two layouts alternate chunk by chunk"), ("packed", "every chunk is packed its own way")):
            rec = SCALING[f"{kind}|16000"].get(arm)
            if not rec:
                continue
            if rec["outcome"] != "equal":
                why = rec.get("why", "")
                if arm == "switch_case_id":
                    why = "one byte numbers at most 256 layouts, and this case has 16,000"
                scale[kind] = {"ok": False, "why": why}
                continue
            notes = []
            for f, say in FIELDS.items():
                v, b = rec[f], best[kind][f]
                if arm != "native_copy" and v >= 3 * b and not (f == "open_seconds" and v < 0.1):
                    notes.append(say(v, b).replace("{doc}", size(rec["zarr_json_bytes"])))
            scale[kind] = {"ok": True, "notes": notes, "zarr_json": size(rec["zarr_json_bytes"]), "stored": size(rec["virtual_bytes"])}
            if notes:
                o["costs"].append(f"With 16,000 chunks where {label}: " + "; ".join(notes) + ".")
        o["scale"] = scale
        if arm == "switch_case_id":
            o["costs"].append("One byte numbers at most 256 layouts. The hashed key avoids both this limit and the renumbering.")
        if arm == "native_copy":
            o["costs"].append("All of the data is stored a second time, and must be rewritten whenever the source is reprocessed.")
        # real data
        pr = PERF["pr"]["arms"].get(arm, {})
        if pr.get("outcome") == "equal":
            series = pr["reads"]["series: every day at one cell"]
            ref = PERF["pr"]["arms"]["switch_position"]["reads"]["series: every day at one cell"]
            same_requests = series["source_requests"] == ref["source_requests"]
            o["good"].append(f"On twelve years of Daymet Puerto Rico a full time series took {series['seconds']:.0f} s and {series['source_requests']:,} source requests"
                             + (", the same requests as every other option." if same_requests else "."))
        na = PERF["na"]["arms"].get(arm, {})
        if na.get("outcome") == "equal":
            o["good"].append("It read Daymet North America across the 2020 chunk-shape change with every value equal to the files.")
        # readers without the feature
        stock = [v["released zarr-python, released Icechunk"]["outcome"] for k, v in UNAWARE.items() if k.startswith(arm + "|")]
        knows = [v["zarr-python with the codecs, released Icechunk"]["outcome"] for k, v in UNAWARE.items() if k.startswith(arm + "|")]
        if "WRONG VALUES" in stock:
            o["released"] = {"verdict": "bad", "text": "A reader without the convention returns an array of the wrong shape, with the padding in it, and no error."}
            o["warnings"].append(o["released"]["text"])
        elif all(s == "reads correctly" for s in stock):
            o["released"] = {"verdict": "good", "text": "Released zarr-python and Icechunk read it correctly as it is."}
            o["good"].append(o["released"]["text"])
        elif all(s == "refuses" for s in stock):
            o["released"] = {"verdict": "bad", "text": "Released zarr-python refuses it with a clear error (unknown codec). Nobody reads wrong data, but nobody can read it without the new codec."}
        else:
            o["released"] = {"verdict": "partial", "text": f"Released zarr-python read {stock.count('reads correctly')} of {len(stock)} stores and refused the rest with an unrelated reshape error."}
        if knows and all(k == "reads correctly" for k in knows) and o["released"]["verdict"] != "good":
            o["good"].append("Once a reader has the codec, released Icechunk serves it: no store change is needed.")
        elif knows and "reads correctly" in knows and "WRONG VALUES" not in knows and o["released"]["verdict"] == "bad":
            o["good"].append(f"Once a reader has the codec, released Icechunk serves it in {knows.count('reads correctly')} of {len(knows)} cases; the rest need the Icechunk change.")
        # operations on the manifest
        if arm in COMPOSE["ordinary_arrays"]:
            o["compose"] = {"verdict": "good", "text": "It is an ordinary Zarr array, with nothing extra to keep in step with the references. (By inspection, not a run.)"}
        else:
            cells = {k: v for k, v in COMPOSE["cells"].items() if k.startswith(arm + "|") and v["outcome"] != "not applicable"}

            def tally(pick) -> tuple[int, int, int, int]:
                chosen = [v for k, v in cells.items() if pick(k.split("|")[2], v)]
                ok = sum(v["outcome"] == "composes" for v in chosen)
                wrong = sum(v["outcome"] == "wrong values, no error" for v in chosen)
                return len(chosen), ok, wrong, len(chosen) - ok - wrong

            def say(what: str, counts: tuple[int, int, int, int]) -> str:
                n, ok, wrong, errors = counts
                if not n:
                    return ""
                if ok == n:
                    return f" {what}: read correctly in all {n} trials."
                if errors == n:
                    return f" {what}: raised an error in all {n} trials."
                parts = [f"read correctly in {ok} of {n} trials"] if ok else []
                if wrong:
                    parts.append(f"returned wrong values with no error in {wrong}{'' if ok else f' of {n} trials'}")
                if errors:
                    parts.append(f"raised an error in {errors}")
                return f" {what}: " + ", ".join(parts) + "."

            adding = ("append a file", "insert a file", "prepend a file")
            drops = tally(lambda op, v: op.startswith("drop"))
            known = tally(lambda op, v: op in adding and v.get("layout") == "already in the array")
            new = tally(lambda op, v: op in adding and v.get("layout") == "new to the array")
            swapped = tally(lambda op, v: op == "replace a file" and v.get("layout") == "already in the array")
            replanned = tally(lambda op, v: v.get("layout") == "changes how existing files are described")
            text = ("Each edit changed only the references of the files involved and updated nothing else."
                    + say("Dropping chunks from the front or the end", drops)
                    + say("Adding a file (appended, inserted or backfilled) whose layout the array already has", known)
                    + say("Replacing a file with another of the same length, in a layout the array has", swapped)
                    + say("Adding a file in a layout new to the array", new)
                    + say("Adding or replacing a file that changes how the others must be described", replanned))
            silent = drops[2] + known[2] + new[2] + swapped[2] + replanned[2]
            everyday = [c for c in (drops, known, swapped) if c[0]]
            if silent:
                verdict = "bad"
                text += " Wrong values with no error are the outcome to avoid."
            elif all(c[1] == c[0] for c in everyday):
                verdict = "good"
                if new[0] and new[1] < new[0]:
                    text += " A new layout must be added to its table first; until then reads raise an error."
            elif not any(c[1] for c in everyday):
                verdict = "bad"
                text += " It never returned wrong values, but no edit works until its own table or list is rewritten too."
            else:
                verdict = "partial"
                text += " Where it failed it raised an error. Every tool that edits the store must also rewrite what the option keeps for itself."
            o["compose"] = {"verdict": verdict, "text": text}
            if arm == "zarr_reference":
                o["compose"]["text"] += " Its chunks are whole files, so it cannot drop less than a file."
            if arm == "switch_position":
                o["compose"]["text"] += " The codec checks that its table covers the array, which is why edits that add or remove chunks raise an error. A replaced file of the same length cannot be checked."
            if arm == "switch_store_context":
                o["compose"]["text"] += " This is the emulation, whose table is indexed by position; a store that kept the context with each reference would carry it along, and that was not tested."
            if arm in ("grid_encoded_shapes", "conditional_with_grid_shapes"):
                o["compose"]["text"] += " The encoded sizes sit in an attribute here; the proposal puts them in the chunk grid, which such a tool rewrites."
        if o["compose"]["verdict"] != "good":
            o["warnings"].append(o["compose"]["text"])
        elif arm not in COMPOSE["ordinary_arrays"]:
            o["good"].append(o["compose"]["text"])
        # without Icechunk
        runs = [v for k, v in KERCHUNK["cells"].items() if k.startswith(arm + "|")]
        if runs:
            ok = sum(v["outcome"] == "equal" for v in runs)
            if ok:
                o["kerchunk"] = {"verdict": "good" if ok == len(runs) else "partial", "text": f"As Kerchunk references it read correctly in {ok} of {len(runs)} cases; the exception is a chunk-shape change."}
            else:
                o["kerchunk"] = {"verdict": "bad", "text": "It cannot be written as Kerchunk references: every chunk is a literal followed by a byte range, and a Kerchunk reference is one or the other."}
        else:
            fits, why = PLAIN_REFS[arm]
            o["kerchunk"] = {"verdict": {True: "good", False: "bad", None: "partial"}[fits], "text": {True: "Yes: ", False: "No: ", None: "In part: "}[fits] + why + ". (By inspection, not a run.)"}
        if "mulated" in tested:
            o["warnings"].append("This option was emulated in Python, in whole or in part, not built into Icechunk or zarr-python. A real implementation could behave differently.")
        data["options"][arm] = o
    for case, (name, example) in CASES.items():
        data["cases"][case] = {"name": name, "example": example}
    for need, (name, detail) in NEEDS.items():
        data["needs"][need] = {"name": name, "detail": detail}
    return data


CSS = """
.paths { display: flex; gap: 0; border-bottom: 2px solid var(--ink); flex-wrap: wrap; }
.paths button { font: inherit; font-family: var(--display); font-weight: 600; font-size: 17px; padding: 12px 20px 10px; background: none; border: 0; border-bottom: 4px solid transparent; margin-bottom: -2px; color: var(--ink-3); cursor: pointer; }
.paths button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
.book { display: grid; grid-template-columns: minmax(230px, 320px) minmax(0, 1fr); gap: 36px; align-items: start; }
.picker { display: flex; flex-direction: column; gap: 18px; position: sticky; top: calc(env(safe-area-inset-top, 0px) + 12px); max-height: calc(100vh - 24px); overflow-y: auto; padding-right: 4px; }
.picker h3 { font-family: var(--mono); font-size: 11.5px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.picker .group { display: flex; flex-direction: column; gap: 6px; }
.pick { font: inherit; font-size: 14.5px; line-height: 1.35; text-align: left; padding: 9px 12px; background: var(--surface); color: var(--ink); border: 1px solid var(--rule); border-radius: 6px; cursor: pointer; }
.pick:hover { border-color: var(--ink-3); }
.pick[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); font-weight: 500; }
.story { display: flex; flex-direction: column; gap: 26px; min-width: 0; }
.story > header { display: flex; flex-direction: column; gap: 8px; }
.story h2 { font-size: 28px; }
.story .how { font-size: 17px; color: var(--ink-2); max-width: 62ch; }
.beat { display: flex; flex-direction: column; gap: 10px; padding-top: 18px; border-top: 1px solid var(--rule); }
.beat > h3 { font-family: var(--display); font-size: 19px; font-weight: 600; display: flex; gap: 10px; align-items: baseline; }
.beat > h3 .mark { font-family: var(--mono); font-size: 15px; width: 1.2em; flex: none; }
.beat.good > h3 .mark { color: var(--good); }
.beat.bad > h3 .mark { color: var(--bad); }
.beat.cost > h3 .mark, .beat.warn > h3 .mark { color: var(--warn); }
.beat ul { max-width: 70ch; }
.beat li b { font-weight: 600; }
.beat .why { display: block; color: var(--ink-2); font-size: 14px; }
.beat p { max-width: 70ch; }
.verdicts { display: flex; flex-direction: column; gap: 0; }
.verdict { display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 4px 6px; padding: 11px 0; border-bottom: 1px solid var(--rule); }
.verdict .mark { font-family: var(--mono); font-weight: 600; }
.verdict.good .mark { color: var(--good); } .verdict.bad .mark { color: var(--bad); } .verdict.partial .mark { color: var(--warn); }
.verdict .who { font-weight: 500; }
.verdict .who button { font: inherit; font-weight: 500; color: var(--accent); background: none; border: 0; padding: 0; cursor: pointer; text-align: left; text-decoration: underline; text-underline-offset: 3px; }
.verdict .because { grid-column: 2; color: var(--ink-2); font-size: 14px; }
.count { font-family: var(--mono); font-size: 13px; color: var(--ink-3); font-weight: 400; margin-left: auto; white-space: nowrap; }
.start { color: var(--ink-2); font-size: 17px; max-width: 56ch; padding-top: 8px; }
@media (max-width: 820px) {
  .book { grid-template-columns: 1fr; gap: 22px; }
  .picker { position: static; max-height: none; overflow: visible; }
  .picker .group { flex-direction: row; flex-wrap: wrap; }
  .story h2 { font-size: 23px; }
  .paths button { font-size: 15px; padding: 10px 12px 8px; }
}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html !== undefined) n.innerHTML = html; return n; };
const slug = (k) => k.replace(/_/g, '-');
const unslug = (s) => s.replace(/-/g, '_');
const state = { path: 'choice', choice: null, subject: null };
const picker = document.getElementById('picker'), story = document.getElementById('story');

function beat(kind, mark, title, count) {
  const b = el('section', 'beat ' + kind);
  const h = el('h3', '', `<span class="mark" aria-hidden="true">${mark}</span><span>${title}</span>` + (count ? `<span class="count">${count}</span>` : ''));
  b.append(h); return b;
}
function list(items) { const ul = el('ul'); items.forEach((i) => ul.append(el('li', '', i))); return ul; }

function renderChoice(key) {
  const o = D.options[key];
  story.replaceChildren();
  const head = el('header');
  head.append(el('div', 'eyebrow', 'Option'), el('h2', '', o.name), el('p', 'how', o.how));
  story.append(head);
  const ok = Object.keys(D.cases).filter((c) => o.cases[c].ok && !o.cases[c].heavy);
  const heavy = Object.keys(D.cases).filter((c) => o.cases[c].ok && o.cases[c].heavy);
  const no = Object.keys(D.cases).filter((c) => !o.cases[c].ok);
  const all = ok.length + heavy.length;
  const g = beat('good', '&#10003;', all === 10 ? 'It can describe all 10 test cases' : `It can describe ${all} of the 10 test cases`, `${all} / 10`);
  g.append(list(ok.concat(heavy).map((c) => `<b>${D.cases[c].name}.</b>` + (o.cases[c].cost ? `<span class="why">${o.cases[c].cost}.</span>` : ''))));
  if (o.good.length) g.append(list(o.good));
  story.append(g);
  if (no.length) {
    const b = beat('bad', '&#10007;', 'It cannot describe these', `${no.length} / 10`);
    b.append(list(no.map((c) => `<b>${D.cases[c].name}.</b><span class="why">${o.cases[c].why[0].toUpperCase() + o.cases[c].why.slice(1)}.</span>`)));
    story.append(b);
  }
  const costs = o.costs.concat(heavy.map((c) => `${D.cases[c].name}: ${o.cases[c].cost}.`));
  if (costs.length) { const c = beat('cost', '!', 'Where it gets expensive'); c.append(list(costs)); story.append(c); }
  else {
    const ran = ['alternate', 'packed'].filter((k) => o.scale[k]);
    const could = ran.filter((k) => o.scale[k].ok);
    if (could.length) {
      const c = beat('good', '&#10003;', 'Nothing stood out as expensive');
      const which = could.length === 2 ? 'both worst cases' : (could[0] === 'alternate' ? 'the worst case where two layouts alternate' : 'the worst case where every chunk is packed its own way');
      c.append(el('p', '', `With 16,000 chunks in ${which}, its stored size, write time, open time and read time were all within three times the best option's.` + (could.length < ran.length ? ' It cannot describe the other worst case at all.' : '')));
      story.append(c);
    }
  }
  if (o.warnings.length) { const w = beat('warn', '!', 'Watch out'); w.append(list(o.warnings)); story.append(w); }
  const t = beat('', '', 'What it takes'); t.append(list(o.takes));
  t.append(el('p', '', `<b>Readers people already have.</b> ${o.released.text}`));
  t.append(el('p', '', `<b>Without Icechunk.</b> ${o.kerchunk.text}`));
  t.append(el('p', 'note', `How it was tested: ${o.tested}`));
  story.append(t);
}

function verdictFor(key, subject) {
  const o = D.options[key];
  if (D.cases[subject]) {
    const c = o.cases[subject];
    if (!c.ok) return ['bad', c.why[0].toUpperCase() + c.why.slice(1) + '.'];
    if (c.cost) return [c.heavy ? 'partial' : 'good', c.cost[0].toUpperCase() + c.cost.slice(1) + '.'];
    const s = subject === 'packing_per_chunk' ? o.scale.packed : subject === 'layouts_alternate' ? o.scale.alternate : null;
    if (s && s.ok && s.notes.length) return ['partial', 'It works on a small record. At 16,000 chunks: ' + s.notes.join('; ') + '.'];
    if (s && !s.ok) return ['partial', 'It works on the small test record but not at 16,000 chunks: ' + s.why + '.'];
    return ['good', s && s.ok ? `At 16,000 chunks: ${s.zarr_json} of zarr.json to read, ${s.stored} stored beside the chunk data.` : ''];
  }
  if (subject === 'need_released_readers') return [o.released.verdict, o.released.text];
  if (subject === 'need_manifest_ops') return [o.compose.verdict, o.compose.text];
  if (subject === 'need_no_icechunk') return [o.kerchunk.verdict, o.kerchunk.text];
  const a = o.scale.alternate, p = o.scale.packed;
  if (!a && !p) return ['bad', 'Not run at this size: it describes neither worst case, or copies the data.'];
  const parts = [], bad = [];
  [['layouts alternate', a], ['every chunk is packed its own way', p]].forEach(([label, s]) => {
    if (!s) return;
    if (!s.ok) bad.push(`cannot describe the case where ${label}`);
    else if (s.notes.length) parts.push(`where ${label}: ${s.notes.join('; ')}`);
  });
  if (!parts.length && !bad.length) return ['good', `Within three times the best option on every measure. In the harder case: ${p.zarr_json} of zarr.json to read, ${p.stored} stored beside the chunk data.`];
  return [bad.length && !parts.length && bad.length === 2 ? 'bad' : 'partial', (bad.concat(parts).join('. ') + '.').replace(/^./, (m) => m.toUpperCase())];
}

function renderSubject(key) {
  const s = D.cases[key] || D.needs[key];
  story.replaceChildren();
  const head = el('header');
  head.append(el('div', 'eyebrow', D.cases[key] ? 'Your files' : 'Your requirement'), el('h2', '', s.name), el('p', 'how', s.example || s.detail));
  story.append(head);
  const groups = { good: [], partial: [], bad: [] };
  Object.keys(D.options).forEach((k) => { const [v, because] = verdictFor(k, key); groups[v].push([k, because]); });
  const titles = { good: ['good', '&#10003;', 'These handle it'], partial: ['cost', '!', 'These handle it, at a cost'], bad: ['bad', '&#10007;', 'These do not'] };
  ['good', 'partial', 'bad'].forEach((v) => {
    if (!groups[v].length) return;
    const b = beat(...titles[v], `${groups[v].length} / ${Object.keys(D.options).length}`);
    const box = el('div', 'verdicts');
    groups[v].forEach(([k, because]) => {
      const row = el('div', 'verdict ' + v);
      const who = el('div', 'who'); const link = el('button', '', D.options[k].name); link.type = 'button';
      link.addEventListener('click', () => go('choice', k)); who.append(link);
      row.append(el('span', 'mark', v === 'good' ? '&#10003;' : v === 'bad' ? '&#10007;' : '!'), who);
      if (because) row.append(el('div', 'because', because));
      box.append(row);
    });
    b.append(box); story.append(b);
  });
}

function renderPicker() {
  picker.replaceChildren();
  const add = (title, entries, current, path) => {
    const g = el('div', 'group-wrap'); g.append(el('h3', '', title));
    const box = el('div', 'group');
    entries.forEach(([k, name]) => { const b = el('button', 'pick', name); b.type = 'button'; b.setAttribute('aria-pressed', String(k === current)); b.addEventListener('click', () => go(path, k)); box.append(b); });
    g.append(box); picker.append(g);
    g.style.display = 'flex'; g.style.flexDirection = 'column'; g.style.gap = '8px';
  };
  if (state.path === 'choice') {
    const byGroup = {};
    Object.entries(D.options).forEach(([k, o]) => (byGroup[o.group] = byGroup[o.group] || []).push([k, o.name.replace('Layout table · ', '')]));
    Object.entries(byGroup).forEach(([g, entries]) => add(g, entries, state.choice, 'choice'));
  } else {
    add('How your files disagree', Object.entries(D.cases).map(([k, c]) => [k, c.name]), state.subject, 'subject');
    add('What you need', Object.entries(D.needs).map(([k, c]) => [k, c.name]), state.subject, 'subject');
  }
}

function render() {
  document.querySelectorAll('.paths button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.path === state.path)));
  renderPicker();
  const key = state.path === 'choice' ? state.choice : state.subject;
  if (!key) { story.replaceChildren(el('p', 'start', state.path === 'choice' ? 'Pick an option to see what follows from choosing it: which files it can describe, which it cannot, and where it gets expensive.' : 'Pick a way your files disagree, or something you need, to see which options handle it.')); return; }
  (state.path === 'choice' ? renderChoice : renderSubject)(key);
}

function go(path, key, quiet) {
  state.path = path;
  if (key) state[path === 'choice' ? 'choice' : 'subject'] = key;
  const k = path === 'choice' ? state.choice : state.subject;
  const hash = k ? (path === 'choice' ? 'choice-' : 'case-') + slug(k) : (path === 'choice' ? 'by-choice' : 'by-case');
  if (!quiet && location.hash.slice(1) !== hash) { try { history.replaceState(null, '', '#' + hash); } catch (e) { location.hash = hash; } }
  render();
  if (key && !quiet && window.matchMedia('(max-width: 820px)').matches) story.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
}

function fromHash() {
  const h = location.hash.slice(1);
  if (h.startsWith('choice-') && D.options[unslug(h.slice(7))]) return go('choice', unslug(h.slice(7)), true);
  if (h.startsWith('case-') && (D.cases[unslug(h.slice(5))] || D.needs[unslug(h.slice(5))])) return go('subject', unslug(h.slice(5)), true);
  if (h === 'by-case') return go('subject', null, true);
  if (!state.choice) state.choice = 'switch_key_parameters';
  go('choice', null, true);
}
document.querySelectorAll('.paths button').forEach((b) => b.addEventListener('click', () => go(b.dataset.path)));
window.addEventListener('hashchange', fromHash);
fromHash();
"""


def render() -> str:
    style = BRIEF.read_text()
    head = style[: style.index("</style>")].replace("<title>Real Collections and Virtual Shards</title>", "<title>Mismatched Granules Storybook</title>")
    data = json.dumps(build()).replace("</", "<\\/")
    return f"""{head}{CSS}</style>

<div class="wrap" style="gap: 32px">
<header>
  <section>
    <div class="eyebrow">Storybook &middot; revised 13 November 2026</div>
    <h1>Choose, and See What Follows</h1>
    <p class="lede">{len(OPTIONS)} ways to make one Zarr array from granules that disagree. Pick an option and read what follows from choosing it, or start from your files and see which options handle them.</p>
    <p class="byline">Every verdict is computed from the runs behind <a href="evaluation.html">Every Option, Tested</a>: ten small test cases, two worst cases at 16,000 chunks, and Daymet. &ldquo;Expensive&rdquo; means {RULE}. Only zarr-python read these arrays, and nothing ran against S3. Prepared with Claude.</p>
  </section>
</header>

<div class="paths" role="tablist" aria-label="Two ways in">
  <button type="button" role="tab" data-path="choice" id="tab-choice" aria-selected="true">Start from an option</button>
  <button type="button" role="tab" data-path="subject" id="tab-subject" aria-selected="false">Start from your files</button>
</div>

<div class="book">
  <nav class="picker" id="picker" aria-label="Choices"></nav>
  <article class="story" id="story" aria-live="polite"></article>
</div>
</div>

<script type="application/json" id="data">{data}</script>
<script>{JS}</script>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render())
    print("wrote", OUT, OUT.stat().st_size)
