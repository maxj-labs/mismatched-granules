"""Build the site's landing page from the result files.

    python site.py        # writes ../site/index.html (page content)

The page is the short version: the problem, the designs, what happens to each when a
store is edited, how often NASA records receive data away from the end, and what to
choose. Numbers come from results/eval/compose.json, results/eval/scaling.json,
results/prevalence.json and the timing results.
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
COMPOSE = json.loads((RESULTS / "eval" / "compose.json").read_text())
SCALING = json.loads((RESULTS / "eval" / "scaling.json").read_text())
PREVALENCE = json.loads((RESULTS / "prevalence.json").read_text())
PERF = json.loads((RESULTS / "eval" / "perf_pr_tmin.json").read_text())
FAST = json.loads((RESULTS / "eval" / "perf_pr_tmin_concurrency_120.json").read_text())
KERCHUNK = json.loads((RESULTS / "eval" / "without_icechunk.json").read_text())
DEMO = (ROOT / "repo" / "examples" / "reproduce.out").read_text()
DEMO_LINES = len((ROOT / "repo" / "examples" / "reproduce.py").read_text().splitlines())
e = html.escape

ADD = ("append a file", "insert a file", "prepend a file")
DROP = ("drop first chunk", "drop first file", "drop last chunk")
DESIGNS = [("switch_position", "Case found from position"), ("switch_key", "Case kept in each reference"),
           ("zarr_reference", "One array per file, joined by reference"), ("array_concat", "One array per run, joined by a list")]
REWRITE = {"switch_position": "its table is rewritten", "switch_key": "the new layout is added to its table",
           "array_concat": "its list of pieces is rewritten", "zarr_reference": "its references are rewritten"}
ROWS = [  # (label, operations, how the arriving file relates to the array)
    ("Append, insert or backfill a file whose layout the array already has", ADD, "already in the array"),
    ("Add a file in a layout new to the array", ADD, "new to the array"),
    ("Replace a file with a reprocessed one of the same length, in a layout the array has", ("replace a file",), "already in the array"),
    ("Drop chunks from the front or the end", DROP, None),
]


def cells(arm: str, operations: tuple, layout: str | None) -> list[dict]:
    out = []
    for key, cell in COMPOSE["cells"].items():
        a, _, op = key.split("|")
        if a == arm and op in operations and cell["outcome"] != "not applicable" and (layout is None or cell.get("layout") == layout):
            out.append(cell)
    return out


def tally(arm: str, operations: tuple, layout: str | None) -> tuple[int, int, int, int]:
    found = cells(arm, operations, layout)
    ok = sum(c["outcome"] == "composes" for c in found)
    wrong = sum(c["outcome"] == "wrong values, no error" for c in found)
    return len(found), ok, wrong, len(found) - ok - wrong


def verdict(arm: str, operations: tuple, layout: str | None) -> str:
    n, ok, wrong, errors = tally(arm, operations, layout)
    if not n:
        return '<span class="dash">not run</span>'
    if ok == n:
        return f'<span class="yes">&#10003; reads correctly</span><span class="sub">all {n} trials, nothing else rewritten</span>'
    if wrong:
        rest = "; ".join(p for p in (f"{errors} refused with an error" if errors else "", f"{ok} read correctly" if ok else "") if p)
        return f'<span class="no">&#10007; can return wrong values</span><span class="sub">{wrong} of {n} trials read wrong values with no error{"; " + rest if rest else ""}</span>'
    if ok:
        return f'<span class="warn">! sometimes</span><span class="sub">{ok} of {n} read correctly; {errors} refused with an error, none silently wrong</span>'
    return f'<span class="warn">! refused until {REWRITE[arm]}</span><span class="sub">all {n} trials failed with an error; none read wrong values</span>'


def operations_table() -> str:
    head = "".join(f"<th>{label}</th>" for _, label in DESIGNS)
    body = "".join(f"<tr><td>{label}</td>" + "".join(f"<td>{verdict(arm, ops, layout)}</td>" for arm, _ in DESIGNS) + "</tr>" for label, ops, layout in ROWS)
    return f'<div class="scroll"><table class="ops"><thead><tr><th>A writer edits the store and rewrites nothing else</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# ------------------------------------------------------------------ how often records receive data away from the end

ONE = 0.01
GROUPS = ["Older data was made in the last year", "Only short delays in the last year", "Older data was made earlier, none in the last year", "Nothing behind in the sample"]


def shares(rec: dict) -> tuple[float, float, float]:
    year = rec.get("last_12_months", {})
    return year.get("late_by_more_than_a_day", 0.0), year.get("late by more than 30 days", 0.0), rec["all"]["late by more than 30 days"]


def group(rec: dict) -> str:
    day, month, earlier = shares(rec)
    if month >= ONE:
        return GROUPS[0]
    if day >= ONE:
        return GROUPS[1]
    if earlier >= ONE:
        return GROUPS[2]
    return GROUPS[3]


def blind(rec: dict) -> str:
    """Why a sample could not have shown late data, if so."""
    made = (date.fromisoformat(rec["produced_to"]) - date.fromisoformat(rec["produced_from"])).days
    if made <= 7:
        return "all files made within a week"
    if abs(rec.get("median_days_from_data_to_production", 1)) < 0.05:
        return "production time equals data time"
    if rec["sampled"] >= 60_000 and rec["granules_in_cmr"] > rec["sampled"]:
        return "newest 60,000 granules only"
    return ""


def surveyed() -> list[dict]:
    return [r for r in PREVALENCE.values() if "all" in r]


def count(which: str) -> int:
    return sum(group(r) == which for r in surveyed())


def prevalence_table() -> str:
    rows = sorted(surveyed(), key=lambda r: (GROUPS.index(group(r)), -shares(r)[1], -shares(r)[0], -shares(r)[2]))
    body, current = "", None
    for r in rows:
        if group(r) != current:
            current = group(r)
            body += f'<tr class="grp"><th colspan="5">{current} &middot; {count(current)} collections</th></tr>'
        day, month, earlier = shares(r)
        version = str(r["version"]).lstrip("vV")
        notes = [f"v{e(version)}", f"data {r['data_from'][:4]}&ndash;{r['data_to'][:4]}", f"files made {r['produced_from'][:7]} to {r['produced_to'][:7]}"]
        if r.get("share_with_that_period", 1) < 0.98:
            notes.append("one product period of several")
        if current == GROUPS[3] and blind(r):
            notes.append(f"<b>{blind(r)}</b>")
        sampled = f'{r["with_production_time"]:,}' + ("" if r["with_production_time"] >= r["granules_in_cmr"] else f' <span class="sub0">of {r["granules_in_cmr"]:,}</span>')
        body += (f'<tr><td><span class="mono">{e(r["short_name"])}</span><span class="sub">{" &middot; ".join(notes)}</span></td>'
                 f'<td class="n">{sampled}</td><td class="n">{day * 100:.1f}%</td><td class="n">{month * 100:.1f}%</td><td class="n">{earlier * 100:.1f}%</td></tr>')
    return f"""<div class="scroll"><table class="survey"><thead><tr><th>Collection</th><th class="n">Granules sampled</th>
<th class="n">Last year of production:<br>over a day behind</th><th class="n">Last year of production:<br>over 30 days behind</th><th class="n">Whole sample:<br>over 30 days behind</th></tr></thead><tbody>{body}</tbody></table></div>"""


def scale(arm: str, kind: str, field: str) -> float:
    return SCALING[f"{kind}|16000"][arm][field]


def size(n: float) -> str:
    for unit, k in (("MB", 1e6), ("KB", 1e3)):
        if n >= k:
            return f"{n / k:.1f} {unit}" if n < 100 * k else f"{n / k:.0f} {unit}"
    return f"{n:.0f} B"


def render() -> str:
    brief = (ROOT / "brief" / "index.html").read_text()
    head = brief[: brief.index("</style>")].replace("<title>One Array from Mismatched Granules</title>", "<title>Mismatched Granules</title>")
    n = len(surveyed())
    series = "series: every day at one cell"
    table_arms = [a for a, v in PERF["arms"].items() if a.startswith("switch") and v.get("reads")]
    slow = [PERF["arms"][a]["reads"][series]["seconds"] for a in table_arms]
    fast = [FAST["arms"][a]["reads"][series]["seconds"] for a in table_arms if FAST["arms"].get(a, {}).get("reads")]
    kpos = [v["outcome"] for k, v in KERCHUNK["cells"].items() if k.startswith("switch_position|")]
    replace = tally("switch_position", ("replace a file",), "already in the array")
    differing = sum(1 for c in cells("switch_position", ("replace a file",), "already in the array") if not c.get("replacement_has_the_same_layout"))
    replans = tally("switch_key", ADD + ("replace a file",), "changes how existing files are described")
    short = tally("switch_case_id", ADD + DROP + ("replace a file",), None)
    nothing = [r for r in surveyed() if group(r) == GROUPS[3]]
    blind_count = sum(1 for r in nothing if blind(r))
    pm, km = PERF["arms"]["switch_position"]["manifest_bytes"], PERF["arms"]["switch_key"]["manifest_bytes"]
    ref_stored = scale("zarr_reference", "alternate", "virtual_bytes") / scale("switch_key", "alternate", "virtual_bytes")
    ref_write = scale("zarr_reference", "alternate", "build_seconds") / scale("switch_key", "alternate", "build_seconds")
    reads = cells("switch_key", DROP, None)[0]["reads"]
    css = """
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; border-top: 2px solid var(--ink); }
.card { display: flex; flex-direction: column; gap: 8px; padding: 20px 22px 8px 0; min-width: 0; }
.card + .card { padding-left: 22px; border-left: 1px solid var(--rule); }
.card .kind { font-family: var(--mono); font-size: 12px; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.06em; }
.card h3 { font-family: var(--display); font-size: 20px; }
.card h3 a { color: var(--ink); text-decoration-color: var(--accent); text-underline-offset: 4px; }
.card p { font-size: 14.5px; color: var(--ink-2); }
.sub { display: block; font-size: 12.5px; color: var(--ink-3); font-weight: 400; margin-top: 2px; }
.sub b { font-weight: 600; color: var(--ink-2); }
.sub0 { font-size: 12.5px; color: var(--ink-3); }
.dash { color: var(--ink-3); }
.warn { color: var(--warn); font-weight: 600; }
table.ops td { padding-top: 11px; padding-bottom: 11px; min-width: 150px; }
table.ops td:first-child { font-weight: 500; min-width: 200px; }
table.survey th { white-space: normal; vertical-align: bottom; }
table.survey tr.grp th { text-align: left; font-size: 13.5px; color: var(--ink); padding-top: 18px; border-bottom: 1px solid var(--ink-3); }
.designs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.design { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 18px 20px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.design h3 { font-family: var(--display); font-size: 18px; }
.design p { font-size: 14.5px; color: var(--ink-2); }
.strip { display: flex; gap: 3px; font-family: var(--mono); font-size: 11.5px; }
.strip span { flex: 1 1 0; min-width: 0; text-align: center; padding: 8px 2px; border-radius: 4px; border: 1px solid var(--rule); background: var(--bg); color: var(--ink-2); white-space: nowrap; overflow: hidden; }
.strip span.b { background: var(--accent-soft); border-color: var(--accent); color: var(--ink); }
.strip span b { font-weight: 600; color: var(--accent); margin-right: 3px; }
.meta { font-family: var(--mono); font-size: 12px; color: var(--ink-2); background: var(--bg); border: 1px dashed var(--ink-3); border-radius: 4px; padding: 7px 10px; }
.ask { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; max-width: 68ch; font-size: 17px; }
@media (max-width: 760px) { .cards, .designs { grid-template-columns: 1fr; } .card + .card { padding-left: 0; border-left: 0; border-top: 1px solid var(--rule); } .strip { font-size: 10px; } }
"""
    return f"""{head}{css}</style>

<div class="wrap">
<header>
  <section>
    <div class="eyebrow">maxj-labs / mismatched-granules</div>
    <h1>One Zarr Array from Granules That Disagree</h1>
    <p class="lede">Many archives of netCDF and HDF5 granules cannot be virtualized as one Zarr array, because the files change compression, chunk shape, data type or fill value partway through the record. This site tests the fixes proposed for Zarr, Icechunk and VirtualiZarr, and says which to choose for which kind of archive.</p>
    <p class="byline">Work in progress, prepared with Claude, revised 13 November 2026. Only zarr-python reads these arrays, the test records are small, and nothing has been run against S3.</p>
  </section>
</header>

<section>
  <h2>The problem, in one example</h2>
  <div class="prose">
    <p>Daymet publishes one netCDF file per year. Until 2019 each day's chunk was compressed one way; from 2020 a shuffle filter was added. A Zarr array has one codec chain for all its chunks, so the years cannot be presented as a single array without copying the data. The proposed fixes all let one array hold chunks encoded in more than one way. They differ in how a reader finds out which way a given chunk was encoded.</p>
  </div>
</section>

<section>
  <h2>Two places to keep that answer</h2>
  <div class="designs">
    <div class="design">
      <h3>In the array's metadata, by position</h3>
      <div class="meta">chunks 0&ndash;2: layout A &nbsp;&middot;&nbsp; chunks 3&ndash;5: layout B</div>
      <div class="strip"><span>2017</span><span>2018</span><span>2019</span><span class="b">2020</span><span class="b">2021</span><span class="b">2022</span></div>
      <p>The array records which run of chunks uses which layout. Nothing is stored with the chunks. It needs no change to the store unless the chunk shape changes, and it can be written as Kerchunk references.</p>
    </div>
    <div class="design">
      <h3>With each chunk, as a key in its reference</h3>
      <div class="meta">layouts: A, B</div>
      <div class="strip"><span><b>A</b>2017</span><span><b>A</b>2018</span><span><b>A</b>2019</span><span class="b"><b>B</b>2020</span><span class="b"><b>B</b>2021</span><span class="b"><b>B</b>2022</span></div>
      <p>Each chunk's reference begins with a few bytes naming its layout. The key goes wherever the reference goes. It needs Icechunk references that are a literal followed by byte ranges, which exist only on a branch, and it cannot be written as Kerchunk references.</p>
    </div>
  </div>
  <div class="prose">
    <p>Both are one codec holding a table of layouts. A different route keeps arrays apart and joins them: <b>one array per file, joined by reference</b> (each chunk of the joined array is a pointer to a file's own array), or <b>one array per run of like files, joined by a list</b> in the joining array's codec. These need nothing new from the store, and every reader needs a codec that opens other arrays.</p>
  </div>
</section>

<section>
  <h2>What happens when the store is edited</h2>
  <div class="prose">
    <p>All four read every test case correctly when the array is built in one go. They part ways when a store is edited afterwards. Each edit below was made by a writer that adds, removes or replaces chunk references, in whatever form the design gives a reference, and rewrites nothing the array already has: not the table, not the list, not another file's references.</p>
  </div>
  {operations_table()}
  <div class="prose">
    <p><b>By position, every edit that adds or removes chunks is refused until the table is rewritten.</b> The codec checks that its table covers the array's chunks and raises when it does not; that check was added during this work, and without it some of these edits read wrong values. One edit it cannot detect: a file replaced by another of the same length in a different layout. All {replace[2]} of the {differing} trials where the layouts differed read wrong values with no error.</p>
    <p><b>By key, nothing needs rewriting unless a layout is new.</b> A file in a layout the array already has can be appended, inserted, backfilled or swapped in. A file in a new layout needs one entry added to the table, and reads are refused until it is. The writer does have to put the key in each new reference, so it is not ignorant of the design; what it need not do is touch anything already stored.</p>
    <p><b>One array per file never needs anything rewritten,</b> because each file keeps its own metadata. It pays for that elsewhere: with 16,000 one-chunk files it stored {ref_stored:.0f}&times; as much beside the chunk data as the keyed design and took {ref_write:.0f}&times; as long to write, and nothing smaller than a whole file can be dropped.</p>
    <p class="note">Each trial reads the whole array, each file's part on its own, and single cells ({reads} reads); it counts as correct only if every read equals the files, and as wrong if any read returns wrong values silently. Trials are the test cases each design can describe. A file that changes how the others must be described (a different fill value, data type or chunk shape) forces the array to be planned again in the table designs: {replans[3]} of {replans[0]} such trials were refused by key. Keys here are 8-byte hashes of the layout; 1-byte numbered keys are not safe to edit this way ({short[2]} of {short[0]} trials read wrong values). Full results: <a href="evaluation.html">Every Option, Tested</a>.</p>
  </div>
</section>

<section>
  <h2>See it happen</h2>
  <div class="prose"><p>Six chunks written in two eras, then two edits. In memory, {DEMO_LINES} lines, and it needs only the zarr-python branch: <a href="https://github.com/maxj-labs/mismatched-granules/blob/main/examples/reproduce.py">examples/reproduce.py</a>. The eras differ only in byte order and nothing is compressed, so a chunk read with the wrong layout cannot fail; it is the worst case for a silent error.</p></div>
<pre>{e(DEMO.strip())}</pre>
</section>

<section>
  <h2>How often do NASA records receive data away from the end?</h2>
  <div class="prose">
    <p>If records only ever grew at the end, edits would be rare. To see, {n} collections were examined through NASA's public metadata catalogue: the seven surveyed earlier in this project and nineteen other well-known ones, chosen by me. For each granule the catalogue gives the time its data covers and the time its file was produced. A granule counts as <em>behind</em> when its file was produced at least a day after files holding newer data: from the point of view of a store that is kept up to date, an insertion, a backfill or a replacement.</p>
  </div>
  {prevalence_table()}
  <div class="prose">
    <p><b>In {count(GROUPS[0])} of {n} collections, at least 1% of the files produced in the last year of the sample hold data more than 30 days older than data already produced.</b> {count(GROUPS[1])} more show only short delays, a few days of latency between tiles or passes. {count(GROUPS[2])} show older data being made earlier in their record but not in the last year, typically a reprocessing campaign. {count(GROUPS[3])} show nothing behind, and for {blind_count} of those the sample could not have shown it, for the reason given in the table.</p>
    <p class="note">This is what public metadata suggests, not a record of what any store experienced. Production time is whatever the provider reports; it is not the time a file was published, and some providers reset it when files are migrated. Only current files are seen, so a file replaced in data order leaves no trace. &ldquo;Behind&rdquo; is measured against the newest data produced so far, so the first filling of a new version, done newest-first, counts as behind throughout. Files made within a day of each other are treated as one delivery. Collections with more than 60,000 granules are sampled from their newest data only. The last year of production is the last in the sample, whenever that was. A store built once, after a reprocessing campaign has finished, would see none of that campaign.</p>
  </div>
</section>

<section>
  <h2>What to choose</h2>
  <div class="prose">
    <p class="ask">After the store is built, will files be inserted, backfilled, replaced or dropped by anything other than a tool that rewrites the design's own table?</p>
    <ul>
      <li><b>If not,</b> finding the case by position is enough and asks the least. It needs no store change where the chunk shape is stable, and it read correctly as Kerchunk references in {kpos.count("equal")} of {len(kpos)} test cases. VirtualiZarr's append rewrites the table as it goes; that path is covered by its tests.</li>
      <li><b>If so,</b> keep the case in each reference. Edits then touch only the references that change, a new layout that is not registered is refused loudly, and a replaced file cannot be misread. The price is the Icechunk change and somewhat larger manifests ({size(km)} against {size(pm)} for twelve years of Daymet Puerto Rico).</li>
      <li><b>If the chunk shape changes</b> (Daymet North America in 2020), the Icechunk change is needed either way, because each Zarr chunk becomes a virtual shard. That removes most of position's advantage.</li>
      <li><b>If every file or chunk has numbers of its own</b> (packing with its own scale and offset), carry those numbers in the reference beside the key, as refkv PR&nbsp;#73 does.</li>
      <li><b>If the files are few and large and edits are frequent,</b> one array per file is the most forgiving, at the costs above.</li>
    </ul>
    <p>Speed did not separate the table designs in these tests. On twelve years of Daymet Puerto Rico, behind a local server adding 20 ms per request, the seven variants read a full series in {min(slow):.0f} to {max(slow):.0f} s with the same requests, and in about {min(fast):.0f} s once zarr-python was allowed more requests in flight.</p>
  </div>
</section>

<section>
  <h2>Go deeper</h2>
  <div class="cards">
    <div class="card">
      <div class="kind">Interactive</div>
      <h3><a href="storybook.html">Storybook</a></h3>
      <p>Pick an implementation and read what follows from choosing it, or pick the way your granules disagree and see which implementations serve it.</p>
    </div>
    <div class="card">
      <div class="kind">The evidence</div>
      <h3><a href="evaluation.html">Every Option, Tested</a></h3>
      <p>Sixteen options from the public discussions against ten cases, seven kinds of edit, two worst cases at 16,000 chunks, and Daymet. How each was built, and the limits of the tests.</p>
    </div>
    <div class="card">
      <div class="kind">Background</div>
      <h3><a href="brief.html">The original proposal</a></h3>
      <p>The first write-up: the switch codec, virtual shards, and results from seven NASA collections. Where it differs from the pages above, they are right.</p>
    </div>
  </div>
</section>

<section>
  <h2>Code</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>Where</th><th>What</th></tr></thead>
    <tbody>
      <tr><td><a href="https://github.com/maxj-labs/mismatched-granules">maxj-labs/mismatched-granules</a></td><td>Test scripts, result files, the demonstration above, and the source of these pages</td></tr>
      <tr><td><a href="https://github.com/maxj-labs/zarr-python/tree/feat/array-concat-codec">zarr-python branch</a></td><td>The <span class="mono">switch</span> codec, by position or by key, and the <span class="mono">array_concat</span> codec</td></tr>
      <tr><td><a href="https://github.com/maxj-labs/icechunk/tree/feat/virtual-ref-prefix">Icechunk branch</a></td><td>Virtual references made of a literal and several byte ranges</td></tr>
      <tr><td><a href="https://github.com/maxj-labs/VirtualiZarr/tree/feat/concat-mismatched-arrays">VirtualiZarr branch</a></td><td>A concat that plans one array from mismatched files, and writes and appends it</td></tr>
    </tbody>
  </table>
  </div>
  <p class="note">Not yet built in any design: inserting, backfilling or replacing through VirtualiZarr. Its writer appends; anything else means planning the array again. The edits above were made on the stored references directly. Discussions these pages respond to: <a href="https://github.com/zarr-developers/zarr-specs/issues/399">zarr-specs#399</a>, <a href="https://github.com/zarr-developers/zarr-specs/issues/288">zarr-specs#288</a>, <a href="https://github.com/zarr-developers/zarr-extensions/issues/74">zarr-extensions#74</a>, <a href="https://github.com/NASA-IMPACT/veda-odd/issues/498">NASA-IMPACT/veda-odd#498</a>, <a href="https://github.com/zarr-developers/VirtualiZarr/issues/1111">VirtualiZarr#1111</a>, <a href="https://github.com/d-v-b/refkv/pull/73">refkv PR #73</a>.</p>
</section>
</div>
"""


if __name__ == "__main__":
    out = ROOT / "site" / "index.html"
    out.write_text(render())
    print("wrote", out)
