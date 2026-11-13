"""The evaluation page's text. report.py supplies the tables and numbers."""

from __future__ import annotations

from pathlib import Path

BRIEF = Path(__file__).resolve().parent.parent / "brief" / "index.html"

EXTRA_CSS = """
th.c, td.c { text-align: center; padding-right: 8px; }
table.grid th.c { white-space: normal; font-size: 11.5px; line-height: 1.25; min-width: 62px; vertical-align: bottom; }
.dash { color: var(--ink-3); font-weight: 600; }
.sub { display: block; font-size: 11px; color: var(--ink-3); font-weight: 400; }
.sub0 { font-size: 13px; color: var(--ink-2); }
.hot { color: var(--warn); font-weight: 600; }
.dashcell { color: var(--ink-3); text-align: center; }
[title] { cursor: help; }
.key { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13.5px; color: var(--ink-2); }
ol { margin: 0; padding-left: 22px; display: flex; flex-direction: column; gap: 6px; }
"""

BY_REFERENCE = ["switch_key", "switch_case_id", "switch_key_parameters"]
BY_REFERENCE_HASHED = ["switch_key", "switch_key_parameters"]
EDITS = ("append a file", "insert a file", "prepend a file", "replace a file")
BY_POSITION = ["switch_position", "switch_index_prefix", "switch_sidecar", "switch_store_context"]
CASE_TABLE = ["switch_position", "switch_key", "switch_case_id", "switch_key_parameters", "switch_index_prefix", "switch_sidecar", "switch_store_context"]


def render(g: dict) -> str:
    style = BRIEF.read_text()
    head = style[: style.index("</style>")].replace("<title>Real Collections and Virtual Shards</title>", "<title>Every Option, Tested</title>")
    count, scale, size, CASES, plain = g["count"], g["scale"], g["size"], g["CASES"], g["plain"]
    total, per_chunk, seconds, requests, manifest, growth = g["compose_total"], g["per_chunk"], g["seconds"], g["requests"], g["manifest"], g["growth"]
    cases = "".join(f"<li><b>{g['CASE_LABEL'][c]}.</b> {g['e'](g['MATRIX']['cases'][c][0].upper() + g['MATRIX']['cases'][c][1:])}.</li>" for c in CASES)
    pr, na, pr120 = g["PERF"]["pr"], g["PERF"]["na"], g["PERF"]["pr120"]
    refkv_rows = {r["workload"]: r for r in g["REFKV"].get("rows", [])}
    long_read = "series: every day at one cell"
    whole_map = "map: 2 days, whole grid, across the 2020 change"

    def trials(arms: list[str], outcome: str) -> int:
        return sum(total(a)[outcome] for a in arms)

    def all_trials(arms: list[str]) -> int:
        return sum(sum(total(a).values()) for a in arms)

    def span(arms: list[str], fn) -> str:
        values = [fn(a) for a in arms]
        return f"{min(values):.2f} to {max(values):.2f}"

    table_pr = [a for a in CASE_TABLE if a in pr["arms"] and pr["arms"][a].get("reads")]
    series = [seconds("pr", a) for a in table_pr]
    latency = seconds("pr", "switch_position") / requests("pr", "switch_position") * 10 * 1000
    fast = [a for a in pr120.get("arms", {}) if pr120["arms"][a].get("reads")]
    fast_table = [seconds("pr120", a) for a in fast if a in CASE_TABLE]
    stored = {a: scale(a, "packed", "virtual_bytes") for a in CASE_TABLE if scale(a, "packed", "outcome") == "equal"}
    stored_alt = {a: scale(a, "alternate", "virtual_bytes") for a in CASE_TABLE if scale(a, "alternate", "outcome") == "equal"}
    doc_small = [a for a in CASE_TABLE if scale(a, "packed", "outcome") == "equal" and scale(a, "packed", "zarr_json_bytes") < 10_000]
    doc_large = [a for a in CASE_TABLE if scale(a, "packed", "outcome") == "equal" and scale(a, "packed", "zarr_json_bytes") >= 10_000]
    concat_ratio = [scale("array_concat", "alternate", "microseconds_per_chunk", n) / scale("switch_key", "alternate", "microseconds_per_chunk", n) for n in (1000, 4000, 16000)]
    kerchunk = {a: [v["outcome"] for k, v in g["KERCHUNK"]["cells"].items() if k.startswith(a + "|")] for a in ("switch_position", "switch_key")}
    known = "zarr-python with the codecs, released Icechunk"
    hashed_known = g["compose_n"](BY_REFERENCE_HASHED, EDITS, "already in the array") + g["compose_n"](BY_REFERENCE_HASHED, g["DROPPING"], None)
    assert hashed_known == g["compose_n"](BY_REFERENCE_HASHED, EDITS, "already in the array", "composes") + g["compose_n"](BY_REFERENCE_HASHED, g["DROPPING"], None, "composes")
    assert max(scale(a, "packed", "zarr_json_bytes") for a in doc_small) < 1000
    assert sorted(a for a in CASE_TABLE if scale(a, "alternate", "zarr_json_bytes") >= 1000) == ["switch_index_prefix", "switch_position"]
    assert scale("switch_index_prefix", "alternate", "zarr_json_bytes") // 1000 == scale("switch_position", "alternate", "zarr_json_bytes") // 1000

    return f"""{head}{EXTRA_CSS}</style>

<div class="wrap">

<header>
  <section>
    <div class="eyebrow">Evaluation &middot; revised 13 November 2026</div>
    <h1>Every Option, Tested</h1>
    <p class="lede">The public discussions on joining mismatched granules into one Zarr array propose about sixteen options. Each was built or emulated here and run against the same ten small test cases. Subsets were then run against edits to a stored array, two worst cases and real Daymet granules.</p>
    <p class="byline">Every mark and number, in the tables and in the text, is generated from result files in <a href="https://github.com/maxj-labs/mismatched-granules">maxj-labs/mismatched-granules</a> (<span class="mono">eval/</span>, <span class="mono">results/eval/</span>). Prepared with Claude. An earlier version of this page ranked the options; its tests were uneven, and this version replaces it.</p>
  </section>
</header>

<section>
  <div class="claims">
    <div class="claim">
      <div class="who">What separates the options</div>
      <div class="what">Whether the array survives an edit to the store</div>
      <div class="why">Each edit changed only the references of the files involved. With a hashed key in each reference, all {hashed_known} trials read correctly when the file's layout was already in the table. By position, every edit that changed the number of chunks was refused until the table was rewritten, and {g["compose_n"](["switch_position"], g["REPLACING"], "already in the array", "wrong values, no error")} replaced-file trials read wrong values with no error.</div>
    </div>
    <div class="claim">
      <div class="who">What does not</div>
      <div class="what">Speed, among the seven layout-table options</div>
      <div class="why">On twelve years of Daymet Puerto Rico they made the same requests and took {min(series):.1f} to {max(series):.1f} s for a full time series. Raising zarr-python's concurrency brought all of them to about {min(fast_table):.0f} s.</div>
    </div>
    <div class="claim">
      <div class="who">What each one needs</div>
      <div class="what">A store change, or a codec that is told the chunk's position</div>
      <div class="why">A key in the reference needs a new kind of Icechunk reference (a literal followed by byte ranges) and cannot be stored as Kerchunk references. Position needs neither, except where the chunk shape changes.</div>
    </div>
  </div>
</section>

<section>
  <h2 id="options">The options</h2>
  <div class="prose"><p>Taken from zarr-specs#399 and #288, zarr-extensions#74 and PR&nbsp;#75, NASA-IMPACT/veda-odd#498, VirtualiZarr#1111 and refkv PR&nbsp;#73. Seven of them are the same codec. It holds a table of <em>layouts</em> (the discussions call them cases): the ways a chunk in this array can be stored. The seven differ only in how a reader learns which layout a chunk has. They share one planner and one encoding of the table, so differences between their rows come from that choice alone.</p></div>
  {g["options_table"]()}
  <div class="prose">
    <p><b><a href="brief.html#virtual-shards">Virtual shards</a></b> (d-v-b, zarr-specs#399) are not a row of their own. Every layout-table option uses them when the chunk shape changes, which is the &ldquo;chunk shape change&rdquo; test case below.</p>
    <p><b>Proposed but not built:</b></p>
    <ul>
      <li><b>A storage transformer</b> (zarr-specs#399 design 2.3; a generalised transformer chain, mkitti). zarr-python refuses any array that declares one, so there is nothing to run it in.</li>
      <li><b>An &ldquo;array to files&rdquo; codec</b> (jbms). It ends the rule of one key per chunk, which Icechunk, VirtualiZarr and zarr-python all assume, and was too large a change to prototype here.</li>
      <li><b>A union data type</b> (d-v-b). It would let one array hold two data types, but does not address codecs, chunk shapes or padding.</li>
      <li><b>Index transforms as in TensorStore</b> (jbms). TensorStore cannot read virtual references, so only the cost of its stack driver on native arrays was measured (under Worst cases).</li>
    </ul>
  </div>
</section>

<section>
  <h2 id="test-cases">Can it describe the files?</h2>
  <div class="prose"><p>Each of the ten test cases is a small set of netCDF4 or HDF5 files that disagree in one way real granules do. An option passes a test case when every read (the whole array, a window across each file boundary, single cells, one row along the joined axis) equals a direct read of the files.</p></div>
  {g["matrix_table"]()}
  <div class="key"><span><span class="yes">&#10003;</span> every read equals the files</span><span><span class="dash">&ndash;</span> the option cannot describe these files</span><span><span class="no">&#10007;</span> reads differ from the files</span></div>
  <div class="prose">
    <p><b>How to read this table.</b> The seven layout-table variants share a planner and a codec that also crops, casts and carries a fill value, so they pass together and are shown as one row. The narrower proposals were each written to solve one problem, so a low count shows how much of the whole problem one of them covers, not that it fails at its own job. The two copying rows are written from the same values they are compared with, so their marks show only that copying is possible, and how much is copied.</p>
  </div>
  <div class="two">
    <div>
      <h3>The test cases</h3>
      <ol>{cases}</ol>
    </div>
    <div>
      <h3>Why an option cannot describe a test case</h3>
      {g["reasons_list"]()}
    </div>
  </div>
</section>

<section>
  <h2 id="readers">Readers without the new codecs, and stores other than Icechunk</h2>
  <div class="prose"><p>Every store that passed above was opened by two more readers: released zarr-python 3.4.2 with released Icechunk, and zarr-python with the new codecs but released Icechunk. &ldquo;Refuses&rdquo; means the reader raises an error and returns no data.</p></div>
  {g["readers_table"]()}
  <div class="prose">
    <p><b>Released software.</b> Every codec-based option is refused with <code>UnknownCodecError</code>. The declared domain is the exception: an ordinary reader sees the padding as data and returns an array of the wrong shape with no error. The grid option, where it pads, is refused only by an unrelated reshape error.</p>
    <p><b>New codecs, released Icechunk.</b> Stores whose references carry a prefix or several ranges are refused with <code>FormatError</code>. Position reads in {g["reader_outcomes"]("switch_position", known).count("reads correctly")} of {len(g["reader_outcomes"]("switch_position", known))}, because it needs the Icechunk change only where the chunk shape changes. The option where the store prepends an index is refused only by luck: released Icechunk returns the bytes with no index in front, and the codec happens to fail on what it takes for one.</p>
    <p><b>Kerchunk references.</b> The position and key options were also written as Kerchunk references and read through fsspec. Position read correctly in {kerchunk["switch_position"].count("equal")} of {len(kerchunk["switch_position"])} test cases; the exception is the chunk-shape change. Key read in {kerchunk["switch_key"].count("equal")} of {len(kerchunk["switch_key"])}, because a Kerchunk reference is either inline bytes or one byte range, never a literal followed by a range. The other options were not tried.</p>
  </div>
</section>

<section>
  <h2 id="edits">Edits to a stored array</h2>
  <div class="prose">
    <p>zarr-specs#399 asks that a solution keep working when a tool edits the manifest (that it &ldquo;compose with manifest-level operations&rdquo;). Seven kinds of edit were tested:</p>
    <ul>
      <li>three drop chunks: the first chunk, the first file, or the last chunk;</li>
      <li>three add a file: appended, inserted in the middle, or put at the front as a backfill;</li>
      <li>one replaces a file with another of the same length.</li>
    </ul>
    <p>Each edit changes only the chunk references of the files involved (including the key in front, for the keyed options) and updates nothing else: not a codec's configuration, a side array, a list of pieces or an attribute. A trial is one edit on one test case. Six of the ten test cases were used, and each option was run on those it can describe. A trial reads the whole array, each file's part on its own, and single cells. It is <em>correct</em> if every read equals the files, <em>wrong</em> if any read returns wrong values without an error, and <em>refused</em> if reads raise an error.</p>
  </div>
  {g["compose_table"]()}
  <div class="prose">
    <p><b>A key in the reference moves with the reference.</b> With hashed 8-byte keys, with or without per-chunk numbers, every drop and every added or replaced file in a layout the array has read correctly. A file in a new layout is refused until that layout is added to the table, unless it differs only in numbers that the reference itself carries.</p>
    <p><b>By position, edits are refused, not misread, with one exception.</b> The codec checks that its table covers exactly the array's chunks, so any edit that adds or removes chunks fails when the array is opened, until the table is rewritten. That check was added during this work; without it, dropping leading chunks read wrong values in some test cases. The exception is replacing a file with one of the same length in a different layout. The chunk count is unchanged, so there is nothing to check, and {g["compose_n"](["switch_position"], g["REPLACING"], "already in the array", "wrong values, no error")} such trials read wrong values with no error. The option where the store prepends the chunk index has the same check and the same exception. Tables kept in side arrays or behind the emulated store call have no such check, and misread more often.</p>
    <p><b>A 1-byte numbered key is not safe to edit.</b> Adding a file at the front or in the middle, or replacing one, can change which number a layout has: {g["compose_n"](["switch_case_id"], EDITS, None, "wrong values, no error")} trials read wrong values with no error. A key that is a hash of the layout does not have the problem.</p>
    <p><b>Keeping each file as its own array survived every edit tried.</b> The reference-per-chunk codec read correctly in every trial, because nothing is shared between files. It cannot drop less than a whole file, and its costs are in the next section. The concatenation codec lists its pieces and checks that list against the array's shape, so it refuses every edit except replacement.</p>
    <p><b>Caveats.</b> Trial counts are small, in places two or three per cell. The store-context row is an emulation whose table is indexed by position; a store that kept the context with each reference would carry it along, and that was not tested. The grid rows keep encoded sizes in an attribute where the proposal puts them in the chunk grid. Renaming, concatenating two joined arrays, and nesting under another codec were not run. VirtualiZarr can only append so far, so these edits were made directly on the stored references.</p>
  </div>
</section>

<section>
  <h2 id="worst-cases">Worst cases</h2>
  <div class="prose"><p>16,000 one-chunk pieces, cut from real netCDF files so that planners and writers see what 16,000 single-chunk granules would give them. Everything is on local disk. Times are the median of three runs on a newly opened array. Repeats of the same read differed by up to {g["spread"]() * 100:.0f}% of the median, so smaller differences mean nothing. A figure is highlighted when it is at least three times the smallest in its column among the options that keep the array virtual.</p></div>
  {g["scaling_table"]()}
  <div class="prose">
    <p><b>Two sizes for metadata.</b> &ldquo;zarr.json&rdquo; is the document a reader parses to open the array. &ldquo;All metadata, stored&rdquo; is everything the repository keeps apart from chunk data (snapshots and manifests, compressed). The two can differ by an order of magnitude, because a long table of similar layouts compresses well.</p>
    <p><b>Where every chunk has numbers of its own,</b> the options that carry those numbers with each chunk keep zarr.json under 1 KB. The options that can only name a layout need one layout per chunk: zarr.json grows to {size(min(scale(a, "packed", "zarr_json_bytes") for a in doc_large))} or more, and opening the array takes {min(scale(a, "packed", "open_seconds") for a in doc_large):.2f} to {max(scale(a, "packed", "open_seconds") for a in doc_large):.2f} s. Once compressed in the repository the gap narrows: stored metadata runs from {size(min(stored.values()))} to {size(max(stored.values()))} across the layout-table options. This worst case varies exactly the two numbers that the per-chunk options carry, so it shows them at their best.</p>
    <p><b>Where two layouts alternate,</b> each layout is written once. zarr.json is under 1 KB for five of the options. For the two whose zarr.json lists which run of chunks has which layout (position, and the store-prepended index), it is {size(scale("switch_position", "alternate", "zarr_json_bytes"))}: a pair of integers per run. Stored metadata is {size(min(stored_alt.values()))} to {size(max(stored_alt.values()))} for all seven. A prefix in every reference does make manifests larger: {size(scale("switch_key_parameters", "packed", "manifest_bytes"))} with a key and two numbers, against {size(scale("switch_position", "packed", "manifest_bytes"))} by position, in the packed case.</p>
    <p><b>Read time per chunk</b> is the wall time to read everything divided by the number of chunks, with ten chunks in flight; it is not the cost of one chunk. The layout-table options took {span([a for a in CASE_TABLE if scale(a, "alternate", "outcome") == "equal"], lambda a: per_chunk(a, "alternate"))} ms in the alternating case and {span([a for a in CASE_TABLE if scale(a, "packed", "outcome") == "equal"], lambda a: per_chunk(a, "packed"))} ms in the packed one. The slow end of the packed range is the options with one layout per chunk ({per_chunk("switch_position", "packed"):.1f} ms by position, {per_chunk("switch_key", "packed"):.1f} ms by key alone). With 16,000 distinct layouts the codec builds a new codec chain for nearly every chunk; the options that carry numbers per chunk took {min(per_chunk(a, "packed") for a in doc_small):.1f} to {max(per_chunk(a, "packed") for a in doc_small):.1f} ms.</p>
    <p><b>Concatenation with one piece per chunk</b> writes one array per piece, which is why its stored size and write time grow. Its read time per chunk also grows with the number of pieces ({", ".join(f"{r:.0f}&times;" for r in concat_ratio)} the key option's at 1,000, 4,000 and 16,000). That is probably my implementation and not the design: {g["tensorstore_sentence"]()}</p>
  </div>
</section>

<section>
  <h2 id="real-granules">Real granules</h2>
  <div class="prose"><p>Ten of the options on Daymet, read through a local HTTP server that counts requests and waits 20 ms on each. Every read starts from a new session.</p></div>
  <h3>Daymet Puerto Rico, {pr["years"][0]}&ndash;{pr["years"][1]}: {pr["shape"][0]:,} daily chunks, compression changes in 2020</h3>
  {g["perf_table"]("pr")}
  <h3>Daymet North America, {na["years"][0]}&ndash;{na["years"][1]}: the chunk shape changes in 2020, so every layout-table option stores virtual shards</h3>
  {g["perf_table"]("na", True)}
  <h3>Puerto Rico again, with zarr-python allowed 120 chunks in flight instead of 10</h3>
  {g["perf_table"]("pr120")}
  <div class="prose">
    <p><b>What this test can and cannot show.</b> It shows that the options make the same requests, and that none adds a round trip. It cannot show small differences in decoding cost: at the default setting a full time series took about {latency:.0f} ms per request with ten in flight, well above the server's 20 ms, so the test harness dominates and a millisecond per chunk is invisible.</p>
    <p><b>The seven layout-table options stay close together.</b> For the full Puerto Rico series they are within {(max(series) / min(series) - 1) * 100:.0f}% of each other at the default setting and {(max(fast_table) / min(fast_table) - 1) * 100:.0f}% with 120 in flight, which is inside the run-to-run noise. Concatenation matched them on Puerto Rico at the default setting, took {seconds("pr120", "array_concat"):.1f} s against {seconds("pr120", "switch_position"):.1f} s with 120 in flight, and made more requests on North America ({requests("na", "array_concat"):,} against {requests("na", "switch_position"):,} for the series).</p>
    <p><b>The reference-per-chunk codec looked {seconds("pr", "switch_position") / seconds("pr", "zarr_reference"):.0f}&times; faster at the default setting, but only because it ran more requests at once.</b> It reads each year through a nested array read, so twelve years ran together, each with ten chunks in flight. With 120 allowed for everyone it took {seconds("pr120", "zarr_reference"):.1f} s, the same as the rest. On whole North America maps it was slower ({seconds("na", "zarr_reference", whole_map):.1f} s against {seconds("na", "switch_position", whole_map):.1f} s) with the same requests.</p>
    <p><b>zarr-python's concurrency setting changed these times by about {seconds("pr", "switch_position") / seconds("pr120", "switch_position"):.0f}&times;,</b> far more than any choice between the options.</p>
    <p><b>refkv as released</b> (0.2.0, no Icechunk), on the same Puerto Rico files and server at the default setting, took {refkv_rows[long_read]["seconds_median"]:.1f} s for the whole series with {refkv_rows[long_read]["source_requests"]:,} requests, and every read equalled the files.</p>
  </div>
</section>

<section>
  <h2 id="reading">Reading the results</h2>
  <div class="prose">
    <p>The tests do not produce a single winner. They show two workable designs for the layout table, and a trade-off between them.</p>
    <ul>
      <li><b>Put a key in each reference</b> (a hashed key, with or without per-chunk numbers; with them it is refkv PR&nbsp;#73's design). Edits touch only the references that change, and a file in a layout missing from the table is refused. It needs Icechunk references that are a literal followed by byte ranges, which virtual shards also need. It cannot be stored as Kerchunk references, and its manifests are larger.</li>
      <li><b>Find the layout from the chunk's position.</b> It needs no store change unless the chunk shape changes, reads through released Icechunk and as Kerchunk references, and stores the least. Its table must be rewritten on every edit. A missed rewrite is refused, except for a replaced file of the same length, which can be misread silently.</li>
      <li><b>If chunks carry numbers of their own</b> (per-file or per-chunk packing), carrying those numbers with each chunk keeps zarr.json small. Three options did that: in the reference, in side arrays, and through a store call. Only the first also read correctly after the edits above.</li>
      <li><b>Keeping files as separate arrays</b> (concatenation, or a reference per chunk) describes everything and reads with released Icechunk once the codec is installed. Joined by reference it survived every edit; joined by a list it refused them. Its costs grow with the number of pieces, so it suits a few eras better than many small granules.</li>
      <li><b>The narrow proposals</b> each solve part of the problem and can be combined; two of them together covered {count("conditional_with_grid_shapes")} of the {len(CASES)} test cases.</li>
    </ul>
    <p>My reading, which the tests inform but do not settle: the deciding question is whether anything edits the store after it is built without also updating the layout table. If files are inserted, backfilled, replaced or dropped that way, the layout belongs in the reference. If the store is built once, or only appended to by a tool that rewrites the table, position is simpler and asks less of the store. The <a href="https://maxj-labs.github.io/mismatched-granules/">overview</a> sets this beside a survey of how often NASA collections change away from the end.</p>
  </div>
</section>

<section>
  <h2 id="limits">Limits of this evaluation</h2>
  <div class="two">
    <div>
      <ul>
        <li>Only zarr-python reads any of these arrays.</li>
        <li>Timings are from one machine. Nothing ran against S3.</li>
        <li>The ten test cases are small files written for this test. They show whether an option can describe a situation, not how common it is.</li>
        <li>The fraction copied by the copying option reflects how the test cases were built: 20% for the padded case here, about 1.4% for a year of Daymet North America.</li>
        <li>Daymet was run for ten options, the Kerchunk test for two, and the worst cases for eleven.</li>
      </ul>
    </div>
    <div>
      <ul>
        <li>Two options are emulations, as marked in the first table: the store context call, and encoded chunk shapes in the grid. Real implementations could behave differently.</li>
        <li>The <span class="mono">conditional</span> codec and the key-and-numbers codec are my implementations of others' designs. refkv itself was run only on Daymet Puerto Rico.</li>
        <li>How the store knows to prepend an index is a stand-in (it reads a codec option); a real design needs a flag of its own.</li>
        <li>For a file without the variable, the expected values are a convention: the fill value the other files declare.</li>
      </ul>
    </div>
  </div>
</section>

</div>
"""

