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
    head = style[: style.index("</style>")].replace("<title>One Array from Mismatched Granules</title>", "<title>Every Option, Tested</title>")
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

    return f"""{head}{EXTRA_CSS}</style>

<div class="wrap">

<header>
  <section>
    <div class="eyebrow">Evaluation &middot; revised 13 November 2026</div>
    <h1>Every Option, Tested</h1>
    <p class="lede">The discussions on joining mismatched granules into one Zarr array propose about sixteen options. Each was built or emulated and run against the same ten small cases. Subsets were then run against seven kinds of edit to a stored array, two worst cases and real Daymet granules. This page reports what each one did.</p>
    <p class="byline">Every mark and number in the tables, and every figure quoted in the text, is generated from result files in <a href="https://github.com/maxj-labs/mismatched-granules">maxj-labs/mismatched-granules</a> (<span class="mono">eval/</span>, <span class="mono">results/eval/</span>). Prepared with Claude. This version replaces one whose tests favoured some options over others; what changed is listed at the end.</p>
  </section>
</header>

<section>
  <div class="claims">
    <div class="claim">
      <div class="who">What separates the options</div>
      <div class="what">What an edit to the store can break</div>
      <div class="why">A writer added, replaced or dropped files and rewrote nothing else. With the case in each reference, {g["compose_n"](BY_REFERENCE_HASHED, EDITS, "already in the array") + g["compose_n"](BY_REFERENCE_HASHED, g["DROPPING"], None)} of {g["compose_n"](BY_REFERENCE_HASHED, EDITS, "already in the array") + g["compose_n"](BY_REFERENCE_HASHED, g["DROPPING"], None)} trials read correctly where the file's layout was already known. By position, every edit that changed the number of chunks was refused until the table was rewritten, and {g["compose_n"](["switch_position"], g["REPLACING"], "already in the array", "wrong values, no error")} replaced-file trials read wrong values with no error.</div>
    </div>
    <div class="claim">
      <div class="who">What does not</div>
      <div class="what">Speed, among the seven case-table options</div>
      <div class="why">On twelve years of Daymet Puerto Rico they made the same requests and took {min(series):.1f} to {max(series):.1f} s for a full series. Raising zarr-python's concurrency moved that to about {min(fast_table):.0f} s for all of them.</div>
    </div>
    <div class="claim">
      <div class="who">What each one needs</div>
      <div class="what">A store change, or a codec that must be told where it is</div>
      <div class="why">Keeping the case in the reference needs Icechunk references that are a literal followed by byte ranges, and cannot be stored as Kerchunk references. Finding it by position needs neither, except where the chunk shape changes.</div>
    </div>
  </div>
</section>

<section>
  <h2>The options</h2>
  <div class="prose"><p>Taken from zarr-specs#399 and #288, zarr-extensions#74 and PR&nbsp;#75, NASA-IMPACT/veda-odd#498, VirtualiZarr#1111 and refkv PR&nbsp;#73. Seven of them are the same codec, a table of cases, and differ only in where a chunk's case is kept. They share one planner and one way of writing the table down, so their rows differ only by that choice.</p></div>
  {g["options_table"]()}
  <div class="prose">
    <p><b>Virtual shards</b> (d-v-b, zarr-specs#399) are not a row of their own: every case-table option uses them when the chunk shape changes, which is the &ldquo;chunk shape change&rdquo; case below.</p>
    <p><b>Proposed but not built.</b></p>
    <ul>
      <li><b>A storage transformer</b> (zarr-specs#399 design 2.3; a generalised transformer chain, mkitti). zarr-python refuses any array that declares one, so there is nothing to run it in.</li>
      <li><b>An &ldquo;array to files&rdquo; codec</b> (jbms). It ends the rule of one key per chunk, which Icechunk, VirtualiZarr and zarr-python all assume. Too large a change to prototype here.</li>
      <li><b>A union data type</b> (d-v-b). It would let one array hold two data types; it does not address codecs, chunk shapes or padding.</li>
      <li><b>Index transforms as in TensorStore</b> (jbms). TensorStore cannot read virtual references, so its stack driver was only measured for cost on native arrays, below.</li>
    </ul>
  </div>
</section>

<section>
  <h2>Can it describe the case?</h2>
  <div class="prose"><p>Ten small sets of netCDF4 or HDF5 files, each a way real granules disagree. An option passes a case when every read (the whole array, a window across each seam, single cells, one row along the joined axis) equals what the files' own reader returns.</p></div>
  {g["matrix_table"]()}
  <div class="key"><span><span class="yes">&#10003;</span> every read equals the files</span><span><span class="dash">&ndash;</span> the option cannot describe the case (hover for why)</span><span><span class="no">&#10007;</span> reads differ from the files</span></div>
  <div class="prose">
    <p><b>How to read this table.</b> The seven case-table rows pass together because they share a planner and a codec that also crops, casts and carries a fill value; they are one result about that bundle, shown seven times. The narrower proposals were written to solve one problem each, so a low count says how much of the whole problem one of them covers, not that it fails at its own job. Two of them used together cover {count("conditional_with_grid_shapes")} of {len(CASES)}. The two copying rows are written from the same values they are compared with, so their marks show only that copying is possible, and how much is copied.</p>
  </div>
  <div class="two">
    <div>
      <h3>The cases</h3>
      <ol>{cases}</ol>
    </div>
    <div>
      <h3>Why an option cannot describe a case</h3>
      {g["reasons_list"]()}
    </div>
  </div>
</section>

<section>
  <h2>Readers that lack the feature, and stores that are not Icechunk</h2>
  <div class="prose"><p>Every store from the table above was opened by two readers: released zarr-python 3.4.2 with released Icechunk, and zarr-python with the new codecs installed but released Icechunk. The arrays planned by position and by key were also written as Kerchunk references and read through fsspec.</p></div>
  {g["readers_table"]()}
  <div class="prose">
    <p><b>Released software.</b> Every codec-based option is refused with <code>UnknownCodecError</code>. The declared domain is the exception: an ordinary reader sees the padding as data and returns an array of the wrong shape with no error. The grid option, where it pads, is refused only by an unrelated reshape error.</p>
    <p><b>Codecs installed, released Icechunk.</b> Stores whose references carry a prefix or several ranges are refused with <code>FormatError</code>. Position selection reads in {g["reader_outcomes"]("switch_position", known).count("reads correctly")} of {len(g["reader_outcomes"]("switch_position", known))}: it needs the Icechunk change only where the chunk shape changes. The store that prepends an index is refused here by luck, not design: released Icechunk hands the codec bytes with no index in front, and the codec fails because what it takes for an index points outside its table.</p>
    <p><b>Kerchunk references.</b> Position selection read correctly in {kerchunk["switch_position"].count("equal")} of {len(kerchunk["switch_position"])} cases, the exception being the chunk-shape change. Key selection read in {kerchunk["switch_key"].count("equal")} of {len(kerchunk["switch_key"])}, because a Kerchunk reference is either inline bytes or one byte range, never a literal followed by a range. The other options were not written as Kerchunk references.</p>
  </div>
</section>

<section>
  <h2>Edits to a stored array</h2>
  <div class="prose"><p>zarr-specs#399 asks that a solution compose with manifest-level operations. Seven kinds of edit were run. Three drop chunks (the first chunk, the first file, the last chunk). Three add a file (appended, inserted in the middle, or put at the front as a backfill). One replaces a file with another of the same length. Each was made by a writer that adds, removes or replaces chunk references, in whatever form the option gives a reference (a key in front, for the keyed options), and rewrites nothing the array already has: not a codec's configuration, a side array, a list of pieces or an attribute. Six of the cases were used; an option was run on those it can describe. A trial reads the whole array, each file's part on its own, and single cells; it is correct only if every read equals the files, and wrong if any read returns wrong values without an error.</p></div>
  {g["compose_table"]()}
  <div class="prose">
    <p><b>A case kept in the reference moves with the reference.</b> With hashed 8-byte keys, with or without per-chunk numbers, every drop and every added or replaced file in a known layout read correctly. A file in a new layout is refused until its case is added to the table, except where the new layout differs only in numbers that the reference itself carries.</p>
    <p><b>By position, edits are refused, not misread, with one exception.</b> The codec now checks that its table covers the array's chunks, so any edit that adds or removes chunks fails when the array is opened until the table is rewritten. Before that check was added, dropping leading chunks read wrong values in some cases. The exception is replacing a file with one of the same length in a different layout: the count is unchanged, nothing can be checked, and {g["compose_n"](["switch_position"], g["REPLACING"], "already in the array", "wrong values, no error")} such trials read wrong values with no error. The index-prefix option has the same check and the same exception. Tables kept in side arrays or by the emulated store call have no such check, and misread more often.</p>
    <p><b>A 1-byte numbered key is not safe to edit.</b> Adding a file at the front or in the middle, or replacing one, can change which number a layout has; {g["compose_n"](["switch_case_id"], EDITS, None, "wrong values, no error")} trials read wrong values with no error. Hashing the layout into the key removes the problem.</p>
    <p><b>Keeping each file as its own array composes with everything tried.</b> The reference-per-chunk codec read correctly in every trial, because nothing is shared between files. It cannot drop less than a file, and its costs are in the next section. The concatenation codec, which lists its pieces, refuses every edit but replacement: it checks its list against the array's shape.</p>
    <p><b>Caveats.</b> A file that changes how the others must be described (a different fill value, data type or chunk shape) forces the table designs to plan the array again; those trials are in the last column. The store-context row is an emulation whose table is indexed by position; a store that kept the context with each reference would carry it along, and that was not tested. The grid rows keep encoded sizes in an attribute where the proposal puts them in the chunk grid. Trial counts are small, in places two or three per cell. Renaming, concatenating two joined arrays and nesting under another codec were not run. None of these edits is yet available through VirtualiZarr except appending; they were made on the stored references directly.</p>
  </div>
</section>

<section>
  <h2>Worst cases</h2>
  <div class="prose"><p>16,000 one-chunk pieces, cut from real netCDF files so that planners and writers see what 16,000 single-chunk granules would give them. Local disk. Every time is the median of three runs on a newly opened array; repeats of the same read differed by up to {g["spread"]() * 100:.0f}% of the median, so differences smaller than that mean nothing. A figure is marked when it is three or more times the smallest in its column among the options that keep the array virtual.</p></div>
  {g["scaling_table"]()}
  <div class="prose">
    <p><b>Two sizes for metadata.</b> &ldquo;zarr.json as read&rdquo; is the document a reader parses when it opens the array. &ldquo;Stored&rdquo; is everything the repository keeps apart from chunk data (snapshots and manifests, compressed). They can differ by an order of magnitude, because a long table of similar cases compresses well.</p>
    <p><b>Where every chunk has numbers of its own,</b> an option that carries those numbers per chunk keeps the document small ({", ".join(plain(a).replace("Case table · ", "") + " " + size(scale(a, "packed", "zarr_json_bytes")) for a in doc_small)}). An option that can only name a case needs one case per chunk ({", ".join(plain(a).replace("Case table · ", "") + " " + size(scale(a, "packed", "zarr_json_bytes")) for a in doc_large)}), and opening the array then takes {min(scale(a, "packed", "open_seconds") for a in doc_large):.2f} to {max(scale(a, "packed", "open_seconds") for a in doc_large):.2f} s. Stored size tells a different story: {", ".join(plain(a).replace("Case table · ", "") + " " + size(v) for a, v in sorted(stored.items(), key=lambda kv: kv[1]))}. This worst case varies exactly the two numbers the templated options parameterise, so it shows the mechanism at its best.</p>
    <p><b>Where two layouts alternate,</b> each case is written once, so the document is under 1 KB for the options that keep the case with the chunk and {size(scale("switch_position", "alternate", "zarr_json_bytes"))} by position, which adds a pair of integers per run. Stored: {", ".join(plain(a).replace("Case table · ", "") + " " + size(v) for a, v in sorted(stored_alt.items(), key=lambda kv: kv[1]))}. A prefix in every reference makes manifests larger: {size(scale("switch_key_parameters", "packed", "manifest_bytes"))} with a key and two numbers against {size(scale("switch_position", "packed", "manifest_bytes"))} by position.</p>
    <p><b>Read time per chunk</b> is the wall time to read everything divided by the number of chunks, with ten chunks in flight; it is not the cost of one chunk. The case-table options took {span([a for a in CASE_TABLE if scale(a, "alternate", "outcome") == "equal"], lambda a: per_chunk(a, "alternate"))} ms in the alternating case and {span([a for a in CASE_TABLE if scale(a, "packed", "outcome") == "equal"], lambda a: per_chunk(a, "packed"))} ms in the packed one. The slow end of the packed range is the options with a case per chunk ({per_chunk("switch_position", "packed"):.1f} ms by position, {per_chunk("switch_key", "packed"):.1f} ms by key alone): with 16,000 distinct codec chains the codec rebuilds a chain for nearly every chunk, where the options that carry numbers per chunk took {min(per_chunk(a, "packed") for a in doc_small):.1f} to {max(per_chunk(a, "packed") for a in doc_small):.1f} ms.</p>
    <p><b>Concatenation with one piece per chunk</b> writes one array per piece, which is what makes its stored size and write time grow. Its read time per chunk also grows with the number of pieces ({", ".join(f"{r:.0f}&times;" for r in concat_ratio)} the key option's at 1,000, 4,000 and 16,000), which points at my implementation of the codec and not at the design: {g["tensorstore_sentence"]()}</p>
  </div>
</section>

<section>
  <h2>Real granules</h2>
  <div class="prose"><p>Ten of the options on Daymet, with the source behind a local HTTP server that waits 20 ms per request and counts them. Each read starts from a new session.</p></div>
  <h3>Daymet Puerto Rico, {pr["years"][0]}&ndash;{pr["years"][1]}: {pr["shape"][0]:,} daily chunks, compression changes in 2020</h3>
  {g["perf_table"]("pr")}
  <h3>Daymet North America, {na["years"][0]}&ndash;{na["years"][1]}: the chunk shape changes in 2020, so every case-table option stores virtual shards</h3>
  {g["perf_table"]("na", True)}
  <h3>Puerto Rico again, with zarr-python allowed 120 chunks in flight instead of 10</h3>
  {g["perf_table"]("pr120")}
  <div class="prose">
    <p><b>What this test can and cannot show.</b> At the default setting a full series took about {latency:.0f} ms per request with ten in flight, well above the server's 20 ms, so the harness itself dominates and differences of a millisecond per chunk are invisible here. It shows that the options make the same requests, and that none adds a round trip.</p>
    <p><b>The seven case-table options stay close together:</b> for the full Puerto Rico series, within {(max(series) / min(series) - 1) * 100:.0f}% of each other at the default setting and {(max(fast_table) / min(fast_table) - 1) * 100:.0f}% with 120 in flight, which is inside the run-to-run noise of this harness. Concatenation matched them on Puerto Rico at the default setting, took {seconds("pr120", "array_concat"):.1f} s against {seconds("pr120", "switch_position"):.1f} s with 120 in flight, and made more requests on North America ({requests("na", "array_concat"):,} against {requests("na", "switch_position"):,} for the series).</p>
    <p><b>The reference-per-chunk codec looked {seconds("pr", "switch_position") / seconds("pr", "zarr_reference"):.0f}&times; faster at the default setting.</b> It reads each year through a nested array read, so twelve years ran at once, each with ten chunks in flight. With 120 allowed for everyone it took {seconds("pr120", "zarr_reference"):.1f} s, the same as the rest. On whole North America maps it was slower ({seconds("na", "zarr_reference", whole_map):.1f} s against {seconds("na", "switch_position", whole_map):.1f} s) with the same requests.</p>
    <p><b>zarr-python's concurrency setting moved these reads by about {seconds("pr", "switch_position") / seconds("pr120", "switch_position"):.0f}&times;,</b> far more than any choice between the options.</p>
    <p><b>refkv as released</b> (0.2.0, no Icechunk), on the same Puerto Rico files and server at the default setting: {refkv_rows[long_read]["seconds_median"]:.1f} s for the whole series with {refkv_rows[long_read]["source_requests"]:,} requests, every read equal to the files.</p>
  </div>
</section>

<section>
  <h2>Reading the results</h2>
  <div class="prose">
    <p>The tests do not produce a single winner. They show two workable designs for the case table, and a trade between them.</p>
    <ul>
      <li><b>Keep the case in each reference</b> (a hashed key, with or without per-chunk numbers; the latter is refkv PR&nbsp;#73's design). Edits touch only the references that change, and a layout that is not in the table is refused. It needs Icechunk references that are a literal followed by byte ranges, which virtual shards also need; it cannot be stored as Kerchunk references; and its manifests are larger.</li>
      <li><b>Find the case from the chunk's position.</b> It needs no store change unless the chunk shape changes, reads through released Icechunk and as Kerchunk references, and stores least. Its table must be rewritten on every edit; with the coverage check, a missed rewrite is refused, except for a replaced file of the same length, which can be misread silently.</li>
      <li><b>If chunks carry numbers of their own</b> (per-file or per-chunk packing), carrying those numbers per chunk keeps what a reader must parse small. Three options did that here: in the reference, in side arrays, and through a store call. Only the first also read correctly after the edits above.</li>
      <li><b>Keeping files as separate arrays</b> (concatenation, or a reference per chunk) describes everything and reads with released Icechunk once the codec is installed. Joined by reference it composed with every edit; joined by a list it refused them. Its costs grow with the number of pieces, so it suits a few eras better than many small granules.</li>
      <li><b>The narrow proposals</b> each solve part of the problem and can be combined; two of them together covered {count("conditional_with_grid_shapes")} of the {len(CASES)} cases.</li>
    </ul>
    <p>My reading, which the tests inform but do not settle: the deciding question is whether a store is edited after it is built by anything that might not update the design's own table. If files are inserted, backfilled, replaced or dropped that way, the case belongs in the reference. If the store is built once or only appended to by a tool that knows the table, position is simpler and asks less of the store. The <a href="https://maxj-labs.github.io/mismatched-granules/">overview</a> sets this beside a survey of how often NASA collections receive data behind what they already hold.</p>
  </div>
</section>

<section>
  <h2>Limits of this evaluation</h2>
  <div class="two">
    <div>
      <ul>
        <li>Two options are emulations, as marked in the first table: the store context call, and encoded chunk shapes in the grid. Real implementations could behave differently.</li>
        <li>The <span class="mono">conditional</span> codec and the key-and-numbers codec are my implementations of others' designs. refkv itself was run only on Daymet Puerto Rico.</li>
        <li>Only zarr-python reads any of these arrays.</li>
        <li>The ten cases are small files written for this test. They show whether an option can describe a situation, not how common it is. The fraction copied by the copying option reflects how the cases were built: 20% for the padded case here, about 1.4% for a year of Daymet North America.</li>
        <li>The comparison for a file without the variable is a convention: the fill value the other files declare.</li>
      </ul>
    </div>
    <div>
      <ul>
        <li>Daymet was run for ten options; the Kerchunk test for two; the worst cases for eleven.</li>
        <li>Timings are from one machine. Nothing ran against S3.</li>
        <li>How the store knows to prepend an index is a stand-in (it reads a codec option); a real design needs a flag of its own.</li>
      </ul>
      <h3>Changed since the first version</h3>
      <ul>
        <li>Position's table is written with each case once. It had repeated the codec chain for every run, which made its metadata look {{megabytes}} large.</li>
        <li>Stored size is reported beside the size of the document, and the side-array option carries per-chunk numbers like the others.</li>
        <li>The edit test runs seven kinds of edit on every option with nothing decided in advance, and judges each on several reads, not only a read of the whole array.</li>
        <li>The position codec checks that its table covers the array, which turned most of its silent failures into refusals.</li>
        <li>Timings are medians of three, and the claim that the options do not differ on speed is limited to what the data shows.</li>
      </ul>
    </div>
  </div>
</section>

</div>
""".replace("{megabytes}", "megabytes")
