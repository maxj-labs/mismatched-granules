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


def render(g: dict) -> str:
    style = BRIEF.read_text()
    head = style[: style.index("</style>")].replace("<title>One Array from Mismatched Granules</title>", "<title>Every Option, Tested</title>")
    count, scale, size, LABEL, CASES = g["count"], g["scale"], g["size"], g["LABEL"], g["CASES"]
    full = [a for a in g["ARMS"][:10] if count(a) == len(CASES)]
    cases = "".join(f"<li><b>{g['CASE_LABEL'][c]}.</b> {g['e'](g['MATRIX']['cases'][c][0].upper() + g['MATRIX']['cases'][c][1:])}.</li>" for c in CASES)
    pr, na = g["PERF"]["pr"], g["PERF"]["na"]
    refkv = g["REFKV"]
    refkv_rows = {r["workload"]: r for r in refkv.get("rows", [])}
    long_read = "series: every day at one cell"
    high = g["load"]("perf_pr_tmin_concurrency_120.json")

    def seconds(report: dict, arm: str, read: str = long_read) -> float:
        return report["arms"][arm]["reads"][read]["seconds"]

    concat_manifest = scale("array_concat", "alternate", "manifest_bytes") / scale("switch_key", "alternate", "manifest_bytes")
    return f"""{head}{EXTRA_CSS}</style>

<div class="wrap">

<header>
  <section>
    <div class="eyebrow">Evaluation &middot; 11 November 2026</div>
    <h1>Every Option, Tested</h1>
    <p class="lede">The discussions on joining mismatched granules into one Zarr array propose about fifteen different mechanisms. Each was built or emulated and run against the same ten cases, the same requirements and the same real granules. This page reports what each one did.</p>
    <p class="byline">Companion to <a href="https://maxj-labs.github.io/mismatched-granules/brief.html">One Array from Mismatched Granules</a>. Every mark and number below is generated from result files in <a href="https://github.com/maxj-labs/mismatched-granules">maxj-labs/mismatched-granules</a> (<span class="mono">eval/</span>, <span class="mono">results/eval/</span>). Prepared with Claude; the judgements at the end are its reading of the results.</p>
  </section>
</header>

<section>
  <div class="claims">
    <div class="claim">
      <div class="who">Held up best</div>
      <div class="what">A case table, with each chunk's case carried in its own reference</div>
      <div class="why">Described {count("switch_key_parameters")} of {len(CASES)} cases, survived a manifest operation untouched, and kept <code>zarr.json</code> at {size(scale("switch_key_parameters", "packed", "zarr_json_bytes"))} with 16,000 chunks each packed differently. Needs the Icechunk change.</div>
    </div>
    <div class="claim">
      <div class="who">Works today, with a catch</div>
      <div class="what">The same table, with the case found from the chunk's position</div>
      <div class="why">Described {count("switch_position")} of {len(CASES)} and needs no store change. A tool that drops chunks without rewriting the table left it reading wrong values, twice without any error.</div>
    </div>
    <div class="claim">
      <div class="who">Most general, most costly</div>
      <div class="what">Array concatenation</div>
      <div class="why">Described {count("array_concat")} of {len(CASES)} and reads with released Icechunk. With one piece per chunk its manifests were {concat_manifest:.0f}&times; larger and each chunk took {scale("array_concat", "alternate", "microseconds_per_chunk") / scale("switch_key", "alternate", "microseconds_per_chunk"):.0f}&times; longer to read.</div>
    </div>
  </div>
</section>

<section>
  <h2>The options</h2>
  <div class="prose"><p>Taken from zarr-specs#399 and #288, zarr-extensions#74 and PR&nbsp;#75, NASA-IMPACT/veda-odd#498, VirtualiZarr#1111 and refkv PR&nbsp;#73. Seven of them are the same codec, a table of cases, and differ only in how a chunk's case is found; that is the question most of the threads argue about, so each way is tested separately.</p></div>
  {g["options_table"]()}
  <div class="prose">
    <p><b>Virtual shards</b> (d-v-b, zarr-specs#399) are not a row of their own: every switch option uses them when the chunk shape changes, which is the &ldquo;chunk shape change&rdquo; case below.</p>
    <p><b>Proposed but not built.</b></p>
    <ul>
      <li><b>A storage transformer</b> (zarr-specs#399 design 2.3; a generalised transformer chain, mkitti). zarr-python refuses any array that declares one, so there is nothing to run it in.</li>
      <li><b>An &ldquo;array to files&rdquo; codec</b> (jbms). It ends the rule of one key per chunk, which Icechunk, VirtualiZarr and zarr-python all assume. Too large a change to prototype here.</li>
      <li><b>A union data type</b> (d-v-b). It would let one array hold two data types; it does not address codecs, chunk shapes or padding, and the data-type case is handled by a cast in the chain.</li>
      <li><b>Index transforms as in TensorStore</b> (jbms). TensorStore cannot read virtual references, so its stack driver was only measured for cost on native arrays, below.</li>
    </ul>
  </div>
</section>

<section>
  <h2>Can it describe the case?</h2>
  <div class="prose"><p>Ten small sets of netCDF4 or HDF5 files, each a way real granules disagree. An option passes a case when every read (the whole array, a window across each seam, single cells, one row along the joined axis) equals what the files' own reader returns.</p></div>
  {g["matrix_table"]()}
  <div class="key"><span><span class="yes">&#10003;</span> every read equals the files</span><span><span class="dash">&ndash;</span> the option cannot describe the case (hover for why)</span><span><span class="no">&#10007;</span> reads differ from the files</span></div>
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
  <h2>Against the requirements in zarr-specs#399</h2>
  <div class="prose"><p>The issue lists what a solution should do: fail safely for readers that do not understand it, compose with operations on the manifest, and work without Icechunk. Each was tested where it could be.</p></div>
  {g["requirements_table"]()}
  <div class="prose">
    <p><b>Readers without the feature.</b> Every store from the matrix was opened by released zarr-python 3.4.2. All the codec options are refused with <code>UnknownCodecError</code>. The declared domain is the exception: an ordinary reader sees the padding as data and returns an array of the wrong shape with no error. The grid option is refused only by an unrelated reshape error. A reader that has the codecs but released Icechunk refuses every store whose references carry a prefix or several ranges (<code>FormatError</code>), and reads the rest correctly. One refusal is luck: with the index-prepending store, released Icechunk hands the codec chunk bytes with no index in front, and the codec fails only because the bytes it takes for an index point outside its table.</p>
    <p><b>Composition.</b> The first chunks of each array were dropped the way a manifest tool would: shape and chunk grid shortened, references moved up, the codec's configuration untouched. Options that find the case from something stored with the chunk still read correctly. Options that find it from the chunk's position do not, and in the alternating-layout and packing cases they returned wrong values without an error. A store that kept the context with each reference would carry it along, which is how that row is emulated.</p>
    <p><b>Without Icechunk.</b> {g["kerchunk_sentence"]()} Concatenation and the side-array option need only plain references. Key selection was also run on refkv as released, which has such references.</p>
    <p class="note">&dagger; by inspection of how the option stores its data, not by a run.</p>
  </div>
</section>

<section>
  <h2>Worst cases</h2>
  <div class="prose"><p>16,000 one-chunk pieces, cut from real netCDF files so that planners and writers see what 16,000 single-chunk granules would give them. Local disk, so the read time is overhead per chunk and not network. The native copy stores eight steps per chunk, so its per-chunk time is not comparable.</p></div>
  {g["scaling_table"]()}
  <div class="prose">
    <p>{g["growth_note"]()}</p>
    <p>Only two options keep <code>zarr.json</code> flat when every chunk has parameters of its own: carrying the numbers in each reference, and a store that returns them as context. A key alone, like position, then needs one case per chunk listed in <code>zarr.json</code>: it still works, as the 8-byte key row shows with 16,000 cases, but the metadata grows with the record.</p>
    <p>The key's length is a setting (<code>key_bytes</code>), not a fixed property. One byte numbers 256 cases, which is why that row cannot describe 16,000 chunks each packed differently; two bytes number 65,536, and the default of eight is long enough to use a hash of the case. Length has little effect on size: on Daymet Puerto Rico the manifests were 74.7 KB with 1-byte keys and 76.4 KB with 8-byte keys.</p>
    <p>Concatenation with one piece per chunk writes one array per piece, which is what makes its manifests and build time grow. {g["tensorstore_sentence"]()} That is the cost of the general design when the layers are native arrays; it says nothing about virtual ones.</p>
  </div>
</section>

<section>
  <h2>Real granules</h2>
  <div class="prose"><p>The same options on Daymet, with the source behind an HTTP server that waits 20 ms per request and counts them. Median of three reads, each from a new session.</p></div>
  <h3>Daymet Puerto Rico, {pr["years"][0]}&ndash;{pr["years"][1]}: {pr["shape"][0]:,} daily chunks, compression changes in 2020</h3>
  {g["perf_table"]("pr")}
  <h3>Daymet North America, {na["years"][0]}&ndash;{na["years"][1]}: the chunk shape changes in 2020, so every switch option stores virtual shards</h3>
  {g["perf_table"]("na")}
  <div class="prose">
    <p><b>The seven ways of finding the case do not differ on speed.</b> They fetch the same chunks and take the same time to within a few percent, on both datasets. Concatenation matched them on Puerto Rico and made slightly more requests on North America (475 against 439 for the series). The reference-per-chunk codec is the outlier in both directions: faster on the Puerto Rico series, for the reason below, and slower on whole North America maps ({seconds(na, "zarr_reference", "map: 2 days, whole grid, across the 2020 change"):.1f} s against {seconds(na, "switch_position", "map: 2 days, whole grid, across the 2020 change"):.1f} s) with the same requests.</p>
    <p><b>One apparent winner was an artefact.</b> The reference-per-chunk codec read the Puerto Rico series in {seconds(pr, "zarr_reference"):.1f} s against {seconds(pr, "switch_position"):.1f} s. It reads each year through a nested array read, so twelve years ran at once, each with zarr-python's default of ten chunks in flight. With that setting raised to 120 for everyone, position took {seconds(high, "switch_position"):.1f} s, key {seconds(high, "switch_key"):.1f} s, reference {seconds(high, "zarr_reference"):.1f} s and concatenation {seconds(high, "array_concat"):.1f} s. zarr-python's concurrency setting moves these reads by about {seconds(pr, "switch_position") / seconds(high, "switch_position"):.0f}&times;, far more than any choice between the options.</p>
    <p><b>refkv as released</b> (0.2.0, no Icechunk), on the same Puerto Rico files and server: {refkv_rows[long_read]["seconds_median"]:.1f} s for the whole series with {refkv_rows[long_read]["source_requests"]:,} requests, every read equal to the files.</p>
  </div>
</section>

<section>
  <h2>What this changes</h2>
  <div class="prose">
    <p>The earlier brief suggested shipping position selection first and adding key selection later. These results support a different emphasis.</p>
    <ul>
      <li><b>Carry the case with the chunk.</b> A key in each reference, followed by any per-chunk numbers, was the only virtual design that described all ten cases, composed with a manifest operation as it stood, and stayed small in both worst cases. This is refkv PR&nbsp;#73's design. It depends on references that are a literal followed by byte ranges, which the Icechunk branch adds and which virtual shards need anyway.</li>
      <li><b>Treat position as the route that needs no store change, not as the foundation.</b> It reads through released Icechunk and through Kerchunk references. Its table must be rewritten by every tool that reorders chunks, and a tool that does not know to do so can produce silently wrong data.</li>
      <li><b>A store context call would be as good on these tests.</b> It matched key + parameters wherever it was run, but only as an emulation: it needs a new store method, a spec change and a place in the manifest, none of which exist.</li>
      <li><b>Keep concatenation as the fallback.</b> It describes everything and needs nothing from the store. Its costs grow with the number of pieces, not chunks, so it is cheap for a few eras and expensive for many small granules.</li>
      <li><b>The narrow options are narrow.</b> <span class="mono">conditional</span> described {count("conditional_codec")} of {len(CASES)} cases; encoded chunk shapes in the grid and a declared domain {count("grid_encoded_shapes")}; copying the chunks that do not fit {count("materialize_odd_chunks")}, storing half or more of the data again in five of them.</li>
    </ul>
    <p>One gap closed along the way: key selection could not give a run of unwritten chunks its own fill value, because such chunks have no reference. They are now stored as the key alone, and the fill-value case passes.</p>
  </div>
</section>

<section>
  <h2>Limits of this evaluation</h2>
  <div class="two">
    <div>
      <ul>
        <li>Two options are emulations, as marked in the first table: the store context call, and encoded chunk shapes in the grid. An implementation inside Icechunk or zarr-python could behave differently, especially on speed. The store that prepends an index was first emulated and then built into the Icechunk branch; its results did not change.</li>
        <li>The <span class="mono">conditional</span> codec and the key + parameters codec are my implementations of others' designs. refkv itself was run only on Daymet Puerto Rico.</li>
        <li>Only zarr-python reads any of these arrays.</li>
      </ul>
    </div>
    <div>
      <ul>
        <li>The ten cases are small files written for this test. They show whether an option can describe a situation, not how common the situation is.</li>
        <li>The composition test is one operation. Renaming an array, concatenating two joined arrays and nesting under another codec were not run.</li>
        <li>Timings are from one machine against a local HTTP server. Nothing ran against S3.</li>
      </ul>
    </div>
  </div>
</section>

</div>
"""
