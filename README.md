# Mismatched granules

Experiments on reading NASA granules whose layouts do not match (compression, chunk
shape, chunk boundaries, fill value or data type change partway through a record) as
one virtual Zarr array.

The library changes are prototypes on three branches:

| Project | Branch | What it adds |
|---|---|---|
| zarr-python | [`maxj-labs/zarr-python@feat/array-concat-codec`](https://github.com/maxj-labs/zarr-python/tree/feat/array-concat-codec) | `switch` codec, chunk coordinates passed to codecs, `array_concat` codec |
| VirtualiZarr | [`maxj-labs/VirtualiZarr@feat/concat-mismatched-arrays`](https://github.com/maxj-labs/VirtualiZarr/tree/feat/concat-mismatched-arrays) | `set_options(mismatched_concat="pieces")`, `plan_joined_array`, Icechunk writer |
| Icechunk | [`maxj-labs/icechunk@feat/virtual-ref-prefix`](https://github.com/maxj-labs/icechunk/tree/feat/virtual-ref-prefix) | virtual references made of a literal and several byte ranges |

This repository holds what was run against them and what came out.

## Layout

| Path | What |
|---|---|
| `brief/index.html` | The write-up: the proposal and the evidence |
| `bench/` | Every approach on the same Daymet files: `compare.py`, `hdr_arms.py`, `refkv_arm.py`, `na_hybrid.py`, `na_shards.py`; `range_server.py` counts the requests that reach the source and can relay to the archive |
| `survey/` | Other collections: `layouts.py` (layouts from metadata-only reads), `pick.py` (find layout change points, download a few granules), `run.py` and `all_vars.py` (join and check), `remote.py` (whole records, metadata only), `test_fill.py` |
| `scripts/` | Daymet download and a first end-to-end build |
| `results/`, `results_http/` | One JSON file per run. `results_http/` holds the runs that counted source requests |
| `bench/vendor/` | The `switch` codec demo from d-v-b/refkv PR #73, unmodified, used as one arm |

## Running

```bash
uv sync                       # builds Icechunk from source
export EARTHDATA_TOKEN=...    # an Earthdata Login bearer token, for downloads and the relay
uv run scripts/download.py pr tmin 2017 2018 2019 2020 2021 2022
uv run python bench/build.py pr tmin 2017 2022
(cd bench && uv run python compare.py pr tmin 2017 2022)
```

To count requests at the source, start `uv run python bench/range_server.py 8765 20`
(port, added latency in ms) and set `BENCH_HTTP=http://127.0.0.1:8765/` for the other
scripts. `survey/remote.py` needs the server started with `EARTHDATA_TOKEN` set.

Downloaded granules go to `data/` and stores to `stores/`; neither is committed.

## Caveats

- These are prototypes. The numbers are from one workstation, reading local files or a
  local HTTP server; nothing here was measured against S3.
- The environment above has not been tested from a clean checkout: the runs used
  local editable checkouts of the three branches.
- Most of the code and the write-up were produced with Claude Code, directed by
  Max Jones.
