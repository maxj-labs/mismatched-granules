# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "tifffile", "numpy"]
# ///
"""Download Daymet granules by name pattern: download.py <region> <var> <year> [<year> ...]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from probe import resolve
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = {r[0]: r for r in json.load(open(f"{here}/scripts/daymet_granules.json"))}
region, var, *years = sys.argv[1:]
for y in years:
    name = f"daymet_v4_daily_{region}_{var}_{y}.nc"; out = f"{here}/data/{name}"
    if os.path.exists(out): print("have", name); continue
    s, final, size, host = resolve(rows[name][2]); t0 = time.time()
    with s.get(final, stream=True, timeout=300) as r, open(out + ".part", "wb") as f:
        r.raise_for_status()
        for c in r.iter_content(1 << 22): f.write(c)
    os.rename(out + ".part", out)
    print(name, round(size / 1e6), "MB", round(time.time() - t0), "s", flush=True)
