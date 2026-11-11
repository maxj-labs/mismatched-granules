# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "h5py", "numpy", "tifffile"]
# ///
"""Sample granules of a collection across its record and report each variable's HDF5 layout,
reading metadata only (range requests).

    layouts.py <short_name> <provider> <n_samples> [granule name pattern] [variable ...]
"""
import json, os, sys, urllib.parse
sys.path.insert(0, os.path.dirname(__file__))
import h5py, requests
from probe import resolve, RangeFile

FN = {1: "deflate", 2: "shuffle", 3: "fletcher32", 4: "szip", 5: "nbit", 6: "scaleoffset", 32015: "zstd", 32001: "blosc"}
short_name, provider, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pattern = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
wanted = sys.argv[5:]
base = f"https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name={short_name}&provider={provider}&sort_key=start_date"
if pattern:
    base += "&options[readable_granule_name][pattern]=true&readable_granule_name=" + urllib.parse.quote(pattern)
hits = int(requests.get(base + "&page_size=0", timeout=60).headers["CMR-Hits"])
out = {"short_name": short_name, "hits": hits, "samples": []}
for k in range(n):
    offset = round(k * (hits - 1) / max(n - 1, 1))
    item = requests.get(base + f"&page_size=1&offset={offset}", timeout=60).json()["items"][0]["umm"]
    url = next(u["URL"] for u in item["RelatedUrls"] if u["Type"] == "GET DATA" and u["URL"].startswith("https"))
    rec = {"granule": item["GranuleUR"], "url": url, "start": item["TemporalExtent"].get("RangeDateTime", {}).get("BeginningDateTime", "")[:10]}
    try:
        s, final, size, host = resolve(url)
        f = RangeFile(s, final, size, block=1 << 20)
        rec["size_mb"] = round(size / 1e6, 1)
        layouts = {}
        with h5py.File(f, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 and obj.size > 1000:
                    pl = obj.id.get_create_plist()
                    layouts[name] = {"shape": list(obj.shape), "chunks": list(obj.chunks) if obj.chunks else None, "dtype": str(obj.dtype),
                                     "filters": [FN.get(pl.get_filter(i)[0], pl.get_filter(i)[0]) for i in range(pl.get_nfilters())],
                                     "nbytes": obj.size * obj.dtype.itemsize}
            h.visititems(visit)
        keep = wanted or [k for k, _ in sorted(layouts.items(), key=lambda kv: -kv[1]["nbytes"])[:3]]
        rec["n_datasets"] = len(layouts)
        rec["layouts"] = {k: layouts[k] for k in keep if k in layouts}
        rec["requests"], rec["mb_read"] = f.nreq, round(f.nbytes / 1e6, 1)
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    out["samples"].append(rec)
    print(rec["start"], rec["granule"][:60], rec.get("size_mb"), rec.get("error", ""), flush=True)
    for k2, v in rec.get("layouts", {}).items():
        print("    ", k2[:50], v["shape"], v["chunks"], v["dtype"], v["filters"], flush=True)
os.makedirs("layouts", exist_ok=True)
json.dump(out, open(f"layouts/{short_name}.json", "w"), indent=1)
