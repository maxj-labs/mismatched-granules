# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "h5py", "numpy", "tifffile"]
# ///
"""Choose and download a few granules per collection into ../data/survey/<name>/.

For the two gridded collections, a binary search over the record (metadata-only reads)
finds the first granule whose layout differs from the first one, and the granules on
each side of that change are taken.
"""
import json, os, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(__file__))
import h5py, requests
from probe import resolve, RangeFile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json?sort_key=start_date"


def query(short_name, provider, pattern=None, extra=""):
    q = f"{CMR}&short_name={short_name}&provider={provider}{extra}"
    if pattern:
        q += "&options[readable_granule_name][pattern]=true&readable_granule_name=" + urllib.parse.quote(pattern)
    return q


def item(q, offset):
    u = requests.get(q + f"&page_size=1&offset={offset}", timeout=60).json()["items"][0]["umm"]
    url = next(x["URL"] for x in u["RelatedUrls"] if x["Type"] == "GET DATA" and x["URL"].startswith("https"))
    return u["GranuleUR"], url


def signature(url, var):
    s, final, size, _ = resolve(url)
    with h5py.File(RangeFile(s, final, size, block=1 << 20), "r") as h:
        d = h[var]
        pl = d.id.get_create_plist()
        return json.dumps([list(d.chunks) if d.chunks else None, [pl.get_filter(i)[0] for i in range(pl.get_nfilters())]])


def seam(q, var):
    hits = int(requests.get(q + "&page_size=0", timeout=60).headers["CMR-Hits"])
    first = signature(item(q, 0)[1], var)
    lo, hi, probes = 0, hits - 1, 2
    assert signature(item(q, hi)[1], var) != first
    while hi - lo > 1:
        mid = (lo + hi) // 2
        probes += 1
        if signature(item(q, mid)[1], var) == first:
            lo = mid
        else:
            hi = mid
    return hi, probes, hits


def download(name, picks):
    out = os.path.join(HERE, "data", "survey", name)
    os.makedirs(out, exist_ok=True)
    files = []
    for ur, url in picks:
        path = os.path.join(out, url.rsplit("/", 1)[1])
        if not os.path.exists(path):
            s, final, size, _ = resolve(url)
            t0 = time.time()
            with s.get(final, stream=True, timeout=600) as r, open(path + ".part", "wb") as f:
                r.raise_for_status()
                for c in r.iter_content(1 << 22):
                    f.write(c)
            os.rename(path + ".part", path)
            print("  ", os.path.basename(path), round(size / 1e6, 1), "MB", round(time.time() - t0), "s", flush=True)
        files.append(os.path.relpath(path, os.path.join(HERE, "data")))
    return files


picks = json.load(open("picks.json")) if os.path.exists("picks.json") else {}
which = sys.argv[1:]
SPECS = {
    "MODISA_L3m_CHL": dict(q=query("MODISA_L3m_CHL", "OB_CLOUD", "*DAY.CHL.chlor_a.4km.nc"), seam_var="chlor_a", each_side=3),
    "OMPS_NPP_NMTO3_L3_DAILY": dict(q=query("OMPS_NPP_NMTO3_L3_DAILY", "GES_DISC"), seam_var="ColumnAmountO3", each_side=3),
    "GPM_2ADPR": dict(q=query("GPM_2ADPR", "GES_DISC", extra="&version=07&temporal=2020-05-09T00:00:00Z,2020-05-10T00:00:00Z"), take=2),
    "ATL08": dict(q=query("ATL08", "NSIDC_CPRD", extra="&temporal=2022-09-27T00:00:00Z,2022-09-28T00:00:00Z"), take=4),
    "TEMPO_NO2_L2": dict(q=query("TEMPO_NO2_L2", "LARC_CLOUD", "*V04*", extra="&temporal=2026-06-10T16:00:00Z,2026-06-10T23:00:00Z"), take=4),
    "OCO2_L2_Lite_FP": dict(q=query("OCO2_L2_Lite_FP", "GES_DISC", extra="&temporal=2019-08-22T00:00:00Z,2019-08-26T00:00:00Z"), take=3),
}
for name, spec in SPECS.items():
    if which and name not in which:
        continue
    print(name, flush=True)
    rec = {}
    if "seam_var" in spec:
        index, probes, hits = seam(spec["q"], spec["seam_var"])
        rec.update(seam_index=index, probes=probes, granules_in_series=hits)
        chosen = [item(spec["q"], i) for i in range(index - spec["each_side"], index + spec["each_side"])]
    else:
        chosen = [item(spec["q"], i) for i in range(spec["take"])]
    rec["granules"] = [c[0] for c in chosen]
    rec["files"] = download(name, chosen)
    picks[name] = rec
    json.dump(picks, open("picks.json", "w"), indent=1)
    print("  ", {k: v for k, v in rec.items() if k != "files"}, flush=True)
