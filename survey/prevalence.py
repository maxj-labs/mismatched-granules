"""How often do NASA collections receive data anywhere but at the end of the record?

    python prevalence.py [short_name ...]

Public CMR granule metadata only. For each collection, the most recent granules by data
time (up to CAP) are put in the order their files were produced, and each is asked:
when this file was made, how far behind the newest data then in hand did it start?

    at the end      it starts at or after everything produced before it (an append),
                    or within a day of that (granules of one day made out of order)
    late            it starts more than a day, a week, or a month before data already
                    in hand: an insertion into the record, or a backfill

and separately:

    replaced        its file was produced more than a day after the granule first
                    entered CMR: the file was replaced in place (reprocessed)
    revised         its CMR record has been revised (weaker: also true for a change of
                    metadata alone)

"Produced" is the granule's ProductionDateTime, the time the file was made. A reprocessing
campaign remakes a whole record, often not in data order, so results are given for all
granules sampled and for those produced in the last 12 months of the sample, which is
closer to what a store that is kept up to date would see.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

CMR = "https://cmr.earthdata.nasa.gov/search/"
HEADERS = {"User-Agent": "cmr-metadata-survey", "Client-Id": "cmr-metadata-survey"}
CAP = 60_000
OUT = Path(__file__).resolve().parent.parent / "results" / "prevalence.json"
COLLECTIONS = [
    # surveyed earlier in this project
    "Daymet_Daily_V4R1_2129", "MODISA_L3m_CHL", "OMPS_NPP_NMTO3_L3_DAILY", "GPM_2ADPR", "ATL08", "TEMPO_NO2_L2", "OCO2_L2_Lite_FP",
    # other widely used collections, gridded and swath
    "MUR-JPL-L4-GLOB-v4.1", "GPM_3IMERGDF", "GPM_3IMERGHH", "M2T1NXSLV", "HLSL30", "HLSS30", "SPL3SMP", "MOD13Q1", "VNP09GA",
    "NLDAS_FORA0125_H", "GLDAS_NOAH025_3H", "AIRS3STD", "SWOT_L2_LR_SSH_BASIC_D", "ATL03", "GEDI02_A", "SENTINEL-1A_SLC", "OPERA_L3_DSWX-HLS_V1",
    "ECCO_L4_SSH_05DEG_DAILY_V4R4", "CYGNSS_L3_SOIL_MOISTURE_V3.2",
]


def get(path: str, params: dict, headers: dict | None = None) -> tuple[dict, dict]:
    url = CMR + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={**HEADERS, **(headers or {})}), timeout=120) as r:
                return json.load(r), dict(r.headers)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError


def when(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pick_collection(short_name: str) -> dict | None:
    """The version of a collection with the most granules."""
    d, _ = get("collections.umm_json", {"short_name": short_name, "page_size": 50, "include_granule_counts": "true"})
    items = [i for i in d.get("items", []) if i["meta"].get("granule-count")]
    if not items:
        return None
    best = max(items, key=lambda i: i["meta"]["granule-count"])
    return {"concept_id": best["meta"]["concept-id"], "short_name": best["umm"]["ShortName"], "version": best["umm"].get("Version"),
            "granules_in_cmr": best["meta"]["granule-count"], "provider": best["meta"]["provider-id"]}


def granules(concept_id: str) -> list[dict]:
    out, after = [], None
    while len(out) < CAP:
        d, h = get("granules.umm_json", {"collection_concept_id": concept_id, "page_size": 2000, "sort_key": "-start_date"},
                   {"CMR-Search-After": after} if after else None)
        items = d.get("items", [])
        for g in items:
            u, m = g["umm"], g["meta"]
            rng = u.get("TemporalExtent", {}).get("RangeDateTime") or {}
            single = u.get("TemporalExtent", {}).get("SingleDateTime")
            dates = {p["Type"]: p["Date"] for p in u.get("ProviderDates", [])}
            out.append({
                "start": when(rng.get("BeginningDateTime") or single), "end": when(rng.get("EndingDateTime") or single),
                "produced": when(u.get("DataGranule", {}).get("ProductionDateTime")),
                "inserted": when(dates.get("Insert") or dates.get("Create")), "revision": m.get("revision-id", 1),
            })
        after = next((v for k, v in h.items() if k.lower() == "cmr-search-after"), None)
        if not items or not after:
            break
    return out


def classify(rows: list[dict]) -> dict:
    """Walk the granules in the order their files were produced."""
    rows = sorted(rows, key=lambda r: r["produced"])
    counts = {"at the end": 0, "late by 1 to 7 days": 0, "late by 7 to 30 days": 0, "late by more than 30 days": 0}
    # The frontier a granule is held against is the newest data among files made at
    # least a day before it. Files made together (a delivery, a reprocessing batch) are
    # not held against each other: the order within a batch means nothing to a store.
    frontier, behind = None, 0
    for r in rows:
        while rows[behind]["produced"] <= r["produced"] - timedelta(days=1):
            frontier = rows[behind]["start"] if frontier is None else max(frontier, rows[behind]["start"])
            behind += 1
        if frontier is None or r["start"] >= frontier - timedelta(days=1):
            counts["at the end"] += 1
        else:
            lag = frontier - r["start"]
            counts["late by 1 to 7 days" if lag <= timedelta(days=7) else "late by 7 to 30 days" if lag <= timedelta(days=30) else "late by more than 30 days"] += 1
    n = len(rows)
    replaced = sum(1 for r in rows if r["inserted"] and r["produced"] - r["inserted"] > timedelta(days=1))
    return {"granules": n, **{k: round(v / n, 4) for k, v in counts.items()},
            "late_by_more_than_a_day": round(1 - counts["at the end"] / n, 4),
            "file_replaced_after_first_ingest": round(replaced / n, 4),
            "record_revised": round(sum(1 for r in rows if r["revision"] > 1) / n, 4)}


def survey(short_name: str) -> dict:
    c = pick_collection(short_name)
    if c is None:
        return {"short_name": short_name, "error": "no collection with granules found under this short name"}
    rows = granules(c["concept_id"])
    usable = [r for r in rows if r["start"] and r["produced"]]
    # A collection may hold products of several periods (daily, 8-day, monthly, a running
    # total). Keep the commonest period, so one series is followed.
    spans = [round(((r["end"] or r["start"]) - r["start"]).total_seconds() / 3600) for r in usable]
    if spans:
        common = max(set(spans), key=spans.count)
        c["granule_hours"], c["share_with_that_period"] = common, round(spans.count(common) / len(spans), 3)
        if c["share_with_that_period"] < 0.98:
            usable = [r for r, s_ in zip(usable, spans) if abs(s_ - common) <= max(1, 0.1 * common)]
    c.update(sampled=len(rows), with_production_time=len(usable))
    if len(usable) < 50:
        c["error"] = "too few granules carry a production time"
        return c
    c["data_from"], c["data_to"] = min(r["start"] for r in usable).date().isoformat(), max(r["start"] for r in usable).date().isoformat()
    c["produced_from"], c["produced_to"] = min(r["produced"] for r in usable).date().isoformat(), max(r["produced"] for r in usable).date().isoformat()
    c["all"] = classify(usable)
    cutoff = max(r["produced"] for r in usable) - timedelta(days=365)
    recent = [r for r in usable if r["produced"] >= cutoff]
    # the frontier a store would hold when the last year began: everything produced before it
    before = [r for r in usable if r["produced"] < cutoff]
    if len(recent) >= 50:
        newest = max(before, key=lambda r: r["start"]) if before else None
        held = [{**newest, "produced": cutoff - timedelta(days=2)}] if newest else []
        c["last_12_months"] = classify(held + recent)
        c["last_12_months"]["granules"] = len(recent)
    # how long after its data a file is usually made, to tell forward processing from reprocessing
    latency = sorted((r["produced"] - r["start"]).total_seconds() / 86400 for r in usable)
    c["median_days_from_data_to_production"] = round(latency[len(latency) // 2], 1)
    return c


if __name__ == "__main__":
    names = sys.argv[1:] or COLLECTIONS
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        try:
            rec = survey(name)
        except Exception as err:
            rec = {"short_name": name, "error": f"{type(err).__name__}: {str(err)[:200]}"}
        report[name] = rec
        OUT.write_text(json.dumps(report, indent=1))
        a, r = rec.get("all", {}), rec.get("last_12_months", {})
        print(f"{name[:28]:28s} v{str(rec.get('version'))[:6]:6s} n={rec.get('with_production_time', 0):6d}  late: all {a.get('late_by_more_than_a_day', '')!s:7} last-yr {r.get('late_by_more_than_a_day', '')!s:7}"
              f"  >30d: {a.get('late by more than 30 days', '')!s:7} {r.get('late by more than 30 days', '')!s:7}  replaced {a.get('file_replaced_after_first_ingest', '')!s:7} revised {a.get('record_revised', '')!s:7} {rec.get('error', '')}", flush=True)
