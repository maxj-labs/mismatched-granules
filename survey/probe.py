# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "tifffile", "numpy", "imagecodecs"]
# ///
"""Range-request probe of ASF Sentinel-1 zips: zip entry methods and inner TIFF layout."""
import io, json, os, sys, zipfile, time
from urllib.parse import urlparse
import requests

TOKEN = os.environ.get("EARTHDATA_TOKEN", "")
OK_HOSTS = (".asf.alaska.edu", ".earthdata.nasa.gov", ".earthdatacloud.nasa.gov")

def resolve(url):
    """Follow redirects by hand; bearer token only to ASF/Earthdata hosts. Returns final URL + headers."""
    s = requests.Session()
    for _ in range(12):
        host = urlparse(url).hostname or ""
        h = {"Range": "bytes=0-0"}
        if TOKEN and host.endswith(OK_HOSTS):
            h["Authorization"] = f"Bearer {TOKEN}"
        for k in range(5):
            r = s.get(url, headers=h, allow_redirects=False, timeout=60)
            if r.status_code < 500: break
            time.sleep(2 ** k)
        if r.status_code in (301, 302, 303, 307, 308):
            url = requests.compat.urljoin(url, r.headers["Location"]); continue
        r.raise_for_status()
        size = int(r.headers["Content-Range"].split("/")[1])
        return s, url, size, host
    raise RuntimeError("too many redirects")

class RangeFile(io.RawIOBase):
    def __init__(self, s, url, size, block=1 << 18):
        self.s, self.url, self.size, self.block = s, url, size, block
        self.pos = 0; self.cache = {}; self.nreq = 0; self.nbytes = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block; b = min(a + self.block, self.size) - 1
            for k in range(5):
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code < 500: break
                time.sleep(2 ** k)
            r.raise_for_status(); self.cache[i] = r.content
            self.nreq += 1; self.nbytes += len(r.content)
        return self.cache[i]
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos); out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

class Sub(io.RawIOBase):
    def __init__(self, f, start, size): self.f, self.start, self.size, self.pos = f, start, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        self.f.seek(self.start + self.pos); d = self.f.read(n); self.pos += len(d); return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def probe(url, label):
    import tifffile
    s, final, size, host = resolve(url)
    f = RangeFile(s, final, size)
    zf = zipfile.ZipFile(f)
    out = {"label": label, "zip_bytes": size, "served_by": host, "entries": []}
    for zi in zf.infolist():
        if zi.is_dir(): continue
        out["entries"].append({"name": zi.filename.split("/", 1)[-1], "method": zi.compress_type,
                               "csize": zi.compress_size, "usize": zi.file_size, "lho": zi.header_offset})
    out["open_requests"], out["open_bytes"] = f.nreq, f.nbytes
    meth = {}
    for e in out["entries"]:
        m = meth.setdefault(e["method"], [0, 0]); m[0] += 1; m[1] += e["csize"]
    out["by_method"] = {str(k): {"n": v[0], "bytes": v[1]} for k, v in meth.items()}
    # inner TIFF layout of the first measurement tiff
    tiffs = [zi for zi in zf.infolist() if zi.filename.lower().endswith((".tiff", ".tif")) and "measurement" in zi.filename]
    out["tiffs"] = []
    for zi in [z for z in tiffs if z.compress_type == 0][:1]:
        r0, b0 = f.nreq, f.nbytes
        import struct
        f.seek(zi.header_offset); lh = f.read(30)
        nlen, xlen = struct.unpack("<HH", lh[26:30])
        data_start = zi.header_offset + 30 + nlen + xlen
        assert zi.compress_type == 0, "inner TIFF is compressed in the zip"
        inner = Sub(f, data_start, zi.file_size)
        if True:
            with tifffile.TiffFile(inner, name="x.tiff") as t:
                p = t.pages[0]
                offs, cnts = p.dataoffsets, p.databytecounts
                d = {"name": zi.filename.split("/", 1)[-1], "zip_method": zi.compress_type,
                     "zip_data_start": data_start, "bigtiff": t.is_bigtiff, "byteorder": t.byteorder,
                     "shape": list(p.shape), "dtype": str(p.dtype), "compression": str(p.compression),
                     "sampleformat": str(p.sampleformat), "bitspersample": p.bitspersample,
                     "is_tiled": p.is_tiled, "rowsperstrip": p.rowsperstrip, "n_segments": len(offs),
                     "first_offsets": list(map(int, offs[:4])), "first_counts": list(map(int, cnts[:4])),
                     "contiguous": bool(all(offs[i] + cnts[i] == offs[i + 1] for i in range(len(offs) - 1))),
                     "uniform_counts": len(set(cnts[:-1])) <= 1, "n_pages": len(t.pages),
                     "tags": sorted(tg.name for tg in p.tags)}
        d["header_requests"], d["header_bytes"] = f.nreq - r0, f.nbytes - b0
        out["tiffs"].append(d)
    out["total_requests"], out["total_bytes"] = f.nreq, f.nbytes
    return out

if __name__ == "__main__":
    for label, url in [a.split("=", 1) for a in sys.argv[1:]]:
        try:
            res = probe(url, label)
        except Exception as ex:
            res = {"label": label, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
        json.dump(res, open(os.path.join(os.path.dirname(__file__), "out", label + ".json"), "w"), indent=1)
        brief = {k: v for k, v in res.items() if k != "entries"}
        print(json.dumps(brief, indent=1))
        for e in res.get("entries", [])[:60]: print("  ", e["method"], e["csize"], e["usize"], e["name"])
