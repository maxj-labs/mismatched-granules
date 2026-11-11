"""Serve data/ over HTTP with byte ranges, counting every request to a file.

    python range_server.py <port> <delay_ms>

GET /__stats returns {"requests": n, "bytes": n}; GET /__reset zeroes them. Each file
request sleeps <delay_ms> first, to stand in for object-store latency.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DELAY = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.0
LOCK = threading.Lock()
STATS = {"requests": 0, "bytes": 0}
LOG: list = []


RESOLVED: dict = {}
LOCAL = threading.local()


_SESSION = None


def session():
    """One session for every handler thread, so connections to the archive are reused."""
    global _SESSION
    import requests
    from requests.adapters import HTTPAdapter

    with LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
            _SESSION.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=64))
    return _SESSION


def resolved(url: str):
    """The signed URL an Earthdata link redirects to, and the object's size (cached)."""
    with LOCK:
        if url in RESOLVED:
            return RESOLVED[url]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "survey"))
    from probe import resolve

    _, final, size, _ = resolve(url)
    with LOCK:
        RESOLVED[url] = (final, size)
    return final, size


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        if self.path == "/__stats":
            with LOCK:
                return self._json(dict(STATS))
        if self.path == "/__log":
            with LOCK:
                return self._json({"log": LOG})
        if self.path == "/__reset":
            with LOCK:
                STATS.update(requests=0, bytes=0)
                LOG.clear()
            return self._json({})
        self._serve(head=False)

    def _remote(self, head: bool):
        """Relay a request for /remote/<host>/<path> to https://<host>/<path>, with the
        Earthdata token, so that only the bytes asked for leave the archive."""
        url = "https://" + self.path[len("/remote/"):]
        try:
            final, size = resolved(url)
            if head:
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", f'"{size}"')
                self.end_headers()
                return
            rng = self.headers.get("Range")
            for attempt in range(3):
                r = session().get(final, headers={"Range": rng} if rng else {}, timeout=120)
                if r.status_code in (401, 403) and attempt == 0:  # the signed URL expired
                    with LOCK:
                        RESOLVED.pop(url, None)
                    final, size = resolved(url)
                    continue
                if r.status_code < 500:
                    break
                time.sleep(2 ** attempt)
            body = r.content
            with LOCK:
                STATS["requests"] += 1
                STATS["bytes"] += len(body)
            self.send_response(r.status_code)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{size}"')
            if "Content-Range" in r.headers:
                self.send_header("Content-Range", r.headers["Content-Range"])
            self.end_headers()
            self.wfile.write(body)
        except Exception as err:  # noqa: BLE001
            body = str(err)[:200].encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _serve(self, head: bool):
        if self.path.startswith("/remote/"):
            return self._remote(head)
        path = DATA / self.path.lstrip("/")
        if not path.is_file() or DATA not in path.resolve().parents:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a == "":
                start = max(0, size - int(b))
            else:
                start = int(a)
                end = min(int(b), size - 1) if b else size - 1
            status = 206
        length = end - start + 1
        if not head:
            time.sleep(DELAY)
            with LOCK:
                STATS["requests"] += 1
                STATS["bytes"] += length
                if len(LOG) < 200:
                    LOG.append([self.path, rng, length])
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", f'"{size}"')
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))


if __name__ == "__main__":
    ThreadingHTTPServer.request_queue_size = 2048
    server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
    server.daemon_threads = True
    server.serve_forever()
