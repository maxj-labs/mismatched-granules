import json
import os
import urllib.request

HTTP = os.environ.get("BENCH_HTTP")


def reset_source() -> None:
    if HTTP:
        urllib.request.urlopen(HTTP + "__reset").read()


def source_stats() -> dict:
    """Requests and megabytes that reached the source files since the last reset."""
    if not HTTP:
        return {}
    stats = json.loads(urllib.request.urlopen(HTTP + "__stats").read())
    return {"source_requests": stats["requests"], "source_megabytes": round(stats["bytes"] / 1e6, 2)}


from zarr.storage import WrapperStore


class Counting(WrapperStore):
    """Counts the store reads zarr makes: how many, and how many bytes came back."""

    gets = 0
    nbytes = 0

    async def get(self, key, prototype, byte_range=None):
        out = await self._store.get(key, prototype, byte_range)
        type(self).gets += 1
        type(self).nbytes += 0 if out is None else len(out)
        return out

    async def get_partial_values(self, prototype, key_ranges):
        out = await self._store.get_partial_values(prototype, key_ranges)
        type(self).gets += len(out)
        type(self).nbytes += sum(0 if o is None else len(o) for o in out)
        return out

    async def get_ranges(self, key, byte_ranges, *, prototype, **kwargs):
        async for group in self._store.get_ranges(key, byte_ranges, prototype=prototype, **kwargs):
            for item in group:
                buf = item[1] if isinstance(item, tuple) else item
                type(self).gets += 1
                type(self).nbytes += 0 if buf is None else len(buf)
            yield group
