"""Position or key: what happens when a store is edited after it is built.

    python examples/reproduce.py

Needs only the zarr-python branch (https://github.com/maxj-labs/zarr-python/tree/feat/array-concat-codec).
Everything is in memory; no files, no Icechunk.

A record of six chunks was written in two eras: the first three chunks are little-endian
int16, the last three big-endian. One array presents all six. Its `switch` codec knows
which chunk is which, either from the chunk's position or from a key in front of each
chunk. Two edits follow, each made by a writer that changes chunks and leaves the codec's
configuration as it found it.

The eras here differ only in byte order and nothing is compressed, so a chunk read with
the wrong layout cannot fail: this is the worst case for a silent error. With compressed
data the wrong decoder usually raises instead.
"""

import numpy as np
import zarr
from zarr.codecs import SwitchCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

store, proto = MemoryStore(), default_buffer_prototype()
data = np.arange(6 * 4, dtype="int16").reshape(6, 4) * 100
little = [{"name": "bytes", "configuration": {"endian": "little"}}]
big = [{"name": "bytes", "configuration": {"endian": "big"}}]
KEY = {"little": bytes.fromhex("a1" * 8), "big": bytes.fromhex("b2" * 8)}
common = dict(chunks=(1, 4), dtype="int16", fill_value=0, compressors=None, filters=None)

by_position = SwitchCodec(axis=0, cases=[{"chunks": 3, "codecs": little}, {"chunks": 3, "codecs": big}])
by_key = SwitchCodec(select="key", cases=[{"key": KEY["little"].hex(), "codecs": little}, {"key": KEY["big"].hex(), "codecs": big}])


def write(name: str, codec: SwitchCodec, eras: list[str], rows: list[int], keyed: bool) -> zarr.Array:
    """An array of the given rows of `data`, row i stored as era `eras[i]` wrote it."""
    zarr.create_array(store, name=name, shape=(len(rows), 4), serializer=codec, **common)
    for i, (row, era) in enumerate(zip(rows, eras)):
        raw = data[row].astype("<i2" if era == "little" else ">i2").tobytes()
        sync(store.set(f"{name}/c/{i}/0", proto.buffer.from_bytes((KEY[era] if keyed else b"") + raw)))
    return zarr.open_array(store, path=name, mode="r")


def report(label: str, name: str, codec: SwitchCodec, eras: list[str], rows: list[int], keyed: bool) -> None:
    try:
        got = write(name, codec, eras, rows, keyed)[:]
    except Exception as err:
        print(f"  {label}: refused with an error ({str(err).split('.')[0]})")
        return
    wrong = int((got != data[rows]).sum())
    print(f"  {label}: {'correct' if not wrong else f'{wrong} of {got.size} values wrong, and no error'}")
    if wrong:
        row = next(i for i in range(len(rows)) if (got[i] != data[rows[i]]).any())
        print(f"    expected row {row}: {data[rows[row]].tolist()}")
        print(f"    read row {row}:     {got[row].tolist()}")


eras = ["little"] * 3 + ["big"] * 3
print("as built: six chunks, two eras")
report("by position", "p0", by_position, eras, list(range(6)), keyed=False)
report("by key     ", "k0", by_key, eras, list(range(6)), keyed=True)

print("\nedit 1: the first two chunks are dropped, and the rest move up")
report("by position", "p1", by_position, eras[2:], list(range(2, 6)), keyed=False)
report("by key     ", "k1", by_key, eras[2:], list(range(2, 6)), keyed=True)

print("\nedit 2: chunk 1 is replaced by a reprocessed copy, now big-endian like the newer era")
replaced = ["little", "big", "little", "big", "big", "big"]
report("by position", "p2", by_position, replaced, list(range(6)), keyed=False)
report("by key     ", "k2", by_key, replaced, list(range(6)), keyed=True)
