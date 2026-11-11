"""The reads every arm performs, and the values they must return."""

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"


class Workloads:
    def __init__(self, region: str, var: str, first: int, last: int) -> None:
        self.var, self.years = var, list(range(first, last + 1))
        self.paths = [DATA / f"daymet_v4_daily_{region}_{var}_{y}.nc" for y in self.years]
        self.files = [h5py.File(p, "r") for p in self.paths]
        nt = [f[var].shape[0] for f in self.files]
        self.offsets = np.concatenate([[0], np.cumsum(nt)])
        self.NT, self.NY, self.NX = int(self.offsets[-1]), *self.files[0][var].shape[1:]
        seam = int(self.offsets[self.years.index(2020)]) if 2020 in self.years and first < 2020 else self.NT // 2
        valid = np.argwhere(self.files[0][var][0] != -9999)
        py, px = (int(v) for v in valid[len(valid) // 2])
        by, bx = int(np.clip(py - 25, 0, self.NY - 50)), int(np.clip(px - 25, 0, self.NX - 50))
        decade = slice(max(0, seam - 5 * 365), min(self.NT, seam + 5 * 365))
        self.selections = {
            "map: 2 days, whole grid, across the 2020 change": (slice(seam - 1, seam + 1), slice(None), slice(None)),
            "box: 30 days x 50x50 cells, across the change": (slice(seam - 15, seam + 15), slice(by, by + 50), slice(bx, bx + 50)),
            "series: 10 years at one cell, across the change": (decade, py, px),
            "series: every day at one cell": (slice(0, self.NT), py, px),
        }

    def truth(self, t: slice, y, x) -> np.ndarray:
        parts = []
        for i, f in enumerate(self.files):
            lo, hi = max(t.start, self.offsets[i]), min(t.stop, self.offsets[i + 1])
            if lo < hi:
                parts.append(f[self.var][lo - self.offsets[i] : hi - self.offsets[i], y, x])
        return np.concatenate(parts)
