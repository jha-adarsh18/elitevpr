"""Resolution of the cached teacher descriptors.

One `.npy` per sequence, one row per frame, row-aligned to
`<features_dir>/<seq>/frames.txt`. A cache with a different row count is a
cache of different frames, so every target drawn from it would belong to the
wrong frame: that is checked here and raised, never worked around.
"""

from pathlib import Path

import numpy as np


class TeacherCache:
    """Locates `<root>/<seq>/<filename>` across one or more cache roots.

    Several roots exist because a corpus added later cannot be written into a
    read-only mount holding the earlier caches. Sequence names are unique
    across corpora, so the first root holding the file is unambiguous.
    """

    def __init__(self, roots, filename):
        if isinstance(roots, str):
            roots = [r.strip() for r in roots.split(",") if r.strip()]
        self.roots = [Path(r) for r in roots]
        self.filename = filename
        if not self.roots:
            raise ValueError("teacher.dirs is empty")

    def path(self, seq):
        tried = [root / seq / self.filename for root in self.roots]
        for path in tried:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"{seq}/{self.filename} not found under any of "
            f"{[str(p) for p in tried]} -- cache that corpus first."
        )

    def open(self, seq, features_dir):
        path = self.path(seq)
        array = np.load(path, mmap_mode="r")
        n_frames = _count_frames(Path(features_dir) / seq / "frames.txt")
        if array.shape[0] != n_frames:
            raise RuntimeError(
                f"{path} has {array.shape[0]} rows but {seq}/frames.txt has "
                f"{n_frames}. Re-cache {seq}."
            )
        return array

    def width(self, seq):
        return int(np.load(self.path(seq), mmap_mode="r").shape[1])


def _count_frames(frames_file):
    if not frames_file.is_file():
        raise FileNotFoundError(f"{frames_file} not found")
    return sum(1 for line in frames_file.read_text().splitlines() if line.strip())
