"""Item 1: the frame index -- exactly the bins the bench scores, exported.

One CSV per traverse:

    traverse, bin_index, t_start, t_end, utm_east, utm_north

plus a SHA-256 per file and a manifest over all of them.

Where each column comes from, and why it is not recomputed independently:

* `bin_index`, `utm_east`, `utm_north` come from the frame filenames
  (`@utm_east@utm_north@frame_N@.png`) read through the bench's own
  `read_images_paths` ordering, which is the sort on `@frame_N@`. These are the
  coordinates `TestDataset` itself uses to find positives, so the index and the
  ground truth cannot disagree by construction.
* `t_start`, `t_end` do not exist in the frames -- the bench forms them inside
  `process_sequence` and keeps only the rendered image. They are recovered by
  re-running that same construction (`export_bins`): uniform `window`-second
  bins over the traverse's GPS-aligned time base, with Brisbane's per-recording
  clock offset applied by the bench's own `load_gps`
  (`gps_times += self.video_beginning[seq]`, brisbane_events.py:395), then the
  non-empty mask that drops bins containing no events. Only the event
  TIMESTAMP column is read for that mask, never the events themselves, so the
  pass costs a column scan rather than a reconstruction.

A traverse whose surviving-bin count does not equal its frame count is
reported, not silently reconciled: the bench pairs frames with positions via
`zip(frames, gt_positions)` (load_and_save.py:190) where `gt_positions` is
indexed over ALL bins and `frames` over the surviving ones, so any dropped bin
shifts every later position. That is exactly the misalignment this file exists
to make visible.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .cache import hash_file

COLUMNS = ("traverse", "bin_index", "t_start", "t_end", "utm_east",
           "utm_north")


def read_traverse_frames(bench, frames_dir):
    """Bin index and UTM in the bench's own order, from the filenames."""
    # the bench's own ordering (the sort on @frame_N@), taken from the module
    # `bench.load()` already imported -- a second import could pick up a
    # different `test_dataset` entirely; see bench._BARE_IMPORTS.
    paths = bench.read_images_paths(str(frames_dir))
    rows = []
    for path in paths:
        parts = Path(path).name.split("@")
        # @utm_east@utm_north@frame_N@.png
        rows.append((int(parts[3].replace("frame_", "")),
                     float(parts[1]), float(parts[2])))
    index = np.array(rows, dtype=[("bin_index", np.int64),
                                  ("utm_east", np.float64),
                                  ("utm_north", np.float64)])
    return index


def write_traverse(out_dir, traverse, index, bins=None):
    """One CSV for one traverse. `bins` supplies t_start/t_end when available."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{traverse}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in index:
            i = int(row["bin_index"])
            if bins is None:
                t_start = t_end = ""
            else:
                t_start, t_end = bins[i]
                t_start, t_end = f"{t_start:.9f}", f"{t_end:.9f}"
            writer.writerow([traverse, i, t_start, t_end,
                             f"{row['utm_east']:.6f}",
                             f"{row['utm_north']:.6f}"])
    return path


def export_positions(bench, data_root, traverses, out_dir, representation,
                     window):
    """The cheap half: index + UTM for every traverse, no event data needed."""
    from .protocol import frames_dir

    records, indices = [], {}
    for traverse in traverses:
        directory = frames_dir(data_root, representation, window) / traverse
        if not directory.is_dir():
            records.append({"traverse": traverse, "error":
                            f"no frames at {directory}"})
            continue
        index = read_traverse_frames(bench, directory)
        indices[traverse] = index
        path = write_traverse(out_dir, traverse, index)
        records.append({
            "traverse": traverse, "file": path.name,
            "sha256": hash_file(path), "n_frames": int(len(index)),
            "bin_index_min": int(index["bin_index"].min()),
            "bin_index_max": int(index["bin_index"].max()),
            "bin_indices_contiguous_from_zero": bool(
                np.array_equal(index["bin_index"],
                               np.arange(len(index), dtype=np.int64))),
            "has_timestamps": False,
            "frames_dir": str(directory),
        })
    return records, indices


def write_manifest(out_dir, records, extra=None):
    out_dir = Path(out_dir)
    manifest = {"columns": list(COLUMNS), "traverses": records}
    if extra:
        manifest.update(extra)
    path = out_dir / "frame_index_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def load_index(path):
    """Read a traverse CSV back as a recarray, for the ground-truth check."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["bin_index"]),
                         float(row["utm_east"]), float(row["utm_north"])))
    return np.array(rows, dtype=[("bin_index", np.int64),
                                 ("utm_east", np.float64),
                                 ("utm_north", np.float64)])


def utms(index):
    return np.stack([index["utm_east"], index["utm_north"]], axis=1)
