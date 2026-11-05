"""One ground truth per (reference, query) pair, built by the bench itself.

The 25 m matrix is produced by calling `TestDataset.get_positives()` -- the
bench's own positive search -- and then checked against a distance threshold
recomputed from the frame index. Both must agree exactly. The 10 m and 70 m
variants come from the same call with a different threshold, so a tolerance
sweep cannot drift away from the protocol.

Two details of the bench's convention that a reimplementation gets wrong, and
which the cross-check therefore pins:

* the comparison is STRICT (`distances < thresh`, test_dataset.py:66), so a
  frame at exactly 25.0 m is NOT a positive;
* query and reference order is the sort on `@frame_N@` in the filename
  (`read_images_paths`), not directory order.

Stored as a bool matrix of shape (n_query, n_ref) -- the same orientation as a
similarity matrix the bench produces, so a scorer never has to transpose and
guess. `no_pos` (queries with no positive at all) is recorded per pair because
the bench has two R@1 denominators that coincide only while it is zero.
"""

import json
from pathlib import Path

import numpy as np

from .cache import hash_file

THRESHOLDS = (10, 25, 70)
PROTOCOL_THRESHOLD = 25


def positives_to_matrix(positives_per_query, n_ref):
    matrix = np.zeros((len(positives_per_query), n_ref), dtype=bool)
    for q, refs in enumerate(positives_per_query):
        if len(refs):
            matrix[q, np.asarray(refs, dtype=int)] = True
    return matrix


def positives_from_utm(query_utms, ref_utms, threshold):
    """The independent check: strict Euclidean threshold on the frame index."""
    d = np.linalg.norm(np.asarray(query_utms, dtype=np.float64)[:, None, :]
                       - np.asarray(ref_utms, dtype=np.float64)[None, :, :],
                       axis=-1)
    return d < float(threshold)


def build_pair(bench, ref_frames_dir, qry_frames_dir, out_dir, pair_name,
               thresholds=THRESHOLDS, frame_index=None):
    """Write one ground-truth file per threshold for a single pair.

    `frame_index` is the (traverse -> recarray) mapping from `frames.py`. When
    given, its UTM columns must equal the ones the bench reads out of the
    filenames; that is what makes the exported frame index trustworthy rather
    than merely plausible.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for threshold in thresholds:
        dataset = bench.TestDataset(str(ref_frames_dir), str(qry_frames_dir),
                                    positive_dist_threshold=threshold,
                                    image_size=None, use_labels=True)
        by_bench = positives_to_matrix(dataset.get_positives(),
                                       dataset.num_database)

        # the independent version, from the frame index when we have one
        if frame_index is not None:
            ref_utms, qry_utms = frame_index["ref"], frame_index["qry"]
            if not np.allclose(ref_utms, dataset.database_utms, atol=1e-6):
                raise AssertionError(
                    f"{pair_name}: the frame index's reference UTM disagrees "
                    "with the coordinates the bench reads from the filenames")
            if not np.allclose(qry_utms, dataset.queries_utms, atol=1e-6):
                raise AssertionError(
                    f"{pair_name}: the frame index's query UTM disagrees with "
                    "the coordinates the bench reads from the filenames")
        else:
            ref_utms, qry_utms = dataset.database_utms, dataset.queries_utms

        independent = positives_from_utm(qry_utms, ref_utms, threshold)
        if not np.array_equal(by_bench, independent):
            n = int((by_bench != independent).sum())
            raise AssertionError(
                f"{pair_name} @{threshold}m: the bench's get_positives() and a "
                f"distance threshold over the frame index disagree on {n} "
                "cells. Do not score anything until this is understood.")

        no_pos = int((~by_bench.any(axis=1)).sum())
        path = out_dir / f"{pair_name}_{threshold}m.npz"
        np.savez_compressed(
            path, positives=by_bench,
            ref_utms=np.asarray(ref_utms, dtype=np.float64),
            qry_utms=np.asarray(qry_utms, dtype=np.float64))
        record = {
            "pair": pair_name, "threshold_m": int(threshold),
            "file": path.name, "sha256": hash_file(path),
            "n_query": int(by_bench.shape[0]), "n_ref": int(by_bench.shape[1]),
            "no_pos": no_pos,
            "positives_per_query_mean": float(by_bench.sum(axis=1).mean()),
            "reference_frames": str(ref_frames_dir),
            "query_frames": str(qry_frames_dir),
            "cross_checked_against_frame_index": frame_index is not None,
        }
        if threshold == PROTOCOL_THRESHOLD and no_pos:
            record["warning"] = (
                f"{no_pos} queries have no positive at {PROTOCOL_THRESHOLD} m; "
                "the bench's two R@1 denominators no longer coincide")
        records.append(record)
    return records


def write_manifest(out_dir, records, extra=None):
    out_dir = Path(out_dir)
    manifest = {"thresholds_m": list(THRESHOLDS),
                "protocol_threshold_m": PROTOCOL_THRESHOLD,
                "pairs": records}
    if extra:
        manifest.update(extra)
    path = out_dir / "ground_truth_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def load(path):
    """Returns (positives, metadata) and the file's own checksum."""
    path = Path(path)
    with np.load(path) as data:
        positives = data["positives"]
        ref_utms = data["ref_utms"]
        qry_utms = data["qry_utms"]
    return positives, {"file": path.name, "sha256": hash_file(path),
                       "ref_utms": ref_utms, "qry_utms": qry_utms}
