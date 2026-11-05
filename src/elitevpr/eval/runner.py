"""Scoring a run end to end, and the JSON that records it.

One JSON per (method, config, seed, pair). It carries the numbers and, beside
them, everything needed to say which artefacts produced them: the ground
truth's checksum, the checkpoint's hash, the config hash and this repository's
commit. A result that cannot name its inputs is not a result.
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ground_truth as gt_mod
from .cache import hash_config, hash_file
from .protocol import PAIRS, pair_name
from .scorer import score_matrix


def repo_commit(repo_root=None):
    """HEAD of this repository, read from .git without invoking git."""
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text().strip()
    if text.startswith("ref: "):
        ref = root / ".git" / text[5:]
        if ref.is_file():
            return ref.read_text().strip()
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + text[5:]):
                    return line.split()[0]
        return None
    return text


def score_pair(bench, matrix, dataset, reference, query, gt_dir, manifest,
               threshold=25):
    """Load the pinned ground truth for one pair and score a matrix with it."""
    name = pair_name(reference, query)
    path = Path(gt_dir) / f"{name}_{threshold}m.npz"
    positives, record = gt_mod.load(path)
    record = {"file": record["file"], "sha256": record["sha256"]}
    scores = score_matrix(bench, matrix, positives, gt_record=record,
                          manifest=manifest)
    return scores, record


def write_result(out_dir, run, dataset, reference, query, scores, gt_record,
                 config=None, checkpoint=None, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    blob = {
        "run": run,
        "dataset": dataset,
        "pair": pair_name(reference, query),
        "reference": reference,
        "query": query,
        "method": config.get("method", "elitevpr"),
        "seed": config.get("seed"),
        "config": config,
        "config_hash": hash_config(config),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": (hash_file(checkpoint)
                              if checkpoint and Path(checkpoint).is_file()
                              else None),
        "ground_truth": gt_record,
        "commit": repo_commit(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scores": scores,
    }
    if extra:
        blob.update(extra)
    path = out_dir / f"{run}__{dataset}__{pair_name(reference, query)}.json"
    path.write_text(json.dumps(blob, indent=1, sort_keys=True))
    return path, blob


def score_run_from_matrices(bench, run, dataset, matrix_for_pair, gt_dir,
                            manifest, out_dir, config=None, checkpoint=None,
                            threshold=25):
    """Score every pair of one dataset for one run.

    `matrix_for_pair(reference, query)` returns the similarity matrix, so the
    caller decides whether it comes from a cache, a file the bench wrote, or a
    model that has just run.
    """
    written = []
    for reference, query in PAIRS[dataset]:
        matrix = matrix_for_pair(reference, query)
        if matrix is None:
            continue
        scores, gt_record = score_pair(bench, matrix, dataset, reference,
                                       query, gt_dir, manifest, threshold)
        path, blob = write_result(out_dir, run, dataset, reference, query,
                                  scores, gt_record, config, checkpoint)
        written.append((path, blob))
    return written
