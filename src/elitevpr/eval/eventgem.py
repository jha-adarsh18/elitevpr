"""EventGeM (arXiv 2603.05807) as a comparable row -- two modes.

    mode="native"   their frames, their ground truth, their metric.
                    Reproduces the published numbers. Nothing of ours is used.
    mode="ours"     their similarity matrix, OUR ground truth and OUR matcher,
                    so the row can sit beside every other method in the table.

Their pipeline is NOT imported into our environment and their repository is not
modified. It ships its own pixi environment and needs a SuperEvent trunk plus
an optional Depth-AnyEvent stage; resolving those against our pinned torch
would put the reported numbers at risk for no gain. So this adapter does two
things only: it drives their CLI, and it reads the artefacts it leaves behind.

Facts about their formats, read from their source (commit of 2026-11-05):

* the global-retrieval matrix is `query_feats @ ref_feats.t()` followed by
  `.T` (feature_extraction.py:269-270), so it is stored as
  **(references, queries)** -- the transpose of ours. `training/common.py:326`
  states the same convention in a comment.
* it is written with `torch.save` to
  `{dataset}_{reference}_{query}_similarity.pt` (feature_extraction.py:301),
  and the README also documents
  `<data-root>/<dataset>/<ref>-<qry>-similarity/{original,reranked}_sim_mat.npy`
  for the shortlist and the re-ranked result.
* their ground truth is a precomputed boolean `<ref>_<query>_GT.npy`
  (inference.py:53), also (references, queries), fetched from the
  `AdamHines/EventGeM` HF dataset -- NOT a 25 m search over GPS.
* their metric is `recallAtK` from the VPR_Tutorial submodule, called on a
  DISTANCE matrix: `recallAtK((1 - original), gt, k)` (analysis.py:45). There
  is no sequence matching anywhere -- single frame only.
* `analysis.recall` nearest-neighbour RESIZES the ground truth when its shape
  does not match the similarity matrix (analysis.py:39). We never do that; a
  shape mismatch raises here, because a stretched ground truth is not one.

The hard part, stated plainly: **frame correspondence.** Their default window
is `--dt-ms 50` and ours is 1000, so a traverse is ~12,800 of their frames
against 639 of ours, and the two matrices are not the same object. Mode "ours"
therefore accepts a matrix only when it is already on our bins -- either
because their extractor was run at `--dt-ms 1000`, or because a timestamp
mapping from the frame index is supplied. Anything else is refused rather than
silently resampled.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .protocol import PAIRS, pair_name
from .scorer import KS, score_matrix

ENV_VAR = "ELITEVPR_EVENTGEM"
UPSTREAM = "https://github.com/AdamDHines/Event-GeM"

# Their published settings (README "List of arguments"). Changing any of these
# means the row is no longer their method at their operating point.
NATIVE_SETTINGS = {
    "dt_ms": 50,
    "max_window_ms": None,      # defaults to dt_ms: each frame integrates fully
    "top_k": 50,
    "match_filter": "mutual",
    "match_ratio": 0.8,
    "ransac_thresh": 5.0,
    "inlier_weight": 0.05,
}
# Their dataset names for the two corpora we score.
DATASET_NAME = {"brisbane": "brisbane_event", "nsavp": "nsavp"}
# Their traverse names are ours for Brisbane; NSAVP needs the R0_ prefix kept.
STAGES = ("original", "reranked")


def repo_root(explicit=None):
    candidate = explicit or os.environ.get(ENV_VAR)
    if not candidate:
        raise SystemExit(
            f"set {ENV_VAR} to a checkout of {UPSTREAM} (cloned WITH "
            "--recurse-submodules: it refuses to start without the superevent "
            "and eventlab submodules)")
    root = Path(candidate).expanduser().resolve()
    if not (root / "eventgem" / "inference.py").is_file():
        raise SystemExit(f"{root} is not an Event-GeM checkout: clone {UPSTREAM}")
    missing = [name for name in ("superevent", "eventlab", "vprtutorial")
               if not any((root / "eventgem" / "external" / name).glob("*"))]
    if missing:
        raise SystemExit(
            f"{root}: submodules {missing} are empty. Their weights and their "
            "recall metric live there; run `git submodule update --init "
            "--recursive` in that checkout.")
    return root


# ------------------------------------------------------------ their protocol

def run_native(root, dataset, reference, query, data_root, settings=None,
               extra_args=(), timeout=None):
    """Drive their CLI once. Returns the completed process.

    Their own caching applies: a second call reuses the features on disk unless
    `--rerun-features` is passed.
    """
    settings = {**NATIVE_SETTINGS, **(settings or {})}
    argv = ["pixi", "run", "eventgem",
            "--dataset", DATASET_NAME.get(dataset, dataset),
            "--reference", reference, "--query", query,
            "--data-root", str(data_root),
            "--dt-ms", str(settings["dt_ms"]),
            "--top-k", str(settings["top_k"]),
            "--match-filter", str(settings["match_filter"]),
            "--ransac-thresh", str(settings["ransac_thresh"]),
            "--inlier-weight", str(settings["inlier_weight"])]
    if settings.get("max_window_ms") is not None:
        argv += ["--max-window-ms", str(settings["max_window_ms"])]
    argv += list(extra_args)
    return subprocess.run(argv, cwd=str(root), capture_output=True, text=True,
                          timeout=timeout)


def native_recall(root, similarity_rq, ground_truth_rq, ks=KS):
    """Their metric, imported from their submodule -- never reimplemented.

    `similarity_rq` and `ground_truth_rq` are both (references, queries), their
    orientation. The similarity is converted to a distance exactly as
    analysis.py:45 does.
    """
    import sys

    vprtutorial = root / "eventgem" / "external" / "vprtutorial"
    if str(vprtutorial) not in sys.path:
        sys.path.insert(0, str(vprtutorial))
    from evaluation.metrics import recallAtK            # their function

    similarity = np.asarray(similarity_rq, dtype=np.float64)
    truth = np.asarray(ground_truth_rq)
    if similarity.shape != truth.shape:
        raise AssertionError(
            f"their similarity matrix is {similarity.shape} and their ground "
            f"truth is {truth.shape}. Their analysis.py:39 would resize the "
            "ground truth here; we do not, because a nearest-neighbour "
            "stretched ground truth is not a ground truth.")
    return {str(k): float(recallAtK(1 - similarity, truth, int(k)))
            for k in ks}


# -------------------------------------------------------------- our protocol

def to_our_orientation(similarity_rq):
    """(references, queries) -> (queries, references), which is ours."""
    return np.asarray(similarity_rq).T


def load_matrix(path):
    """Their matrix, from either the `.pt` or the `.npy` they write."""
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        return torch.load(path, map_location="cpu").float().numpy()
    return np.load(path)


def score_under_our_protocol(bench, similarity_rq, dataset, reference, query,
                             gt_dir, manifest, run="eventgem", stage="original",
                             frame_index=None, threshold=25):
    """Their matrix, our ground truth, our matcher, our L values.

    Refuses rather than resamples when their frame set is not ours. The error
    names the two shapes and the only two legitimate fixes, because a quietly
    interpolated matrix would produce a number that looks comparable and is not.
    """
    from . import ground_truth as gt_mod

    name = pair_name(reference, query)
    positives, record = gt_mod.load(Path(gt_dir) / f"{name}_{threshold}m.npz")
    matrix = to_our_orientation(similarity_rq)

    if matrix.shape != positives.shape:
        n_q, n_ref = positives.shape
        raise AssertionError(
            f"EventGeM's matrix for {name} is {matrix.shape} after transpose "
            f"but our ground truth is {(n_q, n_ref)}.\n"
            "Their default window is --dt-ms 50 and ours is 1000, so their "
            "traverse has about 20x our frames. Either re-run their extractor "
            "at --dt-ms 1000, or supply a timestamp mapping built from the "
            "frame index (frames.py's t_start/t_end) so each of our bins can "
            "be matched to one of theirs. Do NOT interpolate the matrix."
        )

    scores = score_matrix(bench, matrix, positives,
                          gt_record={"file": record["file"],
                                     "sha256": record["sha256"]},
                          manifest=manifest)
    return scores, {"file": record["file"], "sha256": record["sha256"]}, stage


def both_modes(bench, root, dataset, gt_dir, manifest, matrices, out_dir,
               ground_truths_rq=None, settings=None, run="eventgem"):
    """One JSON per pair carrying BOTH readings, side by side.

    `matrices[(reference, query, stage)]` supplies their (references, queries)
    matrix; `ground_truths_rq[(reference, query)]` their GT, when the native
    reading is wanted too. Keeping both in one file is deliberate: the two
    numbers are not comparable, and a reader should never meet one without the
    other.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for reference, query in PAIRS[dataset]:
        blob = {"run": run, "dataset": dataset,
                "pair": pair_name(reference, query),
                "reference": reference, "query": query,
                "method": "eventgem",
                "config": {"method": "eventgem", "mode": "both",
                           **{**NATIVE_SETTINGS, **(settings or {})}},
                "native": {}, "ours": {}}
        for stage in STAGES:
            matrix = matrices.get((reference, query, stage))
            if matrix is None:
                continue
            if ground_truths_rq and (reference, query) in ground_truths_rq:
                blob["native"][stage] = {
                    "recall_at_k": native_recall(
                        root, matrix, ground_truths_rq[(reference, query)]),
                    "ground_truth": "their <ref>_<qry>_GT.npy",
                    "sequence_matching": "none (single frame)",
                }
            scores, gt_record, _ = score_under_our_protocol(
                bench, matrix, dataset, reference, query, gt_dir, manifest,
                run=run, stage=stage)
            blob["ours"][stage] = {"scores": scores, "ground_truth": gt_record}
        path = out_dir / f"{run}__{dataset}__{blob['pair']}.json"
        path.write_text(json.dumps(blob, indent=1, sort_keys=True))
        written.append(path)
    return written
