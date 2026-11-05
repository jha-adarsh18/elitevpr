"""Item 6: the gate. Nothing may use this layer until this passes.

Re-scores the six arms (seed 42, `best` and `last`) from the similarity
matrices the 2026-11-04 run produced, through this layer's ground truth and
this layer's scorer, and compares every cell with that run.

Two comparisons, because the two references have different precision:

* against the per-pair JSONs of 2026-11-04 at **1e-6**. These are the files the
  table was rendered from, so this is the strict check the spec asks for.
* against `RESULTS_six_arms_RatK_2026-11-04.txt` at **5e-3**, which is half of
  that file's last printed digit. The txt is rendered with `%.2f`, so 1e-6
  against it is not a well-defined test; this checks it to the precision it
  actually carries.

Any mismatch is reported in full and the run stops with a non-zero exit.

The 12 single-config baselines are deliberately not included: their matrices
are not on this machine (they are in `workspace_recon.7z`), and the Nov-4 file
contains no baseline cell to compare against.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import bench as bench_mod
from . import frames as frames_mod
from . import ground_truth as gt_mod
from .protocol import (DEFAULT_WINDOW, PAIRS, REPRESENTATION, TRAVERSES,
                       bench_matrix_name, frames_dir, pair_name)
from .runner import score_run_from_matrices

ARMS = ["structural_kl_s42_best",
        "feature_cosine_s42_best", "feature_cosine_s42_last",
        "feature_mse_s42_best", "feature_mse_s42_last",
        "infonce_s42_best", "infonce_s42_last",
        "sp_kd_s42_best", "sp_kd_s42_last",
        "rkd_s42_best", "rkd_s42_last"]
JSON_TOLERANCE = 1e-6
TXT_TOLERANCE = 5e-3
SEQ_LENS = ("1", "10", "20", "30")
KS = ("1", "5", "10")


def parse_reference_txt(path):
    """Pull every per-cell R@K out of the Nov-4 'EVERY CELL' section."""
    cells = {}
    run = dataset = None
    header = re.compile(r"^(\S+)\s+\[(brisbane|nsavp)\]\s+weights")
    row = re.compile(r"^(\S+->\S+)\s+(\d+)\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s")
    for line in Path(path).read_text().splitlines():
        found = header.match(line)
        if found:
            run, dataset = found.group(1), found.group(2)
            continue
        if run is None:
            continue
        found = row.match(line)
        if found:
            pair = found.group(1).replace("->", "__")
            seq_len = found.group(2)
            for k, value in zip(KS, found.group(3, 4, 5)):
                cells[(run, dataset, pair, seq_len, k)] = float(value)
    return cells


def load_reference_jsons(directory):
    """The Nov-4 per-pair JSONs, flattened to the same cell keys."""
    cells = {}
    for path in sorted(Path(directory).glob("*.json")):
        blob = json.loads(path.read_text())
        if "pairs" not in blob or "tag" not in blob:
            continue
        run, dataset = blob["tag"], blob["dataset"]
        for pair, info in blob["pairs"].items():
            name = pair.replace("->", "__")
            for seq_len, cell in info["by_seq_len"].items():
                for k, value in cell["r_at_k"].items():
                    cells[(run, dataset, name, seq_len, k)] = float(value)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default=None, help="official bench checkout")
    ap.add_argument("--matrices", required=True,
                    help="<dir>/<dataset>/<run>/<bench matrix name>.npy")
    ap.add_argument("--brisbane", required=True, help="Brisbane data root")
    ap.add_argument("--nsavp", default=None, help="NSAVP data root")
    ap.add_argument("--checkpoints", default=None,
                    help="dir holding <arm>_s42/{best,last}.pth, for hashes")
    ap.add_argument("--out", required=True, help="where artefacts are written")
    ap.add_argument("--reference-json", required=True,
                    help="the 2026-11-04 per-pair JSON directory")
    ap.add_argument("--reference-txt", default=None,
                    help="RESULTS_six_arms_RatK_2026-11-04.txt")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = bench_mod.load(args.bench)
    print(f"bench       {bench.root}  (pinned files verified)")

    roots = {"brisbane": args.brisbane}
    if args.nsavp:
        roots["nsavp"] = args.nsavp

    # ---- item 1 (positions) and item 2, for every dataset in play ----------
    gt_dir = out / "ground_truth"
    index_dir = out / "frame_index"
    manifests, indices = {}, {}
    for dataset, root in roots.items():
        records, idx = frames_mod.export_positions(
            bench, root, TRAVERSES[dataset], index_dir / dataset,
            REPRESENTATION, DEFAULT_WINDOW)
        frames_mod.write_manifest(index_dir / dataset, records,
                                  {"dataset": dataset, "data_root": root})
        indices[dataset] = idx
        missing = [r["traverse"] for r in records if "error" in r]
        if missing:
            raise SystemExit(f"{dataset}: no frames for {missing}")
        print(f"frame index  {dataset}: " + ", ".join(
            f"{r['traverse']}={r['n_frames']}" for r in records))

        gt_records = []
        for reference, query in PAIRS[dataset]:
            base = frames_dir(root, REPRESENTATION, DEFAULT_WINDOW)
            gt_records += gt_mod.build_pair(
                bench, base / reference, base / query, gt_dir / dataset,
                pair_name(reference, query),
                frame_index={"ref": frames_mod.utms(idx[reference]),
                             "qry": frames_mod.utms(idx[query])})
        manifests[dataset] = json.loads(
            gt_mod.write_manifest(gt_dir / dataset, gt_records,
                                  {"dataset": dataset}).read_text())
        at25 = [r for r in gt_records if r["threshold_m"] == 25]
        print(f"ground truth {dataset}: {len(gt_records)} files, "
              f"no_pos at 25 m = {sum(r['no_pos'] for r in at25)}")

    # ---- items 3 and 5: score every arm from its matrices ------------------
    results_dir = out / "results"
    scored = 0
    for run in ARMS:
        arm = run.rsplit("_s42_", 1)[0]
        ckpt_name = run.rsplit("_", 1)[1]
        checkpoint = None
        if args.checkpoints:
            candidate = Path(args.checkpoints) / f"{arm}_s42" / f"{ckpt_name}.pth"
            checkpoint = candidate if candidate.is_file() else None
        for dataset in roots:
            directory = Path(args.matrices) / dataset / run
            if not directory.is_dir():
                print(f"  skip {run}/{dataset}: no matrices at {directory}")
                continue

            def matrix_for_pair(reference, query, _d=directory):
                path = _d / bench_matrix_name(reference, query)
                return np.load(path) if path.is_file() else None

            written = score_run_from_matrices(
                bench, run, dataset, matrix_for_pair, gt_dir / dataset,
                manifests[dataset], results_dir,
                config={"method": "elitevpr", "seed": 42, "arm": arm,
                        "checkpoint": ckpt_name, "window": DEFAULT_WINDOW,
                        "representation": REPRESENTATION},
                checkpoint=checkpoint)
            scored += len(written)
        print(f"scored      {run}")
    print(f"\n{scored} (run, dataset, pair) results written to {results_dir}")

    # ---- item 6: compare every cell ---------------------------------------
    ours = {}
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        for seq_len, cell in blob["scores"]["by_seq_len"].items():
            for k, value in cell["r_at_k"].items():
                ours[(blob["run"], blob["dataset"], blob["pair"], seq_len,
                      k)] = float(value)

    failures = []
    checked = {"json": 0, "txt": 0}
    reference = load_reference_jsons(args.reference_json)
    for key, value in sorted(reference.items()):
        if key not in ours:
            failures.append(f"  missing cell {key} (in the Nov-4 JSONs, not "
                            "produced here)")
            continue
        delta = abs(ours[key] - value)
        checked["json"] += 1
        if delta > JSON_TOLERANCE:
            failures.append(f"  {key}: ours {ours[key]:.10f} vs Nov-4 JSON "
                            f"{value:.10f}  delta {delta:.3e}")

    if args.reference_txt:
        for key, value in sorted(parse_reference_txt(args.reference_txt).items()):
            if key not in ours:
                continue
            delta = abs(ours[key] - value)
            checked["txt"] += 1
            if delta > TXT_TOLERANCE:
                failures.append(f"  {key}: ours {ours[key]:.4f} vs Nov-4 txt "
                                f"{value:.2f}  delta {delta:.3e}")

    print(f"\ncompared {checked['json']} cells against the Nov-4 JSONs at "
          f"{JSON_TOLERANCE:g}")
    print(f"compared {checked['txt']} cells against the Nov-4 txt at "
          f"{TXT_TOLERANCE:g} (its printed precision)")
    if failures:
        print(f"\nACCEPTANCE FAILED -- {len(failures)} mismatches:")
        print("\n".join(failures[:60]))
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more")
        return 1
    if not checked["json"]:
        print("\nACCEPTANCE FAILED -- no cells were compared")
        return 1
    print("\nACCEPTANCE PASSED -- every cell reproduces the 2026-11-04 run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
