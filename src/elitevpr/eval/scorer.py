"""Matrix in, numbers out. The only place a recall is computed.

Input is a similarity matrix (higher = more similar, rows = queries) and a
ground-truth file. Output is R@1/5/10 at L = 1, 10, 20, 30 and AR@K as the mean
over L = {10, 20, 30}.

The matcher is the bench's `seq_match_row`, imported. The R@1 convention is the
bench's too, and rather than trust that this module reproduces it, every cell
is checked against it: `plot_simmat_with_matches` is the function whose
`argmax` the bench reports, so it is called on the same matrix and the two R@1
values must agree. That check needs no model and no frames -- only the matrix --
which is why it can run on every cell of every arm rather than on a sample.

Four assertions, all fatal:
  1. the matrix shape matches the ground truth (hence the frame index);
  2. the ground-truth file's checksum matches the manifest;
  3. no_pos = 0 at 25 m, so the bench's two R@1 denominators coincide;
  4. our R@1 equals the bench's own R@1, for every L.
"""

import numpy as np

SEQ_LENS = (1, 10, 20, 30)
KS = (1, 5, 10)
AR_LENS = (10, 20, 30)
R1_TOLERANCE = 1e-9      # percentage points; observed worst case 1.4e-14


def seq_matched(bench, matrix, seq_len):
    """The bench's own per-row matcher, stacked. L=1 is the raw matrix."""
    if seq_len == 1:
        return matrix
    return np.stack([
        bench.seq_match_row(matrix, i, seq_len,
                            seq_match_type=bench.seq_match_type) / seq_len
        for i in range(matrix.shape[0])])


def recalls(scores, positives, ks=KS):
    """R@K over every query (the bench's denominator, main.py:686).

    A stable sort is used so `order[0]` is exactly what `np.argmax` picks when
    two references tie, which is what makes the R@1 column comparable with the
    bench cell for cell.
    """
    top = max(ks)
    hits = {k: 0 for k in ks}
    hits_valid = {k: 0 for k in ks}
    n_valid = 0
    for i in range(scores.shape[0]):
        order = np.argsort(-scores[i], kind="stable")[:top]
        truth = positives[i]
        has_positive = bool(truth.any())
        n_valid += has_positive
        for k in ks:
            hit = int(truth[order[:k]].any())
            hits[k] += hit
            if has_positive:
                hits_valid[k] += hit
    n = scores.shape[0]
    return ({k: 100.0 * v / n for k, v in hits.items()},
            {k: 100.0 * v / n_valid if n_valid else 0.0
             for k, v in hits_valid.items()})


def bench_recall_at_1(bench, matrix, positives, seq_len):
    """The bench's own R@1 for this matrix: its argmax, its denominator."""
    per_query = [np.flatnonzero(row).tolist() for row in positives]
    _, matches = bench.plot_simmat_with_matches(
        matrix, per_query, None, patch_num="frame", seq_len=seq_len,
        seq_match_type=bench.seq_match_type)
    correct = sum(is_relevant for _q, _best, _score, is_relevant in matches)
    return 100.0 * correct / len(matches) if matches else 0.0


def score_matrix(bench, matrix, positives, gt_record=None, manifest=None,
                 seq_lens=SEQ_LENS, ks=KS, check_against_bench=True):
    """Every cell for one (method, pair), with the four assertions applied."""
    matrix = np.asarray(matrix)
    positives = np.asarray(positives, dtype=bool)

    if matrix.shape != positives.shape:
        raise AssertionError(
            f"similarity matrix is {matrix.shape} but the ground truth is "
            f"{positives.shape}: the matrix was not built from these frames")

    if gt_record is not None and manifest is not None:
        pinned = _manifest_entry(manifest, gt_record)
        if pinned["sha256"] != gt_record["sha256"]:
            raise AssertionError(
                f"ground truth {gt_record['file']} does not match the "
                f"manifest: expected {pinned['sha256']}, found "
                f"{gt_record['sha256']}")
        if int(pinned["threshold_m"]) == 25 and int(pinned["no_pos"]):
            raise AssertionError(
                f"{pinned['pair']}: no_pos={pinned['no_pos']} at 25 m, so the "
                "bench's two R@1 denominators differ and the cell is ambiguous")

    out = {"n_query": int(matrix.shape[0]), "n_ref": int(matrix.shape[1]),
           "by_seq_len": {}}
    for seq_len in seq_lens:
        scores = seq_matched(bench, matrix, seq_len)
        every, valid = recalls(scores, positives, ks)
        cell = {"r_at_k": {str(k): every[k] for k in ks},
                "r_at_k_valid_only": {str(k): valid[k] for k in ks}}
        if check_against_bench:
            reference = bench_recall_at_1(bench, matrix, positives, seq_len)
            delta = abs(every[1] - reference)
            cell["bench_r_at_1"] = reference
            cell["r_at_1_delta_vs_bench"] = delta
            if delta > R1_TOLERANCE:
                raise AssertionError(
                    f"L={seq_len}: our R@1 {every[1]:.10f} != the bench's "
                    f"{reference:.10f} (delta {delta:.2e}). The scorer is not "
                    "reproducing the benchmark; no number from this run is "
                    "usable.")
        out["by_seq_len"][str(seq_len)] = cell

    if all(L in seq_lens for L in AR_LENS):
        out["ar_at_k"] = {
            str(k): sum(out["by_seq_len"][str(L)]["r_at_k"][str(k)]
                        for L in AR_LENS) / len(AR_LENS) for k in ks}
    return out


def _manifest_entry(manifest, gt_record):
    for entry in manifest["pairs"]:
        if entry["file"] == gt_record["file"]:
            return entry
    raise AssertionError(
        f"{gt_record['file']} is not in the ground-truth manifest: it was not "
        "produced by this layer, so nothing pins what it contains")
