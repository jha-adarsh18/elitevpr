"""The evaluation layer: cache identity, ground-truth convention, assertions.

Most of this runs without the benchmark, against a stub that stands in for the
two imported callables. That is deliberate: the assertions are what make the
layer trustworthy, so they must be tested on failing inputs, and a real bench
cannot be made to fail on demand. The tests that need the real checkout skip
unless `ELITEVPR_BENCH` is set.
"""

import json
import os

import numpy as np
import pytest

from elitevpr.eval import cache, frames, ground_truth, protocol, report, scorer

BENCH = os.environ.get("ELITEVPR_BENCH")
needs_bench = pytest.mark.skipif(not BENCH,
                                 reason="ELITEVPR_BENCH is not set")


# --------------------------------------------------------------- the stub ---

class StubBench:
    """`seq_match_row` as the identity, and a truthful `plot_simmat`.

    At L=1 the scorer uses the raw matrix, so the identity matcher makes the
    whole path exercisable; `plot_simmat_with_matches` reproduces the bench's
    contract (argmax per row, with an `is_relevant` flag) so the R@1
    cross-check is a real check rather than a tautology.
    """

    seq_match_type = "modified"
    root = "stub"

    def __init__(self, r1_offset=0.0):
        self.r1_offset = r1_offset

    def seq_match_row(self, matrix, row, seq_len, seq_match_type=None):
        return matrix[row] * seq_len          # scaled; argsort is unchanged

    def plot_simmat_with_matches(self, matrix, positives_per_query, save_path,
                                 patch_num="", seq_len=1, seq_match_type=None):
        matches = []
        for i in range(matrix.shape[0]):
            best = int(np.argmax(matrix[i]))
            matches.append((i, best, float(matrix[i][best]),
                            int(best in positives_per_query[i])))
        if self.r1_offset:                    # pretend the bench disagrees
            matches[0] = (0, 0, 0.0, 1 - matches[0][3])
        return matrix, matches


def _simple_problem(n_query=6, n_ref=5):
    rng = np.random.default_rng(0)
    matrix = rng.random((n_query, n_ref))
    positives = np.zeros((n_query, n_ref), dtype=bool)
    for i in range(n_query):
        positives[i, int(np.argmax(matrix[i]))] = True      # R@1 = 100
    return matrix, positives


# ---------------------------------------------------------------- caching ---

def test_cache_key_is_not_determined_by_the_method_name():
    """The bug this layer exists to prevent: the bench keys on method only, so
    two checkpoints of one method shared a cached matrix."""
    a, _ = cache.cache_key("elitevpr", "cfg", "ckpt_A", 1.0, "s1__night")
    b, _ = cache.cache_key("elitevpr", "cfg", "ckpt_B", 1.0, "s1__night")
    assert a != b


@pytest.mark.parametrize("field,value", [("method", "other"),
                                         ("config_hash", "other"),
                                         ("checkpoint_hash", "other"),
                                         ("window", 0.05),
                                         ("pair", "s1__morning")])
def test_every_key_component_changes_the_key(field, value):
    base = dict(method="elitevpr", config_hash="c", checkpoint_hash="k",
                window=1.0, pair="s1__night")
    changed = dict(base, **{field: value})
    assert cache.cache_key(**base)[0] != cache.cache_key(**changed)[0]


def test_config_hash_ignores_key_order():
    assert (cache.hash_config({"a": 1, "b": 2})
            == cache.hash_config({"b": 2, "a": 1}))


def test_cache_roundtrip_and_single_computation(tmp_path):
    store = cache.MatrixCache(tmp_path)
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    calls = []

    def compute():
        calls.append(1)
        return matrix

    got, meta, hit = store.get_or_compute("elitevpr", "c", "k", 1.0, "p",
                                          compute)
    assert not hit and len(calls) == 1 and np.array_equal(got, matrix)
    assert meta["checkpoint_hash"] == "k" and meta["shape"] == [3, 4]

    got, meta, hit = store.get_or_compute("elitevpr", "c", "k", 1.0, "p",
                                          compute)
    assert hit and len(calls) == 1 and np.array_equal(got, matrix)


# ----------------------------------------------------------- ground truth ---

def test_the_threshold_is_strict_like_the_bench():
    """`distances < thresh` (test_dataset.py:66): 25.0 m is NOT a positive."""
    ref = np.array([[0.0, 0.0]])
    qry = np.array([[25.0, 0.0], [24.999, 0.0]])
    got = ground_truth.positives_from_utm(qry, ref, 25)
    assert got.tolist() == [[False], [True]]


def test_positives_to_matrix_matches_the_list_form():
    matrix = ground_truth.positives_to_matrix([[0, 2], [], [1]], n_ref=3)
    assert matrix.dtype == bool
    assert matrix.tolist() == [[True, False, True],
                               [False, False, False],
                               [False, True, False]]


# ----------------------------------------------------------------- scorer ---

def test_scorer_reproduces_recall_at_k_and_averages_the_right_lengths():
    matrix, positives = _simple_problem()
    out = scorer.score_matrix(StubBench(), matrix, positives)
    for seq_len in ("1", "10", "20", "30"):
        cell = out["by_seq_len"][seq_len]["r_at_k"]
        assert cell["1"] == 100.0                 # by construction
        assert cell["1"] <= cell["5"] <= cell["10"]
    assert out["ar_at_k"]["1"] == pytest.approx(
        sum(out["by_seq_len"][L]["r_at_k"]["1"] for L in ("10", "20", "30")) / 3)


def test_scorer_refuses_a_matrix_that_does_not_match_the_ground_truth():
    matrix, positives = _simple_problem()
    with pytest.raises(AssertionError, match="not built from these frames"):
        scorer.score_matrix(StubBench(), matrix[:, :4], positives)


def test_scorer_refuses_when_its_r_at_1_differs_from_the_bench():
    matrix, positives = _simple_problem()
    with pytest.raises(AssertionError, match="not reproducing the benchmark"):
        scorer.score_matrix(StubBench(r1_offset=1.0), matrix, positives)


def test_scorer_refuses_a_ground_truth_the_manifest_does_not_pin():
    matrix, positives = _simple_problem()
    manifest = {"pairs": [{"file": "a_25m.npz", "sha256": "expected",
                           "threshold_m": 25, "no_pos": 0, "pair": "a"}]}
    with pytest.raises(AssertionError, match="does not match the manifest"):
        scorer.score_matrix(StubBench(), matrix, positives,
                            gt_record={"file": "a_25m.npz", "sha256": "other"},
                            manifest=manifest)
    with pytest.raises(AssertionError, match="not in the ground-truth manifest"):
        scorer.score_matrix(StubBench(), matrix, positives,
                            gt_record={"file": "elsewhere.npz", "sha256": "x"},
                            manifest=manifest)


def test_scorer_refuses_a_pair_with_queries_that_have_no_positive():
    """At 25 m the bench's two R@1 denominators coincide only while no_pos=0."""
    matrix, positives = _simple_problem()
    manifest = {"pairs": [{"file": "a_25m.npz", "sha256": "s",
                           "threshold_m": 25, "no_pos": 3, "pair": "a"}]}
    with pytest.raises(AssertionError, match="no_pos=3"):
        scorer.score_matrix(StubBench(), matrix, positives,
                            gt_record={"file": "a_25m.npz", "sha256": "s"},
                            manifest=manifest)


def test_recall_ties_break_the_way_argmax_does():
    """A stable sort is what keeps the R@1 column comparable with the bench."""
    matrix = np.array([[1.0, 1.0, 0.0]])          # columns 0 and 1 tie
    positives = np.array([[True, False, False]])
    every, _valid = scorer.recalls(matrix, positives, ks=(1,))
    assert every[1] == 100.0                       # argmax picks index 0


# --------------------------------------------------------------- protocol ---

def test_pairs_are_the_benchs_own_experiment_pairs():
    """testing.py:160 -- changing these silently changes every column."""
    assert protocol.PAIRS["brisbane"] == [
        ("sunset1", "night"), ("sunset1", "morning"), ("sunset1", "sunrise"),
        ("sunset1", "sunset2"), ("sunset1", "daytime")]
    assert protocol.PAIRS["nsavp"] == [
        ("R0_FA0", "R0_FS0"), ("R0_FA0", "R0_FN0"), ("R0_FN0", "R0_FS0"),
        ("R0_RA0", "R0_RN0"), ("R0_RS0", "R0_RN0")]
    # every pair used by a column must be a pair we built a ground truth for
    for dataset, columns in protocol.COLUMNS.items():
        for pairs in columns.values():
            for pair in pairs:
                assert pair in protocol.PAIRS[dataset]


def test_bench_matrix_name_matches_the_benchs_own_filename():
    assert protocol.bench_matrix_name("sunset1", "night") == (
        "sunset1_vs_night_elitevpr_l2_reconstruct_eliteHistogram_1.0_frame.npy")


# ----------------------------------------------------------------- frames ---

def test_frame_index_csv_roundtrips(tmp_path):
    index = np.array([(0, 1.5, 2.5), (1, 3.5, 4.5)],
                     dtype=[("bin_index", np.int64), ("utm_east", np.float64),
                            ("utm_north", np.float64)])
    path = frames.write_traverse(tmp_path, "night", index)
    back = frames.load_index(path)
    assert back["bin_index"].tolist() == [0, 1]
    assert np.allclose(frames.utms(back), frames.utms(index))


# ----------------------------------------------------------------- report ---

def test_tables_are_built_from_result_jsons_only(tmp_path):
    for reference, query in protocol.PAIRS["brisbane"]:
        blob = {"run": "rkd_s42_best", "dataset": "brisbane",
                "pair": protocol.pair_name(reference, query),
                "config": {"arm": "rkd"},
                "scores": {"ar_at_k": {"1": 50.0, "5": 60.0, "10": 70.0}}}
        (tmp_path / f"x_{query}.json").write_text(json.dumps(blob))
    runs = report.load_results(tmp_path)
    rows = report.main_table(runs, "brisbane")
    assert rows[0]["day_day@1"] == 50.0 and rows[0]["day_night@10"] == 70.0
    assert "rkd_s42_best" in report.render(rows, "t")


# ------------------------------------------------------- the real checkout ---

@needs_bench
def test_the_pinned_bench_verifies_and_imports():
    from elitevpr.eval import bench as bench_mod

    loaded = bench_mod.load(BENCH)
    assert callable(loaded.seq_match_row)
    assert callable(loaded.plot_simmat_with_matches)
    seen = bench_mod.verify(loaded.root)
    assert seen == bench_mod.PINNED_FILES


@needs_bench
def test_a_tampered_bench_is_refused(tmp_path):
    from elitevpr.eval import bench as bench_mod

    target = tmp_path / "vpr_methods_evaluation"
    target.mkdir(parents=True)
    (target / "main.py").write_text("# not the pinned file\n")
    (target / "test_dataset.py").write_text("# nor this one\n")
    with pytest.raises(SystemExit, match="not the pinned one"):
        bench_mod.load(str(tmp_path))


# --------------------------------------------------------------- eventgem ---

def test_eventgem_matrix_orientation_is_transposed_to_ours():
    """Theirs is (references, queries) -- feature_extraction.py:269-270 does
    `query_feats @ ref_feats.t()` then `.T`. Ours is (queries, references)."""
    from elitevpr.eval import eventgem

    theirs = np.arange(6, dtype=np.float32).reshape(3, 2)      # 3 refs, 2 qrys
    ours = eventgem.to_our_orientation(theirs)
    assert ours.shape == (2, 3)
    assert np.array_equal(ours, theirs.T)


def test_eventgem_under_our_protocol_refuses_a_foreign_frame_set(tmp_path):
    """Their 50 ms traverse has ~20x our frames; the matrix is a different
    object and must not be quietly resampled onto our bins."""
    from elitevpr.eval import eventgem, ground_truth

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    positives = np.zeros((6, 5), dtype=bool)
    positives[np.arange(6), np.arange(6) % 5] = True
    np.savez_compressed(gt_dir / "sunset1__night_25m.npz", positives=positives,
                        ref_utms=np.zeros((5, 2)), qry_utms=np.zeros((6, 2)))
    theirs = np.zeros((50, 60), dtype=np.float32)              # their bins
    with pytest.raises(AssertionError, match="--dt-ms 1000"):
        eventgem.score_under_our_protocol(
            StubBench(), theirs, "brisbane", "sunset1", "night", gt_dir,
            manifest=None)


def test_eventgem_under_our_protocol_scores_a_matching_frame_set(tmp_path):
    """On our bins it is just another matrix: our ground truth, our matcher."""
    from elitevpr.eval import eventgem

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    matrix, positives = _simple_problem(n_query=6, n_ref=5)
    np.savez_compressed(gt_dir / "sunset1__night_25m.npz", positives=positives,
                        ref_utms=np.zeros((5, 2)), qry_utms=np.zeros((6, 2)))
    scores, record, stage = eventgem.score_under_our_protocol(
        StubBench(), matrix.T, "brisbane", "sunset1", "night", gt_dir,
        manifest=None)
    assert scores["by_seq_len"]["1"]["r_at_k"]["1"] == 100.0
    assert "ar_at_k" in scores and stage == "original"
    assert record["sha256"]


def test_eventgem_native_settings_are_the_published_ones():
    """A changed default silently moves their operating point."""
    from elitevpr.eval import eventgem

    assert eventgem.NATIVE_SETTINGS["dt_ms"] == 50
    assert eventgem.NATIVE_SETTINGS["top_k"] == 50
    assert eventgem.NATIVE_SETTINGS["ransac_thresh"] == 5.0
    assert eventgem.DATASET_NAME["brisbane"] == "brisbane_event"


def test_eventgem_requires_a_checkout_and_says_how():
    from elitevpr.eval import eventgem

    with pytest.raises(SystemExit, match="recurse-submodules"):
        eventgem.repo_root(None if not os.environ.get(eventgem.ENV_VAR) else "")
