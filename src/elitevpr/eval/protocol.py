"""The protocol, in one place: traverses, pairs, and how columns are formed.

Taken from the benchmark's own indices (`testing.py:157`, the `seq_id_to_name`
map) and its own experiment pairs (`testing.py:160`). Written down here so a
table can never be built from a different set of pairs than the ground truth
was built for.

The column definitions match arXiv 2509.01968 and our own recorded runs:
Brisbane uses sunset1 as the reference against the other five traverses, and
AR@K is the mean over sequence lengths 10, 20 and 30.

NSAVP R1 is deliberately absent: those recordings are not on disk yet. Adding
them is a new entry in TRAVERSES and PAIRS plus a ground-truth rebuild; no
scorer change.
"""

SEQ_INDEX = {0: "night", 1: "morning", 2: "sunrise", 3: "sunset1",
             4: "sunset2", 5: "daytime", 6: "R0_FA0", 7: "R0_FS0",
             8: "R0_FN0", 9: "R0_RA0", 10: "R0_RS0", 11: "R0_RN0"}

TRAVERSES = {
    "brisbane": ["sunset1", "sunset2", "morning", "sunrise", "daytime",
                 "night"],
    "nsavp": ["R0_FA0", "R0_FS0", "R0_FN0", "R0_RA0", "R0_RS0", "R0_RN0"],
}

# (reference, query), exactly the bench's experiment_pairs.
PAIRS = {
    "brisbane": [("sunset1", "night"), ("sunset1", "morning"),
                 ("sunset1", "sunrise"), ("sunset1", "sunset2"),
                 ("sunset1", "daytime")],
    "nsavp": [("R0_FA0", "R0_FS0"), ("R0_FA0", "R0_FN0"),
              ("R0_FN0", "R0_FS0"), ("R0_RA0", "R0_RN0"),
              ("R0_RS0", "R0_RN0")],
}

# How the two headline columns are formed. NSAVP's split follows our recorded
# convention (results_all.md): day-day is the single FA0->FS0 pair, and
# day-night averages the four pairs that involve a night traverse.
COLUMNS = {
    "brisbane": {
        "day_day": [("sunset1", "morning"), ("sunset1", "sunrise"),
                    ("sunset1", "sunset2"), ("sunset1", "daytime")],
        "day_night": [("sunset1", "night")],
    },
    "nsavp": {
        "day_day": [("R0_FA0", "R0_FS0")],
        "day_night": [("R0_FA0", "R0_FN0"), ("R0_FN0", "R0_FS0"),
                      ("R0_RA0", "R0_RN0"), ("R0_RS0", "R0_RN0")],
    },
}

DATASET_TYPE = {"brisbane": "Brisbane", "nsavp": "NSAVP"}
REPRESENTATION = "eliteHistogram"
DEFAULT_WINDOW = 1.0            # time_res, seconds


def pair_name(reference, query):
    return f"{reference}__{query}"


def frames_dir(data_root, representation=REPRESENTATION,
               window=DEFAULT_WINDOW):
    """`<data_root>/image_reconstructions/fixed_timebins_<window>/<rep>`.

    The bench's own layout (`load_and_save.make_paths`), so a frame set built
    by the bench is found without copying anything.
    """
    from pathlib import Path
    return (Path(data_root) / "image_reconstructions"
            / f"fixed_timebins_{window}" / representation)


def bench_matrix_name(reference, query, method="elitevpr",
                      representation=REPRESENTATION,
                      window=DEFAULT_WINDOW, patch_or_frame="frame"):
    """The filename the bench caches a matrix under (`main.py:651`).

    Needed only to READ matrices the bench already produced -- this layer's own
    cache is content-addressed instead; see cache.py for why.
    """
    return (f"{reference}_vs_{query}_{method}_l2_reconstruct_"
            f"{representation}_{window}_{patch_or_frame}.npy")
