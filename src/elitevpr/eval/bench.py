"""The one dependency: the official benchmark, imported and never modified.

Everything that decides a number -- the sequence matcher, the ground-truth
search, the R@1 convention -- is imported from
`github.com/theresejoseph/ensemble_event_vpr_bench`. This module locates that
checkout, verifies it is the pinned one, and hands back the three callables the
rest of the layer uses.

Why a hash and not a version string: the layer's entire value is that every
method is scored against ONE fixed ground truth. A silent edit to `main.py`
would move every number in every table at once and nothing would notice. The
two files below are hashed on every import, so that cannot happen quietly.

Verified 2026-11-05: these two files are byte-identical between the official
repo and the fork that produced every number up to now, so adopting the
official repo as the dependency changes no result. The fork's divergences are
all in the frame-BUILDING path (`load_and_save.py` writes `.jpg` for every
representation, `datasets/brisbane_events.py` reads a downsampled per-sequence
GPS file); those live here as adapters instead -- see `adapters.py`.
"""

import hashlib
import os
import sys
from pathlib import Path

# sha256 of the only two bench files this layer's numbers depend on.
PINNED_FILES = {
    "vpr_methods_evaluation/main.py":
        "023272928ff3c410fb8da8acc440b409c904893d2ea5094aff7e1bdf48b7943f",
    "vpr_methods_evaluation/test_dataset.py":
        "42936e9aa3afed6d5e2776a6e5d44b4fe761ced91d0cf1124b5996df7a3f6e8c",
}
UPSTREAM = "https://github.com/theresejoseph/ensemble_event_vpr_bench"
ENV_VAR = "ELITEVPR_BENCH"


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bench_root(explicit=None):
    """Where the official checkout lives: argument, then $ELITEVPR_BENCH."""
    candidate = explicit or os.environ.get(ENV_VAR)
    if not candidate:
        raise SystemExit(
            f"set {ENV_VAR} to a checkout of {UPSTREAM} (or pass --bench)")
    root = Path(candidate).expanduser().resolve()
    if not (root / "vpr_methods_evaluation" / "main.py").is_file():
        raise SystemExit(f"{root} is not a bench checkout: clone {UPSTREAM}")
    return root


def verify(root):
    """Hash the pinned files; raise unless every one matches.

    `strict=False` is deliberately not offered. A run against an unpinned
    bench produces numbers that cannot be compared with any other run, which
    is the exact failure this layer exists to prevent.
    """
    seen = {}
    wrong = []
    for rel, expected in PINNED_FILES.items():
        path = root / rel
        if not path.is_file():
            wrong.append(f"  {rel}: missing")
            continue
        got = sha256_file(path)
        seen[rel] = got
        if got != expected:
            wrong.append(f"  {rel}:\n    expected {expected}\n    found    {got}")
    if wrong:
        raise SystemExit(
            "the bench checkout is not the pinned one, so its numbers are not "
            "comparable with anything already recorded:\n" + "\n".join(wrong)
            + f"\n\nPinned to {UPSTREAM}. If the upgrade is intended, update "
              "PINNED_FILES and re-run the acceptance test before trusting "
              "any table."
        )
    return seen


# The bench is scripts, not a package: `main.py` reaches for its siblings by
# bare name (`import parse`, `from test_dataset import TestDataset`, lines 7
# and 17). Any of those names already importable from somewhere else wins, and
# this repository really does have a `tests/test_dataset.py` that pytest
# registers as `test_dataset` -- which silently handed the bench OUR module and
# failed with "cannot import name 'TestDataset'". So these names are loaded
# from the bench's own files for the duration of the import, then the previous
# occupants are put back.
_BARE_IMPORTS = ("test_dataset", "parse", "parser_config")


def _load_by_path(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(explicit=None):
    """Import the bench and return its callables, after verifying the checkout.

    Only these three names are ever used; nothing in this layer reimplements
    them.
    """
    root = bench_root(explicit)
    verify(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    displaced = {name: sys.modules.pop(name, None) for name in _BARE_IMPORTS}
    try:
        bench_test_dataset = _load_by_path(
            "test_dataset", root / "vpr_methods_evaluation" / "test_dataset.py")
        _load_by_path("parse", root / "vpr_methods_evaluation" / "parse.py")
        from vpr_methods_evaluation.main import (  # noqa: E402
            plot_simmat_with_matches, seq_match_row)
        test_dataset_module = bench_test_dataset
    finally:
        for name, module in displaced.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)

    return Bench(root=root, seq_match_row=seq_match_row,
                 plot_simmat_with_matches=plot_simmat_with_matches,
                 TestDataset=test_dataset_module.TestDataset,
                 test_dataset_module=test_dataset_module)


class Bench:
    """The imported surface, so callers cannot reach past it by accident."""

    def __init__(self, root, seq_match_row, plot_simmat_with_matches,
                 TestDataset, test_dataset_module=None):
        self.root = root
        self.seq_match_row = seq_match_row
        self.plot_simmat_with_matches = plot_simmat_with_matches
        self.TestDataset = TestDataset
        # `read_images_paths` lives here too; the frame index needs the bench's
        # own ordering rather than a second sort that could disagree.
        self.test_dataset_module = test_dataset_module
        self.read_images_paths = getattr(test_dataset_module,
                                         "read_images_paths", None)

    # The matcher's default in the bench's own call sites (main.py:446).
    seq_match_type = "modified"
