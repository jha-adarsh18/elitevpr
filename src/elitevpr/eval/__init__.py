"""A unified evaluation layer around the official benchmark.

Every method, ours or anyone else's, produces one thing: a similarity matrix.
This layer turns matrices into numbers under a single ground truth, importing
the benchmark and modifying none of it.

    bench         locate, hash-verify and import the official checkout
    protocol      traverses, pairs and how the headline columns are formed
    frames        the frame index: the exact bins the bench scores (item 1)
    ground_truth  positive matrices at 10/25/70 m, cross-checked (item 2)
    scorer        matrix + ground truth -> R@K at every L (item 3)
    cache         content-addressed matrices, never keyed by method name alone
    report        tables, built from result JSONs only
"""

from . import (bench, cache, frames, ground_truth, protocol, report,  # noqa
               scorer)

__all__ = ["bench", "cache", "frames", "ground_truth", "protocol", "report",
           "scorer"]
