"""Content-addressed cache for similarity matrices.

The bench caches a matrix under
`logs/<subfolder>/<ref>_vs_<qry>_<method>_...npy` -- keyed on the METHOD NAME,
with no checkpoint identity anywhere in the path. Scoring a second checkpoint
under the same method name therefore re-reads the first one's matrix and
reports it as the second one's result, silently. That happened in this project:
every arm had to be given its own copy of the bench tree to avoid it.

Here the key is a hash of everything that can change a matrix:

    method | config hash | checkpoint hash | window | pair

and the five components are stored next to the matrix, so a cache hit can
always be explained and a collision cannot hide.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

KEY_FIELDS = ("method", "config_hash", "checkpoint_hash", "window", "pair")


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path, chunk=1 << 20):
    """sha256 of a file's bytes -- used for checkpoint identity."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_config(config):
    """sha256 of a canonical JSON rendering, so key order cannot matter."""
    return hash_bytes(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                 default=str).encode())


def cache_key(method, config_hash, checkpoint_hash, window, pair):
    """A key that changes when ANY of the five components changes."""
    parts = {"method": str(method), "config_hash": str(config_hash),
             "checkpoint_hash": str(checkpoint_hash), "window": str(window),
             "pair": str(pair)}
    return hash_config(parts)[:32], parts


class MatrixCache:
    """`<root>/<method>/<key>.npy` plus a sidecar naming the five components.

    The method name is a directory only so a human can browse the cache; it is
    never the identity. Two checkpoints of one method land in different files
    because the key hashes the checkpoint.
    """

    def __init__(self, root):
        self.root = Path(root)

    def _paths(self, method, key):
        base = self.root / str(method) / key
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def load(self, method, key):
        matrix_path, meta_path = self._paths(method, key)
        if not (matrix_path.is_file() and meta_path.is_file()):
            return None, None
        return np.load(matrix_path), json.loads(meta_path.read_text())

    def store(self, method, key, parts, matrix, extra=None):
        matrix_path, meta_path = self._paths(method, key)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(matrix)
        np.save(matrix_path, matrix)
        meta = dict(parts)
        meta.update({"key": key, "shape": list(matrix.shape),
                     "dtype": str(matrix.dtype),
                     "matrix_sha256": hash_file(matrix_path)})
        if extra:
            meta.update(extra)
        meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
        return matrix_path, meta

    def get_or_compute(self, method, config_hash, checkpoint_hash, window,
                       pair, compute, extra=None):
        """Returns (matrix, meta, hit). `compute` is only called on a miss."""
        key, parts = cache_key(method, config_hash, checkpoint_hash, window,
                               pair)
        matrix, meta = self.load(method, key)
        if matrix is not None:
            return matrix, meta, True
        matrix = compute()
        _, meta = self.store(method, key, parts, matrix, extra)
        return matrix, meta, False
