"""Event frames paired with their cached teacher descriptor."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .teacher_cache import TeacherCache

# Column 0 is the timestamp, column 1 the RGB frame, column 2 the event
# histogram. DSEC and ViViD write 3 columns, M3ED 6 (its extra columns are
# other event representations, which are never written and never read).
EVENT_COLUMN = 2
_VALID_COLUMNS = (3, 6)


def sequence_name(rel_path):
    """'rgb/zurich_city_03_a_000123.npy' -> 'zurich_city_03_a'"""
    return Path(rel_path).stem.rsplit("_", 1)[0]


class PairsDataset(Dataset):
    """One preprocessed corpus: `<root>/pairs.txt` plus its teacher cache."""

    def __init__(self, root, features_dir, teacher_cache, sequences=None,
                 pair_stride=1):
        if pair_stride < 1:
            raise ValueError(f"pair_stride must be >= 1, got {pair_stride}")

        self.root = Path(root)
        self.features_dir = Path(features_dir)
        self.teacher_cache = teacher_cache

        pairs_file = self.root / "pairs.txt"
        if not pairs_file.is_file():
            raise FileNotFoundError(
                f"{pairs_file} not found. root must point at a preprocessed "
                f"corpus containing a master pairs.txt."
            )

        wanted = set(sequences) if sequences is not None else None
        seen = set()
        self.pairs = []
        for line in pairs_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entries = line.split(",")
            if len(entries) not in _VALID_COLUMNS:
                raise ValueError(
                    f"{pairs_file}: expected {' or '.join(map(str, _VALID_COLUMNS))} "
                    f"columns, got {len(entries)}"
                )
            seq = sequence_name(entries[1])
            seen.add(seq)
            if wanted is not None and seq not in wanted:
                continue
            self.pairs.append({
                "timestamp_us": int(entries[0]),
                "sequence": seq,
                # frames.txt stores '<seq>/<rgb rel path>' -- the same join the
                # caching script used, and the only link from a pair to a row.
                "feature_key": f"{seq}/{entries[1]}",
                "event_path": self.root / seq / entries[EVENT_COLUMN],
            })

        if pair_stride > 1:
            self.pairs = self.pairs[::pair_stride]
        if not self.pairs:
            raise RuntimeError(
                f"No pairs in {pairs_file} for sequences={sorted(wanted or [])}. "
                f"Present: {sorted(seen)}"
            )

        self._row_index = {}
        for seq in {pair["sequence"] for pair in self.pairs}:
            frames_file = self.features_dir / seq / "frames.txt"
            if not frames_file.is_file():
                raise FileNotFoundError(
                    f"{frames_file} not found -- check features_dir="
                    f"{self.features_dir}"
                )
            for row, line in enumerate(frames_file.read_text().splitlines()):
                if line.strip():
                    self._row_index[line.strip()] = row

        self._mmaps = {}

        # Fail on a wrong root or an unaligned cache now, not mid-epoch.
        probe = self.pairs[0]
        if not probe["event_path"].is_file():
            raise FileNotFoundError(
                f"{probe['event_path']} not found -- check root={self.root}"
            )
        if probe["feature_key"] not in self._row_index:
            raise RuntimeError(
                f"{probe['feature_key']} is not in frames.txt -- check "
                f"features_dir={self.features_dir}"
            )
        self._teacher(probe)

    def __len__(self):
        return len(self.pairs)

    def sequence_names(self):
        return sorted({pair["sequence"] for pair in self.pairs})

    @property
    def teacher_dim(self):
        return self.teacher_cache.width(self.pairs[0]["sequence"])

    def _teacher(self, pair):
        seq = pair["sequence"]
        if seq not in self._mmaps:
            self._mmaps[seq] = self.teacher_cache.open(seq, self.features_dir)
        row = self._mmaps[seq][self._row_index[pair["feature_key"]]]
        return torch.from_numpy(np.ascontiguousarray(row).astype(np.float32))

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        events = np.load(pair["event_path"])          # float16 (3, H, W)
        return (torch.from_numpy(events.astype(np.float32)),
                self._teacher(pair),
                pair["timestamp_us"])


class ConcatPairsDataset(ConcatDataset):
    """Several corpora as one dataset, each keeping its own root and cache."""

    def __init__(self, datasets):
        datasets = list(datasets)
        if not datasets:
            raise ValueError("at least one source is required")
        super().__init__(datasets)
        # ConcatDataset indexes its parts in order, so this stays aligned with
        # dataset indices -- the sampler maps an index to its sequence here.
        self.pairs = [pair for ds in datasets for pair in ds.pairs]

        widths = {ds.teacher_dim for ds in datasets}
        if len(widths) > 1:
            raise ValueError(f"corpora disagree on teacher width: {sorted(widths)}")
        self.teacher_dim = widths.pop()

        counts = Counter(name for ds in datasets for name in ds.sequence_names())
        duplicates = sorted(name for name, c in counts.items() if c > 1)
        if duplicates:
            raise ValueError(
                f"sequence names present in more than one source: {duplicates}"
            )

    def sequence_names(self):
        return sorted({pair["sequence"] for pair in self.pairs})


def build_split(dataset_cfgs, split_key, teacher_file, pair_stride_override=None):
    """Build one split from every corpus that contributes to it.

        null           -> every sequence in that corpus's pairs.txt
        [] or missing  -> that corpus sits this split out
        list of names  -> exactly those sequences

    Each corpus resolves its teacher descriptors under its own `teacher_dir`,
    so a corpus can never be paired with another corpus's cache.
    """
    parts = []
    for cfg in dataset_cfgs:
        if split_key not in cfg:
            continue
        seq_list = cfg[split_key]
        if seq_list is not None and len(seq_list) == 0:
            continue
        stride = (pair_stride_override if pair_stride_override is not None
                  else cfg.get("pair_stride", 1))
        parts.append(PairsDataset(
            root=cfg.root_dir,
            features_dir=cfg.output_dir,
            teacher_cache=TeacherCache(cfg.teacher_dir, teacher_file),
            sequences=list(seq_list) if seq_list is not None else None,
            pair_stride=stride,
        ))
    if not parts:
        raise ValueError(f"no corpus contributes to '{split_key}'")
    return parts[0] if len(parts) == 1 else ConcatPairsDataset(parts)
