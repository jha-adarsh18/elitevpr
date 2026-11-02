"""Contiguous-block sampler.

The objective is a relation between frames inside one batch, so it can only
teach from frames that are actually related and present together. Drawing 32
indices independently from a 21k-frame corpus spanning ~58 traverses gives a
teacher matrix that is nearly the identity: measured on the MegaLoc cache at
B=32, a random sampler leaves 0.0007 of the softmax mass off the diagonal,
spread over 24.8 effective neighbours, while contiguous blocks leave 0.0659
over 3.2. The random-sampler run converged to a training loss of 0.0007, i.e.
it reached the ceiling of its own objective in two epochs.

Blocks are contiguous in `timestamp_us` within a sequence, not in pairs.txt
order, so a block is temporally adjacent however the master file was
concatenated. Day/night balance moves to the block level, because balancing
individual frames would break contiguity.
"""

import torch
from torch.utils.data import Sampler


class BlockSampler(Sampler):
    """Indices in contiguous runs of `block` frames from one sequence.

    `batch_size % block == 0` is required of the caller, otherwise the
    DataLoader slices across block boundaries.
    """

    def __init__(self, dataset, night_seqs, block=8, night_frac=0.5,
                 num_samples=None, generator=None, verbose=True):
        self.block = int(block)
        self.night_frac = float(night_frac)
        self.generator = generator
        n = len(dataset) if num_samples is None else int(num_samples)
        self.n_blocks = max(1, n // self.block)

        by_seq = {}
        for idx, pair in enumerate(dataset.pairs):
            by_seq.setdefault(pair["sequence"], []).append(
                (pair["timestamp_us"], idx))

        night_seqs = set(night_seqs or [])
        self.day_groups, self.night_groups = [], []
        dropped = []
        for seq, rows in by_seq.items():
            rows.sort(key=lambda row: row[0])
            idxs = torch.tensor([row[1] for row in rows], dtype=torch.long)
            if idxs.numel() < self.block:
                dropped.append(seq)
                continue
            target = self.night_groups if seq in night_seqs else self.day_groups
            target.append(idxs)

        if not self.day_groups and not self.night_groups:
            raise ValueError(
                f"no sequence has at least block={self.block} frames"
            )
        if not self.night_groups:
            self.night_frac = 0.0
        if not self.day_groups:
            self.night_frac = 1.0

        if verbose:
            print(f"Block sampler: block={self.block}, "
                  f"night_frac={self.night_frac:.2f}, "
                  f"{len(self.day_groups)} day / {len(self.night_groups)} night "
                  f"sequences, {self.n_blocks} blocks/epoch "
                  f"({self.n_blocks * self.block} samples)")
            if dropped:
                print(f"  dropped {len(dropped)} sequences shorter than block: "
                      f"{sorted(dropped)[:5]}{' ...' if len(dropped) > 5 else ''}")

    def __len__(self):
        return self.n_blocks * self.block

    def __iter__(self):
        g = self.generator
        if g is None:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        def pick(groups):
            # A sequence is drawn uniformly, so a long traverse is no more
            # likely to be sampled than a short one.
            j = int(torch.randint(len(groups), (1,), generator=g).item())
            idxs = groups[j]
            start = int(torch.randint(idxs.numel() - self.block + 1, (1,),
                                      generator=g).item())
            return idxs[start:start + self.block]

        for _ in range(self.n_blocks):
            night = torch.rand((), generator=g).item() < self.night_frac
            yield from pick(self.night_groups if night else self.day_groups).tolist()
