"""Patch aggregation.

`clamp` is stock GeM and the historical default. Its `clamp(min=eps)` is a
numerical guard when pooling post-ReLU CNN maps, but the student pools an
`nn.Linear` output, which is signed: measured 2026-08-04, the clamp deletes
62-70% of token entries, those entries receive zero gradient, and every
descriptor is forced into the positive orthant -- effective rank 1.10 of 1024
and mean cosine 0.950-0.956 between different places.

State-dict keys are deliberately incompatible across the families: `clamp`
registers `gem.p`, the others `pool.p`, so a strict load raises on a checkpoint
from the other family instead of pooling it the wrong way.
"""

import torch
import torch.nn as nn


class GeM(nn.Module):
    """Stock generalised mean. The control; do not change."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=1, keepdim=True).pow(1.0 / self.p).squeeze(1)


class SignedGeM(nn.Module):
    """GeM that carries the sign bit through.

        m = mean_n( sign(x) * |x|^p );   out = sign(m) * |m|^(1/p)

    At p=1 this is exactly mean pooling, so the learned p is itself readable.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        m = (torch.sign(x) * x.abs().clamp(min=self.eps).pow(self.p)).mean(dim=1)
        return torch.sign(m) * m.abs().clamp(min=self.eps).pow(1.0 / self.p)


class MeanPool(nn.Module):
    """Plain mean: the p=1 control."""

    def forward(self, x):
        return x.mean(dim=1)


_POOLERS = {"signed": SignedGeM, "mean": MeanPool}
POOLINGS = sorted(_POOLERS) + ["clamp"]


def build_pooler(pooling, p=3.0):
    if pooling == "clamp":
        return GeM(p=p)
    if pooling not in _POOLERS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    cls = _POOLERS[pooling]
    return cls(p=p) if cls is SignedGeM else cls()
