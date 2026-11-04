"""Feature regression on the descriptor itself, squared-error form.

    L = 1/B * sum_i || P(s_i)/||P(s_i)|| - t_i ||_2^2

`P(s_i)` is normalised before the difference is taken, so the loss cannot be
driven down by shrinking the projection's norm instead of turning it. On unit
vectors ||a - b||^2 = 2(1 - a.b), so this is exactly twice `FeatureCosine` and
is reported as a sanity check on that arm, not as an independent row.

`P` is a train-only head, discarded at inference, as in `FeatureCosine`.
"""

import torch.nn as nn
import torch.nn.functional as F


class FeatureMSE(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.head = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_global, teacher_global):
        s = F.normalize(self.head(student_global.float()), p=2, dim=-1)
        t = F.normalize(teacher_global.float(), p=2, dim=-1)
        return (s - t).pow(2).sum(dim=-1).mean()
