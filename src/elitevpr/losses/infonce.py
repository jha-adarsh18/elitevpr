"""Cross-modal InfoNCE against the frozen teacher.

    z_ij = cos( P(s_i), t_j ) / tau,   L = cross_entropy(z, diagonal)

The teacher is frozen, so only the student-to-teacher direction is scored;
there is no symmetric term. `P` is a train-only head as in `FeatureCosine`.

One property of this arm is not benign and belongs in the ablation's caption:
the batch is built by `BlockSampler` as contiguous clips, so the off-diagonal
entries of a row include frames metres away from the query, which this
objective pushes apart as negatives. That is the published form of the loss,
so it is what is measured here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, student_dim, teacher_dim, temperature=0.07):
        super().__init__()
        self.head = nn.Linear(student_dim, teacher_dim)
        self.temperature = float(temperature)

    def forward(self, student_global, teacher_global):
        s = F.normalize(self.head(student_global.float()), p=2, dim=-1)
        t = F.normalize(teacher_global.float(), p=2, dim=-1)
        logits = s @ t.T / self.temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, target)
