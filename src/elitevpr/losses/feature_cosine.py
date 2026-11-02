"""Feature regression on the descriptor itself.

    L = 1/B * sum_i ( 1 - cos( P(s_i), t_i ) )

`P` maps the student's 1024-d descriptor into the teacher's width for the loss
only and is discarded at inference, so the deployed checkpoint is
architecturally identical to the relational arm and the two differ in the
objective alone.
"""

import torch.nn as nn
import torch.nn.functional as F


class FeatureCosine(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.head = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_global, teacher_global):
        s = F.normalize(self.head(student_global.float()), p=2, dim=-1)
        t = F.normalize(teacher_global.float(), p=2, dim=-1)
        return (1.0 - (s * t).sum(dim=-1)).mean()
