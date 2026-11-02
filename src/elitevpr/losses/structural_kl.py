"""Relational KL: match the batch-wise similarity structure of the teacher.

Three properties are load-bearing and were each paid for:

* The self-term is dropped (B x B -> B x (B-1)). With the diagonal present it
  dominates the softmax and the loss degenerates into a uniformity prior.
* Both matrices are z-scored, which is what makes the temperature comparable
  across teachers. One shared tau=0.05 on raw similarities gave MegaLoc a
  gradient norm of 2.1e-09 -- dead -- while giving DINOv3 a literally uniform
  target. tau is therefore in units of sigma, and 1.0 is the calibrated value.
* float32 throughout: a std or softmax over a near-degenerate similarity
  matrix is exactly where fp16 becomes inf/nan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_diag(sim):
    n = sim.shape[0]
    return sim[~torch.eye(n, dtype=torch.bool, device=sim.device)].view(n, n - 1)


def standardize(sim):
    return (sim - sim.mean()) / sim.std().clamp(min=1e-6)


class StructuralKL(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_global, teacher_global):
        s = F.normalize(student_global.float(), p=2, dim=-1)
        t = F.normalize(teacher_global.float(), p=2, dim=-1)
        s_sim = standardize(drop_diag(s @ s.T)) / self.temperature
        t_sim = standardize(drop_diag(t @ t.T)) / self.temperature
        return F.kl_div(F.log_softmax(s_sim, dim=-1),
                        F.softmax(t_sim, dim=-1),
                        reduction="batchmean")
