"""Relational KD (Park et al., CVPR 2019): distance + angle terms.

    L = w_d * huber( d(s)/mean d(s), d(t)/mean d(t) )
      + w_a * huber( cos angles in s, cos angles in t )

Both terms are ratios of geometric quantities, so they compare the student's
1024-d space with the teacher's 8448-d space directly and no projection head
is needed -- the same reason `SimilarityPreserving` needs none.

Descriptors are L2-normalised first. The distance term already divides by the
batch-mean distance, so the normalisation is not what makes it scale-free; it
is there so every relational arm reads the same geometry (unit sphere), and so
that the published weights w_d=1, w_a=2 are applied to terms of the size their
authors calibrated them on.

The angle term materialises a B x B x D tensor of pair directions: 32 x 32 x
8448 in float32 is ~35 MB for the teacher, which is why B=32 is affordable.
"""

import torch.nn as nn
import torch.nn.functional as F


def _pdist_ratio(x):
    d = (x.unsqueeze(0) - x.unsqueeze(1)).norm(p=2, dim=-1)
    return d / d[d > 0].mean().clamp(min=1e-12)


def _angles(x):
    e = F.normalize(x.unsqueeze(0) - x.unsqueeze(1), p=2, dim=-1)
    return (e @ e.transpose(1, 2)).reshape(-1)


class RKD(nn.Module):
    def __init__(self, weight_distance=1.0, weight_angle=2.0):
        super().__init__()
        self.weight_distance = float(weight_distance)
        self.weight_angle = float(weight_angle)

    def forward(self, student_global, teacher_global):
        s = F.normalize(student_global.float(), p=2, dim=-1)
        t = F.normalize(teacher_global.float(), p=2, dim=-1)
        # A zero-weighted term is SKIPPED, not multiplied by zero: the angle
        # term alone materialises two B x B x D tensors (35 MB at B=32 for an
        # 8448-d teacher), and the single-term arms `rkd_dist` / `rkd_angle`
        # exist precisely to run without the other one. The returned value is
        # unchanged whenever both weights are non-zero.
        loss = s.new_zeros(())
        if self.weight_distance:
            loss = loss + self.weight_distance * F.smooth_l1_loss(
                _pdist_ratio(s), _pdist_ratio(t))
        if self.weight_angle:
            loss = loss + self.weight_angle * F.smooth_l1_loss(
                _angles(s), _angles(t))
        return loss
