"""Similarity-preserving KD (Tung & Mori, ICCV 2019).

    G_s = s s^T,  G_t = t t^T,  each row L2-normalised
    L   = 1/B^2 * || G_s - G_t ||_F^2

Row normalisation is what makes the two Gram matrices comparable despite the
student's 1024-d and the teacher's 8448-d descriptors, so no projection is
needed. The diagonal is kept and nothing is z-scored: this is the published
objective, which is the point of the comparison.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimilarityPreserving(nn.Module):
    def forward(self, student_global, teacher_global):
        s, t = student_global.float(), teacher_global.float()
        g_s = F.normalize(s @ s.T, p=2, dim=1)
        g_t = F.normalize(t @ t.T, p=2, dim=1)
        return (g_s - g_t).pow(2).sum() / g_s.shape[0] ** 2
