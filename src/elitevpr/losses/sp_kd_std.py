"""Similarity-preserving KD with the two fixes that rescued the KL arm.

`SimilarityPreserving` is the published objective (Tung & Mori, ICCV 2019): it
row-normalises each Gram matrix and keeps the diagonal. This arm replaces that
normalisation with the pair that `StructuralKL` needed, and keeps the Frobenius
form instead of a KL:

    G  = s s^T                             raw Gram, nothing normalised
    G' = drop_diag(G)                      B x B -> B x (B-1)
    Z  = (G' - mean G') / std G'           one scalar mean/std per matrix
    L  = mean( (Z_s - Z_t)^2 )

Why each piece:

* **The diagonal goes.** G[i, i] carries no relation at all -- for unit-norm
  descriptors it is the constant 1 -- so keeping it dilutes the informative
  part of the matrix by 1/B and lets a model score well while learning nothing.
* **Each matrix is z-scored by its OWN off-diagonal statistics.** The student's
  1024-d and the teacher's 8448-d Grams occupy different ranges -- measured
  off-diagonal cosine was 0.96 for one teacher and near 0 for another -- so a
  raw Frobenius difference is dominated by that offset rather than by the
  relational structure. Standardising both to zero mean and unit variance makes
  the comparison about the RANKING of the pairs. This is the fix that moved the
  KL arm from a dead gradient (2.1e-09) to a live one, and because it is taken
  after the diagonal is dropped, the statistics are off-diagonal by
  construction.

**No row-wise L2 normalisation of the descriptors, by measurement.** An earlier
version normalised both inputs. It was removed because:

* the teacher cache is already unit-norm -- MegaLoc's own output, measured on
  the real DSEC cache at row norms 0.999927 to 1.000014 (std 6e-06), so the
  call was a no-op on that side;
* `standardize` removes a global scale exactly. Scaling the student by c scales
  G by c^2, which the z-score divides out: measured identical to 6 decimal
  places at c = 1, 5 and 0.01;
* what remains is per-ROW norm sensitivity, and it is small. With the student's
  directions fixed and lognormal row norms spanning 2x, 7x and 51x, the loss
  moved +0.0024, +0.0054 and +0.0130 against a base of 1.992 -- 0.1% to 0.7%,
  and not monotonic in the spread.

The two helpers are imported from `structural_kl` rather than copied: these are
the same two operations, and two divergent copies of a load-bearing fix is how
the tau bug came back the second time. Note `StructuralKL` keeps its own
normalisation -- it is parity-locked to the reference implementation by test.
"""

import torch.nn as nn

from .structural_kl import drop_diag, standardize


class SimilarityPreservingStandardized(nn.Module):
    def forward(self, student_global, teacher_global):
        s, t = student_global.float(), teacher_global.float()
        z_s = standardize(drop_diag(s @ s.T))
        z_t = standardize(drop_diag(t @ t.T))
        return (z_s - z_t).pow(2).mean()
