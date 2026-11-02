"""One arm is one loss: a module here plus a file in configs/loss/."""

from .feature_cosine import FeatureCosine
from .sp_kd import SimilarityPreserving
from .structural_kl import StructuralKL

__all__ = ["FeatureCosine", "SimilarityPreserving", "StructuralKL", "build_loss"]


def build_loss(cfg, student_dim, teacher_dim):
    """`cfg` is the selected configs/loss/<name>.yaml group."""
    name = str(cfg.name)
    if name == "structural_kl":
        return StructuralKL(temperature=cfg.temperature)
    if name == "feature_cosine":
        return FeatureCosine(student_dim=student_dim, teacher_dim=teacher_dim)
    if name == "sp_kd":
        return SimilarityPreserving()
    raise ValueError(f"unknown loss {name!r}")
