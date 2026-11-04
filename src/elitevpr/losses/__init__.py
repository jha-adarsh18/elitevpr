"""One arm is one loss: a module here plus a file in configs/loss/."""

from .feature_cosine import FeatureCosine
from .feature_mse import FeatureMSE
from .infonce import InfoNCE
from .rkd import RKD
from .sp_kd import SimilarityPreserving
from .structural_kl import StructuralKL

__all__ = ["FeatureCosine", "FeatureMSE", "InfoNCE", "RKD",
           "SimilarityPreserving", "StructuralKL", "build_loss"]


def build_loss(cfg, student_dim, teacher_dim):
    """`cfg` is the selected configs/loss/<name>.yaml group."""
    name = str(cfg.name)
    if name == "structural_kl":
        return StructuralKL(temperature=cfg.temperature)
    if name == "feature_cosine":
        return FeatureCosine(student_dim=student_dim, teacher_dim=teacher_dim)
    if name == "feature_mse":
        return FeatureMSE(student_dim=student_dim, teacher_dim=teacher_dim)
    if name == "infonce":
        return InfoNCE(student_dim=student_dim, teacher_dim=teacher_dim,
                       temperature=cfg.temperature)
    if name == "rkd":
        return RKD(weight_distance=cfg.weight_distance,
                   weight_angle=cfg.weight_angle)
    if name == "sp_kd":
        return SimilarityPreserving()
    raise ValueError(f"unknown loss {name!r}")
