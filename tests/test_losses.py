"""Loss parity and loss algebra.

`REFERENCE_STRUCTURAL_LOSS` is the settled objective, copied verbatim from the
pipeline that produced every reported number
(`<reference>/scripts/train_megaloc.py`, `structural_loss`). Two tests use it:
one checks our module reproduces it exactly, one checks the reference file
itself still contains that text, so a change on either side is caught.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from elitevpr.losses import FeatureCosine, SimilarityPreserving, StructuralKL, build_loss

REFERENCE_STRUCTURAL_LOSS = '''def structural_loss(student_global, teacher_global, temperature):
    """The settled fixed loss: drop the self-term, z-score, tau in sigma units.

    float32 throughout -- std/softmax on a near-degenerate similarity matrix is
    exactly where autocast fp16 turns into inf/nan.
    """
    s = F.normalize(student_global.float(), p=2, dim=-1)
    t = F.normalize(teacher_global.float(), p=2, dim=-1)
    s_sim, t_sim = _drop_diag(s @ s.T), _drop_diag(t @ t.T)
    s_sim = (s_sim - s_sim.mean()) / s_sim.std().clamp(min=1e-6)
    t_sim = (t_sim - t_sim.mean()) / t_sim.std().clamp(min=1e-6)
    return F.kl_div(F.log_softmax(s_sim / temperature, dim=-1),
                    F.softmax(t_sim / temperature, dim=-1),
                    reduction="batchmean")'''

REFERENCE = Path(os.environ.get(
    "ELITEVPR_REFERENCE", "/media/adarsh/Disk1/workspace/E-LiteVPR"))


def reference_structural_loss(student_global, teacher_global, temperature):
    def _drop_diag(sim):
        n = sim.shape[0]
        return sim[~torch.eye(n, dtype=torch.bool, device=sim.device)].view(n, n - 1)

    scope = {"F": F, "torch": torch, "_drop_diag": _drop_diag}
    exec(REFERENCE_STRUCTURAL_LOSS, scope)
    return scope["structural_loss"](student_global, teacher_global, temperature)


@pytest.mark.parametrize("temperature", [1.0, 0.05])
@pytest.mark.parametrize("batch", [8, 32])
def test_structural_kl_matches_reference(batch, temperature):
    student = torch.randn(batch, 1024, requires_grad=True)
    teacher = torch.randn(batch, 8448)

    ours = StructuralKL(temperature=temperature)(student, teacher)
    theirs = reference_structural_loss(student, teacher, temperature)
    assert torch.equal(ours, theirs)

    ours.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


@pytest.mark.skipif(not REFERENCE.is_dir(), reason="reference repo not mounted")
def test_reference_source_has_not_drifted():
    source = (REFERENCE / "scripts" / "train_megaloc.py").read_text()
    assert REFERENCE_STRUCTURAL_LOSS in source, (
        "the reference objective changed: re-check the parity snapshot")


def test_structural_kl_is_zero_when_structures_agree():
    teacher = torch.randn(16, 8448)
    student = teacher[:, :1024]
    loss = StructuralKL(temperature=1.0)(student, student)
    assert loss.abs() < 1e-6
    assert StructuralKL(1.0)(student, teacher) > 0


def test_feature_cosine_is_one_minus_mean_cosine():
    criterion = FeatureCosine(student_dim=16, teacher_dim=24)
    student, teacher = torch.randn(8, 16), torch.randn(8, 24)

    expected = (1.0 - F.cosine_similarity(criterion.head(student), teacher,
                                          dim=-1)).mean()
    assert torch.allclose(criterion(student, teacher), expected, atol=1e-6)


def test_feature_cosine_bounds_and_head_is_trainable():
    criterion = FeatureCosine(student_dim=4, teacher_dim=4)
    with torch.no_grad():
        criterion.head.weight.copy_(torch.eye(4))
        criterion.head.bias.zero_()
    aligned = torch.randn(5, 4)
    assert criterion(aligned, aligned).abs() < 1e-6
    assert criterion(aligned, -aligned).item() == pytest.approx(2.0, abs=1e-5)
    assert [name for name, _ in criterion.named_parameters()] == \
        ["head.weight", "head.bias"]


def test_sp_kd_matches_the_published_form():
    student, teacher = torch.randn(12, 1024), torch.randn(12, 8448)
    g_s = F.normalize(student @ student.T, p=2, dim=1)
    g_t = F.normalize(teacher @ teacher.T, p=2, dim=1)
    expected = (g_s - g_t).pow(2).sum() / 12 ** 2
    assert torch.allclose(SimilarityPreserving()(student, teacher), expected)


def test_sp_kd_is_zero_on_an_identical_gram():
    descriptors = torch.randn(10, 64)
    assert SimilarityPreserving()(descriptors, descriptors).abs() < 1e-10


def test_sp_kd_invariant_to_descriptor_scale():
    student, teacher = torch.randn(10, 64), torch.randn(10, 96)
    criterion = SimilarityPreserving()
    assert torch.allclose(criterion(student, teacher),
                          criterion(student * 5.0, teacher * 0.2), atol=1e-5)


def test_build_loss_dispatch():
    from omegaconf import OmegaConf

    assert isinstance(build_loss(OmegaConf.create(
        {"name": "structural_kl", "temperature": 1.0}), 1024, 8448), StructuralKL)
    assert isinstance(build_loss(OmegaConf.create({"name": "feature_cosine"}),
                                 1024, 8448), FeatureCosine)
    assert isinstance(build_loss(OmegaConf.create({"name": "sp_kd"}), 1024, 8448),
                      SimilarityPreserving)
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss(OmegaConf.create({"name": "nope"}), 1024, 8448)
