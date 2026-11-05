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

from elitevpr.losses import (FeatureCosine, FeatureMSE, InfoNCE, RKD,
                             SimilarityPreserving,
                             SimilarityPreservingStandardized, StructuralKL,
                             build_loss)

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


def test_feature_mse_is_twice_feature_cosine():
    """On unit vectors ||a-b||^2 = 2(1-a.b), so B is A doubled, exactly."""
    student, teacher = torch.randn(8, 16), torch.randn(8, 24)
    mse = FeatureMSE(student_dim=16, teacher_dim=24)
    cosine = FeatureCosine(student_dim=16, teacher_dim=24)
    cosine.load_state_dict(mse.state_dict())
    assert torch.allclose(mse(student, teacher), 2.0 * cosine(student, teacher),
                          atol=1e-6)


def test_feature_mse_cannot_be_gamed_by_shrinking_the_projection():
    """The projection is normalised, so scaling its output changes nothing."""
    criterion = FeatureMSE(student_dim=4, teacher_dim=4)
    student, teacher = torch.randn(6, 4), torch.randn(6, 4)
    before = criterion(student, teacher)
    with torch.no_grad():
        criterion.head.weight.mul_(1e-3)
        criterion.head.bias.mul_(1e-3)
    assert torch.allclose(criterion(student, teacher), before, atol=1e-4)


def test_infonce_matches_the_published_form():
    criterion = InfoNCE(student_dim=16, teacher_dim=24, temperature=0.07)
    student, teacher = torch.randn(8, 16), torch.randn(8, 24)

    zs = F.normalize(criterion.head(student), dim=1)
    logits = zs @ F.normalize(teacher, dim=1).T / 0.07
    expected = F.cross_entropy(logits, torch.arange(8))
    assert torch.allclose(criterion(student, teacher), expected, atol=1e-6)


def test_infonce_is_minimised_by_the_matching_pair():
    """A perfectly aligned batch must beat the same batch with rolled targets."""
    criterion = InfoNCE(student_dim=4, teacher_dim=4, temperature=0.07)
    with torch.no_grad():
        criterion.head.weight.copy_(torch.eye(4))
        criterion.head.bias.zero_()
    teacher = F.normalize(torch.randn(6, 4), dim=1)
    assert criterion(teacher, teacher) < criterion(teacher.roll(1, 0), teacher)


def test_rkd_is_zero_when_the_geometry_agrees():
    descriptors = torch.randn(10, 64)
    assert RKD()(descriptors, descriptors).abs() < 1e-7


def test_rkd_needs_no_projection_and_has_no_parameters():
    """Distances and angles are ratios, so 1024-d vs 8448-d compares directly."""
    criterion = RKD()
    assert list(criterion.parameters()) == []
    loss = criterion(torch.randn(8, 1024), torch.randn(8, 8448))
    assert torch.isfinite(loss) and loss > 0


def test_rkd_invariant_to_descriptor_scale():
    student, teacher = torch.randn(10, 64), torch.randn(10, 96)
    criterion = RKD()
    assert torch.allclose(criterion(student, teacher),
                          criterion(student * 5.0, teacher * 0.2), atol=1e-5)


def test_rkd_weights_the_two_terms_as_published():
    student, teacher = torch.randn(8, 32), torch.randn(8, 48)
    only_d = RKD(weight_distance=1.0, weight_angle=0.0)(student, teacher)
    only_a = RKD(weight_distance=0.0, weight_angle=1.0)(student, teacher)
    both = RKD(weight_distance=1.0, weight_angle=2.0)(student, teacher)
    assert torch.allclose(both, only_d + 2.0 * only_a, atol=1e-6)


def test_build_loss_dispatch():
    from omegaconf import OmegaConf

    assert isinstance(build_loss(OmegaConf.create(
        {"name": "structural_kl", "temperature": 1.0}), 1024, 8448), StructuralKL)
    assert isinstance(build_loss(OmegaConf.create({"name": "feature_cosine"}),
                                 1024, 8448), FeatureCosine)
    assert isinstance(build_loss(OmegaConf.create({"name": "sp_kd"}), 1024, 8448),
                      SimilarityPreserving)
    assert isinstance(build_loss(OmegaConf.create({"name": "feature_mse"}),
                                 1024, 8448), FeatureMSE)
    assert isinstance(build_loss(OmegaConf.create(
        {"name": "infonce", "temperature": 0.07}), 1024, 8448), InfoNCE)
    assert isinstance(build_loss(OmegaConf.create(
        {"name": "rkd", "weight_distance": 1.0, "weight_angle": 2.0}),
        1024, 8448), RKD)
    assert isinstance(build_loss(OmegaConf.create({"name": "sp_kd_std"}),
                                 1024, 8448), SimilarityPreservingStandardized)
    for name, wd, wa in (("rkd_dist", 1.0, 0.0), ("rkd_angle", 0.0, 2.0)):
        built = build_loss(OmegaConf.create(
            {"name": name, "weight_distance": wd, "weight_angle": wa}),
            1024, 8448)
        assert isinstance(built, RKD)
        assert (built.weight_distance, built.weight_angle) == (wd, wa)
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss(OmegaConf.create({"name": "nope"}), 1024, 8448)


# ----------------------------------------------------------------- sp_kd_std

def test_sp_kd_std_is_zero_on_identical_geometry():
    """Same relational structure in both spaces -> nothing to learn."""
    x = torch.randn(12, 64)
    q, _ = torch.linalg.qr(torch.randn(64, 64))           # a rotation preserves
    criterion = SimilarityPreservingStandardized()        # every inner product
    assert criterion(x, x @ q).abs() < 1e-5


def test_sp_kd_std_drops_the_diagonal_and_z_scores_each_matrix():
    """Reproduce the documented expression exactly, off the module's own path."""
    from elitevpr.losses.structural_kl import drop_diag, standardize

    student, teacher = torch.randn(9, 32), torch.randn(9, 48)
    z_s = standardize(drop_diag(student @ student.T))
    z_t = standardize(drop_diag(teacher @ teacher.T))
    expected = (z_s - z_t).pow(2).mean()
    assert torch.allclose(SimilarityPreservingStandardized()(student, teacher),
                          expected, atol=1e-7)
    # and the standardised matrices really are zero-mean / unit-std, which is
    # the property the raw Frobenius form lacks
    for z in (z_s, z_t):
        assert z.mean().abs() < 1e-5 and abs(z.std().item() - 1.0) < 1e-3
        assert z.shape == (9, 8)                          # the diagonal is gone


def test_sp_kd_std_is_invariant_to_a_global_descriptor_scale():
    """This is why the row-wise L2 normalisation of the inputs was dropped:
    scaling by c scales the Gram by c^2 and the z-score divides it out. The
    published arm is not scale-free -- its raw Gram grows with the norm."""
    student, teacher = torch.randn(10, 64), torch.randn(10, 96)
    criterion = SimilarityPreservingStandardized()
    assert torch.allclose(criterion(student, teacher),
                          criterion(student * 7.0, teacher * 0.1), atol=1e-5)


def test_sp_kd_std_does_not_normalise_per_row_and_that_is_deliberate():
    """Pins the decision, so nobody re-adds F.normalize to "fix" it.

    Without the input normalisation, PER-ROW norms are no longer neutralised.
    Measured cost: with the directions fixed and lognormal row norms spanning
    ~51x, the loss moves ~0.7% of its value -- small, and the teacher cache is
    unit-norm already (measured 0.999927..1.000014 on the real DSEC cache), so
    only the student side is affected at all.
    """
    criterion = SimilarityPreservingStandardized()
    directions = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
    teacher = torch.nn.functional.normalize(torch.randn(16, 96), dim=-1)
    g = torch.Generator().manual_seed(1)
    rescaled = directions * torch.exp(torch.randn(16, 1, generator=g))
    base = criterion(directions, teacher)
    moved = criterion(rescaled, teacher)
    assert not torch.allclose(base, moved, atol=1e-5)       # it does matter
    assert (moved - base).abs() < 0.05 * base               # but only a little


def test_sp_kd_std_is_not_the_published_sp_kd():
    """Guard against the two arms silently converging on one implementation."""
    student, teacher = torch.randn(16, 64), torch.randn(16, 128)
    assert not torch.allclose(SimilarityPreservingStandardized()(student, teacher),
                              SimilarityPreserving()(student, teacher), atol=1e-3)


def test_sp_kd_std_is_parameter_free_across_mismatched_widths():
    criterion = SimilarityPreservingStandardized()
    assert list(criterion.parameters()) == []
    loss = criterion(torch.randn(8, 1024), torch.randn(8, 8448))
    assert torch.isfinite(loss) and loss > 0


def test_sp_kd_std_has_a_live_gradient():
    student = torch.randn(16, 64, requires_grad=True)
    SimilarityPreservingStandardized()(student, torch.randn(16, 128)).backward()
    assert student.grad.norm() > 1e-4


# --------------------------------------------- rkd single-term arms

def test_rkd_dist_skips_the_angle_term_exactly():
    student, teacher = torch.randn(8, 32), torch.randn(8, 48)
    skipped = RKD(weight_distance=1.0, weight_angle=0.0)(student, teacher)
    import torch.nn.functional as F
    from elitevpr.losses.rkd import _pdist_ratio
    s = F.normalize(student.float(), p=2, dim=-1)
    t = F.normalize(teacher.float(), p=2, dim=-1)
    assert torch.allclose(skipped,
                          F.smooth_l1_loss(_pdist_ratio(s), _pdist_ratio(t)),
                          atol=1e-7)


def test_rkd_angle_skips_the_distance_term_exactly():
    student, teacher = torch.randn(8, 32), torch.randn(8, 48)
    skipped = RKD(weight_distance=0.0, weight_angle=1.0)(student, teacher)
    import torch.nn.functional as F
    from elitevpr.losses.rkd import _angles
    s = F.normalize(student.float(), p=2, dim=-1)
    t = F.normalize(teacher.float(), p=2, dim=-1)
    assert torch.allclose(skipped, F.smooth_l1_loss(_angles(s), _angles(t)),
                          atol=1e-7)


def test_single_term_rkd_arms_still_train():
    for wd, wa in ((1.0, 0.0), (0.0, 2.0)):
        student = torch.randn(12, 64, requires_grad=True)
        RKD(weight_distance=wd, weight_angle=wa)(
            student, torch.randn(12, 96)).backward()
        assert student.grad.norm() > 1e-6, (wd, wa)
