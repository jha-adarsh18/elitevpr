import importlib.util
import os
from pathlib import Path

import pytest
import timm
import torch

from elitevpr.models import EventViTStudent, build_pooler

REFERENCE = Path(os.environ.get(
    "ELITEVPR_REFERENCE", "/media/adarsh/Disk1/workspace/E-LiteVPR"))
BACKBONE = "vit_tiny_patch16_224"


def _student(pooling, **kwargs):
    return EventViTStudent(backbone_name=BACKBONE, teacher_dim=32, num_patches=4,
                           img_size=32, pooling=pooling, pretrained=False,
                           **kwargs)


def test_forward_shape_and_finiteness():
    model = _student("signed").eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 32) and torch.isfinite(out).all()


@pytest.mark.parametrize("pooling,key", [("clamp", "gem.p"), ("signed", "pool.p"),
                                         ("mean", None)])
def test_state_dict_keys_identify_the_pooling_family(pooling, key):
    keys = set(_student(pooling).state_dict())
    prefixes = {k.split(".")[0] for k in keys}
    assert prefixes <= {"input_norm", "backbone", "proj", "gem", "pool"}
    assert {"proj.weight", "proj.bias", "input_norm.weight"} <= keys
    if key:
        assert key in keys
    else:
        assert not {"gem.p", "pool.p"} & keys


def test_a_checkpoint_does_not_load_across_pooling_families():
    with pytest.raises(RuntimeError):
        _student("clamp").load_state_dict(_student("signed").state_dict())


def test_signed_pooling_keeps_negative_entries():
    patches = torch.randn(4, 16, 32)
    assert (build_pooler("signed")(patches) < 0).any()
    assert (build_pooler("clamp")(patches) > 0).all()


def test_unknown_pooling_rejected():
    with pytest.raises(ValueError, match="pooling must be one of"):
        _student("gem")


@pytest.mark.skipif(not (REFERENCE / "scripts" / "model.py").is_file(),
                    reason="reference repo not mounted")
def test_descriptor_matches_the_reference_student(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "reference_model", REFERENCE / "scripts" / "model.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ours = _student("signed").eval()

    # The reference hardcodes pretrained=True; the architecture is what is
    # under test, so our weights are copied over afterwards.
    create_model = timm.create_model
    monkeypatch.setattr(reference.timm, "create_model",
                        lambda name, **kw: create_model(
                            name, **{**kw, "pretrained": False}))
    theirs = reference.EventViTStudent(
        backbone_name=BACKBONE, teacher_dim=32, num_patches=4, img_size=32,
        in_channels=3, pooling="signed").eval()
    assert set(ours.state_dict()) == set(theirs.state_dict())

    theirs.load_state_dict(ours.state_dict())
    events = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(ours(events), theirs(events)[1])
