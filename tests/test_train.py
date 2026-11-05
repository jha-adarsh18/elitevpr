"""End-to-end wiring: every arm must train, checkpoint and resume on CPU."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from elitevpr.train import main_worker

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")
ARMS = ["structural_kl", "feature_cosine", "feature_mse", "infonce",
        "rkd", "rkd_dist", "rkd_angle", "sp_kd", "sp_kd_std"]


def _cfg(smoke_root, loss, **extra):
    overrides = [
        f"loss={loss}", "logging=none",
        "~corpora.m3ed", "~corpora.vivid",
        f"data_root={smoke_root}",
        f"checkpoint_dir={smoke_root}/ckpt_{loss}",
        "corpora.dsec.train_seq_list=[day_one,day_two,night_one]",
        "corpora.dsec.val_seq_list=[day_two]",
        "corpora.dsec.night_sequences=[night_one]",
        "model.backbone_name=vit_tiny_patch16_224",
        "model.img_size=32", "model.num_patches=4", "model.teacher_dim=32",
        "model.pretrained=false",
        "training.epochs=1", "training.batch_size=8", "training.block_size=4",
        "num_workers=0", "allow_cpu=true",
    ]
    overrides += [f"{k}={v}" for k, v in extra.items()]
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("loss", ARMS)
def test_every_arm_trains_and_writes_both_checkpoints(smoke_root, loss):
    result = main_worker(_cfg(smoke_root, loss))

    assert torch.isfinite(torch.tensor(result["best_val"]))
    for key in ("best", "last"):
        state = torch.load(result[key], map_location="cpu", weights_only=True)
        # Only the student is deployed: no loss head reaches the checkpoint.
        assert "pool.p" in state
        assert not any(k.startswith("head.") for k in state)


def test_the_loss_head_lives_only_in_the_resume_state(smoke_root):
    main_worker(_cfg(smoke_root, "feature_cosine"))
    resume = torch.load(smoke_root / "ckpt_feature_cosine" / "resume.pth",
                        map_location="cpu", weights_only=False)
    assert set(resume["criterion"]) == {"head.weight", "head.bias"}
    assert resume["epoch"] == 0


def test_resume_continues_from_the_stored_epoch(smoke_root, capsys):
    main_worker(_cfg(smoke_root, "structural_kl"))
    capsys.readouterr()
    main_worker(_cfg(smoke_root, "structural_kl", **{"training.epochs": 2}))
    assert "resumed from" in capsys.readouterr().out

    resume = torch.load(smoke_root / "ckpt_structural_kl" / "resume.pth",
                        map_location="cpu", weights_only=False)
    assert resume["epoch"] == 1


def test_cpu_training_is_refused_without_the_flag(smoke_root):
    cfg = _cfg(smoke_root, "structural_kl")
    cfg.allow_cpu = False
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(SystemExit, match="CUDA is not available"):
        main_worker(cfg)


def test_batch_size_must_be_a_multiple_of_the_block(smoke_root):
    cfg = _cfg(smoke_root, "structural_kl", **{"training.block_size": 5})
    with pytest.raises(SystemExit, match="multiple of"):
        main_worker(cfg)


def test_unknown_night_sequence_is_rejected(smoke_root):
    cfg = _cfg(smoke_root, "structural_kl")
    cfg.corpora.dsec.night_sequences = ["night_one", "typo_seq"]
    with pytest.raises(ValueError, match="typo_seq"):
        main_worker(cfg)


def test_data_root_may_be_unset_when_every_corpus_is_explicit():
    """The Kaggle-built caches do not follow the data_root convention."""
    from omegaconf import OmegaConf

    overrides = ["checkpoint_dir=/runs/x"]
    for corpus in ("dsec", "m3ed", "vivid"):
        overrides += [f"corpora.{corpus}.root_dir=/root/{corpus}",
                      f"corpora.{corpus}.output_dir=/feats/{corpus}",
                      f"corpora.{corpus}.teacher_dir=/cache/megaloc"]
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["corpora"]["m3ed"]["teacher_dir"] == "/cache/megaloc"


def test_random_sampler_arm_trains_and_reports_itself(smoke_root, capsys):
    """`training.sampler=random` is the rkd_randbatch control: uniform draws
    over the whole corpus instead of contiguous clips."""
    cfg = _cfg(smoke_root, "rkd", **{"training.sampler": "random"})
    cfg.checkpoint_dir = f"{smoke_root}/ckpt_rkd_randbatch"
    result = main_worker(cfg)
    out = capsys.readouterr().out
    assert "sampler   random" in out
    assert "drawn uniformly at random" in out
    assert "night_frac ignored" in out
    assert torch.isfinite(torch.tensor(result["best_val"]))
    assert Path(f"{smoke_root}/ckpt_rkd_randbatch/best.pth").is_file()


def test_random_sampler_ignores_the_block_divisibility_rule(smoke_root):
    """The constraint exists for contiguity; without blocks it must not fire."""
    cfg = _cfg(smoke_root, "rkd", **{"training.sampler": "random",
                                     "training.batch_size": 6})
    cfg.checkpoint_dir = f"{smoke_root}/ckpt_odd_batch"
    main_worker(cfg)      # block_size=4 does not divide 6; this must not raise


def test_unknown_sampler_is_refused(smoke_root):
    with pytest.raises(SystemExit, match="training.sampler"):
        main_worker(_cfg(smoke_root, "rkd", **{"training.sampler": "shuffle"}))


def test_teacher_group_moves_the_cache_dir_and_the_filename():
    """One override must repoint every corpus, or an arm would mix teachers."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for teacher, dir_name, filename in (("megaloc", "megaloc", "megaloc.npy"),
                                        ("mixvpr", "mixvpr", "mixvpr.npy"),
                                        ("dinov3_gem", "gem", "gem.npy"),
                                        ("dinov3_cls", "cls", "cls.npy")):
        with initialize_config_dir(config_dir=CONFIGS, version_base=None):
            cfg = compose(config_name="config",
                          overrides=[f"teacher={teacher}", "data_root=/data",
                                     "checkpoint_dir=/ckpt"])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["teacher"]["file"] == filename
        for corpus in ("dsec", "m3ed", "vivid"):
            assert resolved["corpora"][corpus]["teacher_dir"] == \
                f"/data/{corpus}/{dir_name}"


def test_megaloc_remains_the_default_teacher():
    """The reported recipe must not move because a group was introduced."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="config",
                      overrides=["data_root=/data", "checkpoint_dir=/ckpt"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["teacher"]["name"] == "megaloc"
    assert resolved["teacher"]["file"] == "megaloc.npy"
    assert resolved["corpora"]["dsec"]["teacher_dir"] == "/data/dsec/megaloc"
    assert resolved["training"]["sampler"] == "block"
