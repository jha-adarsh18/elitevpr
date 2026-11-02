import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from elitevpr.data import PairsDataset, TeacherCache, build_split
from elitevpr.data.dataset import sequence_name


def _cfg(corpus, **overrides):
    cfg = {"train_seq_list": None, "val_seq_list": [], "pair_stride": 1,
           "night_sequences": [], **corpus, **overrides}
    return OmegaConf.create(cfg)


def test_sequence_name_strips_frame_index():
    assert sequence_name("rgb/zurich_city_03_a_000123.npy") == "zurich_city_03_a"


@pytest.mark.parametrize("fixture", ["corpus", "corpus6"])
def test_loads_both_column_contracts(request, fixture):
    corpus = request.getfixturevalue(fixture)
    ds = PairsDataset(corpus["root_dir"], corpus["output_dir"],
                      TeacherCache(corpus["teacher_dir"], "megaloc.npy"))
    assert len(ds) == 120
    assert len(ds.sequence_names()) == 3
    events, teacher, ts = ds[0]
    assert events.shape == (3, 32, 32) and events.dtype == torch.float32
    assert teacher.shape == (64,) and ts == 1_000_000


def test_teacher_row_matches_the_cache(corpus):
    ds = PairsDataset(corpus["root_dir"], corpus["output_dir"],
                      TeacherCache(corpus["teacher_dir"], "megaloc.npy"))
    cache = np.load(f"{corpus['teacher_dir']}/day_one/megaloc.npy")
    pair_idx = next(i for i, p in enumerate(ds.pairs)
                    if p["sequence"] == "day_one" and p["timestamp_us"] == 7_000_000)
    _events, teacher, _ts = ds[pair_idx]
    assert torch.allclose(teacher, torch.from_numpy(cache[6].astype(np.float32)))


def test_sequence_filter_and_stride(corpus):
    ds = PairsDataset(corpus["root_dir"], corpus["output_dir"],
                      TeacherCache(corpus["teacher_dir"], "megaloc.npy"),
                      sequences=["night_one"], pair_stride=4)
    assert len(ds) == 10 and ds.sequence_names() == ["night_one"]


def test_unaligned_cache_raises(corpus):
    path = f"{corpus['teacher_dir']}/day_one/megaloc.npy"
    np.save(path, np.load(path)[:-1])
    with pytest.raises(RuntimeError, match="frames.txt"):
        PairsDataset(corpus["root_dir"], corpus["output_dir"],
                     TeacherCache(corpus["teacher_dir"], "megaloc.npy"))


def test_missing_cache_names_every_root(corpus):
    cache = TeacherCache([corpus["teacher_dir"], "/nope"], "mixvpr.npy")
    with pytest.raises(FileNotFoundError, match="mixvpr.npy"):
        cache.path("day_one")


def test_build_split_honours_list_semantics(corpus, corpus6):
    cfgs = [_cfg(corpus, val_seq_list=["day_two"]),
            _cfg(corpus6, train_seq_list=None, val_seq_list=[])]
    train = build_split(cfgs, "train_seq_list", "megaloc.npy")
    val = build_split(cfgs, "val_seq_list", "megaloc.npy")
    assert len(train) == 240                     # both corpora
    assert val.sequence_names() == ["day_two"]   # second corpus sits it out


def test_duplicate_sequence_names_rejected(corpus):
    with pytest.raises(ValueError, match="more than one source"):
        build_split([_cfg(corpus), _cfg(corpus)], "train_seq_list", "megaloc.npy")


def test_corpora_must_agree_on_teacher_width(corpus, corpus6, tmp_path):
    import numpy as np
    for seq in ["m3ed_day_one", "m3ed_day_two", "m3ed_night_one"]:
        path = f"{corpus6['teacher_dir']}/{seq}/megaloc.npy"
        rows = np.load(path).shape[0]
        np.save(path, np.zeros((rows, 128), np.float16))
    with pytest.raises(ValueError, match="disagree on teacher width"):
        build_split([_cfg(corpus), _cfg(corpus6)], "train_seq_list", "megaloc.npy")
