import pytest
import torch

from elitevpr.data import BlockSampler, PairsDataset, TeacherCache


def _dataset(corpus):
    return PairsDataset(corpus["root_dir"], corpus["output_dir"],
                        TeacherCache(corpus["teacher_dir"], "megaloc.npy"))


def test_blocks_are_contiguous_in_time(corpus, night_seqs):
    ds = _dataset(corpus)
    sampler = BlockSampler(ds, night_seqs, block=8, night_frac=0.5,
                           generator=torch.Generator().manual_seed(0),
                           verbose=False)
    indices = list(sampler)
    assert len(indices) == len(sampler) == len(ds) // 8 * 8

    for start in range(0, len(indices), 8):
        block = indices[start:start + 8]
        pairs = [ds.pairs[i] for i in block]
        assert len({p["sequence"] for p in pairs}) == 1
        stamps = [p["timestamp_us"] for p in pairs]
        assert stamps == sorted(stamps)
        step = stamps[1] - stamps[0]
        assert all(b - a == step for a, b in zip(stamps, stamps[1:]))


def test_night_fraction_is_per_block(corpus, night_seqs):
    ds = _dataset(corpus)
    sampler = BlockSampler(ds, night_seqs, block=8, night_frac=0.5,
                           num_samples=8 * 2000,
                           generator=torch.Generator().manual_seed(0),
                           verbose=False)
    night = {name for name in night_seqs}
    blocks = list(sampler)
    kinds = [ds.pairs[blocks[i]]["sequence"] in night
             for i in range(0, len(blocks), 8)]
    assert 0.45 < sum(kinds) / len(kinds) < 0.55


def test_night_fraction_collapses_without_a_night_pool(corpus):
    sampler = BlockSampler(_dataset(corpus), [], block=8, night_frac=0.5,
                           verbose=False)
    assert sampler.night_frac == 0.0


def test_sequences_shorter_than_the_block_are_dropped(corpus, night_seqs):
    ds = _dataset(corpus)
    ds.pairs = [p for p in ds.pairs if p["sequence"] != "day_two"]
    sampler = BlockSampler(ds, night_seqs, block=8, night_frac=0.5,
                           verbose=False)
    assert len(sampler.day_groups) == 1 and len(sampler.night_groups) == 1

    ds.pairs = ds.pairs[:4]
    with pytest.raises(ValueError, match="at least block"):
        BlockSampler(ds, night_seqs, block=8, verbose=False)


def test_num_samples_sets_blocks_per_epoch(corpus, night_seqs):
    sampler = BlockSampler(_dataset(corpus), night_seqs, block=8,
                           num_samples=800, verbose=False)
    assert sampler.n_blocks == 100 and len(sampler) == 800
