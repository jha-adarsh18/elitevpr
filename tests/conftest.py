import numpy as np
import pytest
import torch

SEQUENCES = {
    "day_one": ("day", 40),
    "day_two": ("day", 40),
    "night_one": ("night", 40),
}
TEACHER_DIM = 64
HW = 32                      # the student is built at img_size=HW in tests


def _write_corpus(root, columns, prefix="", dim=TEACHER_DIM):
    """A synthetic corpus with the on-disk contract the real ones use."""
    root.mkdir(parents=True, exist_ok=True)
    features = root / "features"
    cache = root / "megaloc"
    lines = []
    rng = np.random.default_rng(0)

    for name, (_kind, n_frames) in SEQUENCES.items():
        seq = prefix + name
        (root / seq / "rgb").mkdir(parents=True, exist_ok=True)
        (root / seq / "events" / "histogram").mkdir(parents=True, exist_ok=True)
        (features / seq).mkdir(parents=True, exist_ok=True)
        (cache / seq).mkdir(parents=True, exist_ok=True)

        frames = []
        for i in range(n_frames):
            name = f"{seq}_{i:06d}.npy"
            np.save(root / seq / "events" / "histogram" / name,
                    rng.random((3, HW, HW)).astype(np.float16))
            np.save(root / seq / "rgb" / name, np.zeros((2, 2), np.uint8))
            frames.append(f"{seq}/rgb/{name}")
            cols = [str(1_000_000 * (i + 1)), f"rgb/{name}",
                    f"events/histogram/{name}"]
            if columns == 6:
                cols += [f"events/voxel/{name}",
                         f"events/timesurface_net/{name}",
                         f"events/timesurface_zero/{name}"]
            lines.append(",".join(cols))

        (features / seq / "frames.txt").write_text("\n".join(frames) + "\n")
        np.save(cache / seq / "megaloc.npy",
                rng.random((n_frames, dim)).astype(np.float16))

    (root / "pairs.txt").write_text("\n".join(lines) + "\n")
    return {"root_dir": str(root), "output_dir": str(features),
            "teacher_dir": str(cache)}


@pytest.fixture
def corpus(tmp_path):
    """3-column corpus (DSEC / ViViD contract)."""
    return _write_corpus(tmp_path / "corpus3", columns=3)


@pytest.fixture
def corpus6(tmp_path):
    """6-column corpus (M3ED contract)."""
    return _write_corpus(tmp_path / "corpus6", columns=6, prefix="m3ed_")


@pytest.fixture
def night_seqs():
    return [name for name, (kind, _) in SEQUENCES.items() if kind == "night"]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def smoke_root(tmp_path):
    """A corpus laid out exactly as configs/datasets/dsec.yaml expects it."""
    _write_corpus(tmp_path / "dsec", columns=3)
    return tmp_path
