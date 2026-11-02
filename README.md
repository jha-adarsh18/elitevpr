# E-LiteVPR

An event-native ViT-S student (22 M params, 1024-d) distilled from a frozen,
VPR-trained RGB teacher. The student sees only a 3-channel event histogram; the
teacher sees the time-aligned RGB frame and its descriptors are cached once.
The training signal is the pairwise geometry of the teacher's embedding within
a batch, so the recipe needs no GPS and no place labels. The deployed model is
the backbone, its per-patch projection and the pooling — nothing else.

## Install

```bash
pixi install
pixi run test
```

## Train

One entry point; an arm is a loss and nothing else changes.

```bash
# the reported recipe: relational KL, three corpora, MegaLoc teacher
pixi run train data_root=/data checkpoint_dir=/runs/structural_kl

# feature regression:  L = 1/B sum_i ( 1 - cos(P(s_i), t_i) )
pixi run train loss=feature_cosine data_root=/data checkpoint_dir=/runs/feature_cosine

# similarity-preserving KD (Tung & Mori):  L = 1/B^2 || G_s - G_t ||_F^2
pixi run train loss=sp_kd data_root=/data checkpoint_dir=/runs/sp_kd
```

`data_root` is expected to hold, per corpus:

```
<data_root>/<corpus>/pairs.txt                     master pairs file
<data_root>/<corpus>/<seq>/events/histogram/*.npy  float16 (3, H, W)
<data_root>/<corpus>/<seq>/rgb/*.npy
<data_root>/<corpus>/features/<seq>/frames.txt     fixes the cache row order
<data_root>/<corpus>/megaloc/<seq>/megaloc.npy     float16 (N, 8448)
```

Any corpus whose layout differs overrides its own keys, e.g.
`corpora.m3ed.teacher_dir=/mnt/cache/megaloc`. Drop a corpus with
`~corpora.vivid`; add one with `+datasets@corpora.<name>=<name>`.

**Score both checkpoints.** `best.pth` is selected on validation loss, which is
DSEC-only and day-dominated; `last.pth` has beaten it on day–night in every run
so far.

## Running from the Kaggle-built caches

The teacher descriptors and the `frames.txt` trees were built on Kaggle and do
not follow the `data_root` convention: DSEC and M3ED **share one** MegaLoc
cache directory while keeping separate `frames.txt` trees. Point each corpus at
its own pair of directories instead of setting `data_root`:

```bash
KG=/kaggle/input       # or wherever the pulled copies landed on the box
MEGALOC=$KG/notebooks/jayeshkumar888/dsec-m3ed/feats/megaloc

pixi run train \
  corpora.dsec.root_dir=$KG/datasets/jhag18/dsec-all-clean \
  corpora.dsec.output_dir=$KG/notebooks/jhag18/dsec-all/teacher_features \
  corpora.dsec.teacher_dir=$MEGALOC \
  corpora.m3ed.root_dir=$KG/datasets/jhag18/m3ed-processed \
  corpora.m3ed.output_dir=$KG/notebooks/jayeshkumar888/dsec-m3ed/feats/m3ed \
  corpora.m3ed.teacher_dir=$MEGALOC \
  corpora.vivid.root_dir=/data/vivid \
  corpora.vivid.output_dir=/data/vivid/megaloc_features \
  corpora.vivid.teacher_dir=/data/vivid/megaloc_features \
  checkpoint_dir=/runs/structural_kl
```

`data_root` may be left unset once every corpus names its own directories.

Each corpus resolves its own cache, so the multi-root comma list — and Hydra's
sweep-on-bare-comma trap — is gone. A missing descriptor names every directory
it looked in, and a cache whose row count disagrees with that corpus's
`frames.txt` stops the run before the first step.

ViViD is the one corpus whose MegaLoc cache was written to `/kaggle/working`
rather than to a dataset mount: confirm it survived in the notebook's saved
output before counting on it (`~corpora.vivid` is the two-corpus arm).

## Layout

```
src/elitevpr/
  data/      pairs dataset, teacher cache, block sampler
  models/    student, pooling families
  losses/    one module per arm + the registry
  train.py   the only entry point
configs/     hydra groups: corpora, model, training, loss, logging
tests/       parity against the reference pipeline + end-to-end CPU runs
```

## What is load-bearing

* **The temperature is not scale-free.** The relational loss drops the
  self-term and z-scores both similarity matrices, so `temperature` is in units
  of σ and 1.0 is the calibrated value. One shared τ=0.05 on raw similarities
  gave MegaLoc a gradient norm of 2.1e-09 — dead.
* **Contiguous blocks, not random batches.** A random sampler leaves 0.0007 of
  the relational target's mass off the diagonal, spread over 24.8 effective
  neighbours; blocks of 8 leave 0.0659 over 3.2.
* **`pooling=signed`.** Stock GeM's `clamp` truncates the signed projection
  output: 62–70% of token entries deleted, zero gradient through them, every
  descriptor in the positive orthant (effective rank 1.10 of 1024).
* **Only the forward runs under autocast.** A std or softmax over a
  near-degenerate similarity matrix is where fp16 becomes inf/nan.
* **Seed variance.** Brisbane day–night σ = 4.71, NSAVP σ = 1.65 over 3 seeds.
  Nothing under ~2σ is a result.
