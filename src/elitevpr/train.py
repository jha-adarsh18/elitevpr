"""Distil an event-native student from a cached, frozen RGB teacher.

One entry point. An arm is a loss: `loss=structural_kl` (the reported recipe),
`loss=feature_cosine`, `loss=sp_kd`. Nothing else differs between them.
"""

import math
import os
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from elitevpr.data import BlockSampler, build_split
from elitevpr.losses import build_loss
from elitevpr.models import EventViTStudent
from elitevpr.tracking import build_logger


def corpora(cfg):
    """The corpus configs, in name order. `~corpora.<name>` drops one."""
    if not cfg.get("corpora"):
        raise SystemExit("no corpora selected: see configs/config.yaml")
    return [cfg.corpora[key] for key in sorted(cfg.corpora)]


def night_sequences(dataset_cfgs, datasets):
    """Night traverses, checked against the sequences actually loaded.

    A typo'd name would otherwise be silently sampled as day.
    """
    listed = []
    for cfg in dataset_cfgs:
        listed.extend(cfg.get("night_sequences") or [])
    known = set()
    for dataset in datasets:
        known |= set(dataset.sequence_names())
    unknown = sorted(set(listed) - known)
    if unknown:
        raise ValueError(f"night_sequences not in the loaded splits: {unknown}")
    return listed


def run_epoch(model, criterion, loader, optimizer, scaler, scheduler, params,
              device, weight, amp, train_mode):
    model.train(train_mode)
    criterion.train(train_mode)
    total, steps = 0.0, 0

    pbar = tqdm(loader, desc="train" if train_mode else "val")
    for events, teacher, _ts in pbar:
        events = events.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            # Only the forward runs under autocast. Inside an autocast region
            # the loss's own matmuls would be re-cast to fp16, and a std or
            # softmax over a near-degenerate similarity matrix is exactly
            # where that becomes inf/nan.
            with torch.autocast(device_type="cuda", enabled=amp):
                descriptor = model(events)
            loss = weight * criterion(descriptor, teacher)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {steps}: {loss.item()}")

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total += loss.item()
        steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total / max(steps, 1)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = steps_per_epoch * int(cfg.epochs)
    warmup = max(1, int(round(float(cfg.warmup_frac) * total)))
    floor = float(cfg.min_lr_frac)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    print(f"LR schedule: {warmup} warmup steps then cosine to {floor:g}x over "
          f"{total} steps")
    return optim.lr_scheduler.LambdaLR(optimizer, factor)


def main_worker(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    if not torch.cuda.is_available() and not bool(cfg.allow_cpu):
        raise SystemExit(
            "CUDA is not available. Pass allow_cpu=true only to smoke-test "
            "the wiring -- 30 epochs on CPU is never what was intended."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.training.enable_amp) and device.type == "cuda"

    dataset_cfgs = corpora(cfg)
    train_ds = build_split(dataset_cfgs, "train_seq_list", cfg.teacher_file)
    val_ds = build_split(dataset_cfgs, "val_seq_list", cfg.teacher_file,
                         pair_stride_override=1)
    print(f"train pairs: {len(train_ds)}   val pairs: {len(val_ds)}")

    night = night_sequences(dataset_cfgs, [train_ds, val_ds])

    block = int(cfg.training.block_size)
    batch = int(cfg.training.batch_size)
    if batch % block:
        raise SystemExit(f"batch_size={batch} must be a multiple of "
                         f"block_size={block}")
    sampler = BlockSampler(train_ds, night, block=block,
                           night_frac=float(cfg.training.night_frac))

    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=True,
                          persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)

    model = EventViTStudent(
        backbone_name=cfg.model.backbone_name,
        teacher_dim=cfg.model.teacher_dim,
        num_patches=cfg.model.num_patches,
        img_size=cfg.model.img_size,
        in_channels=cfg.model.in_channels,
        pooling=cfg.model.pooling,
        pretrained=bool(cfg.model.pretrained),
    ).to(device)

    teacher_width = train_ds.teacher_dim
    criterion = build_loss(cfg.loss, student_dim=int(cfg.model.teacher_dim),
                           teacher_dim=teacher_width).to(device)

    # Loss parameters exist only for arms that need a projection head; they are
    # optimised with the model and dropped from the deployed checkpoint.
    params = list(model.parameters()) + list(criterion.parameters())
    lr = cfg.training.learning_rate * (batch / cfg.training.lr_base_batch_size) ** 0.5
    optimizer = optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    scheduler = (build_scheduler(optimizer, cfg.training, len(train_dl))
                 if cfg.training.use_scheduler else None)

    print("=" * 70)
    print(f"teacher   {cfg.teacher_file} ({teacher_width}-d)")
    # Night names may include validation traverses; report the pool the
    # sampler will actually draw from.
    night_in_train = set(night) & set(train_ds.sequence_names())
    print(f"corpora   {len(dataset_cfgs)}: {', '.join(sorted(cfg.corpora))} -- "
          f"{len(train_ds.sequence_names())} train sequences, "
          f"{len(night_in_train)} night")
    print(f"student   {cfg.model.backbone_name}, {cfg.model.teacher_dim}-d, "
          f"pooling={cfg.model.pooling}")
    print(f"loss      {cfg.loss.name} (weight {float(cfg.loss.weight):g})")
    print(f"batch     {batch} in blocks of {block}, "
          f"night_frac={float(cfg.training.night_frac):g} -- "
          f"{len(train_dl)} steps/epoch x {cfg.training.epochs} epochs")
    print(f"optimiser AdamW lr={lr:g}, amp={amp}")
    print("=" * 70)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    resume_path = checkpoint_dir / "resume.pth"

    start_epoch, best_val, patience, run_id = 0, float("inf"), 0, None
    if cfg.resume and resume_path.is_file():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        criterion.load_state_dict(state["criterion"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val, patience, run_id = state["best_val"], state["patience"], state["run_id"]
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    logger = build_logger(cfg.logging,
                          OmegaConf.to_container(cfg, resolve=True), run_id)
    weight = float(cfg.loss.weight)

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        print(f"\nepoch {epoch + 1}/{cfg.training.epochs}")
        train_loss = run_epoch(model, criterion, train_dl, optimizer, scaler,
                               scheduler, params, device, weight, amp, True)
        val_loss = run_epoch(model, criterion, val_dl, optimizer, None, None,
                             params, device, weight, amp, False)
        print(f"train {train_loss:.6f}   val {val_loss:.6f}")
        logger.log({"epoch": epoch + 1, "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), best_path)
            print(f"new best: {val_loss:.6f}")
        else:
            patience += 1

        # `last` is written every epoch and must be scored alongside `best`:
        # validation is DSEC-only and day-dominated, and `last` has beaten
        # `best` on day-night in every expert run so far.
        torch.save(model.state_dict(), last_path)
        _save_resume(resume_path, model, criterion, optimizer, scaler,
                     scheduler, epoch, best_val, patience, logger.run_id, cfg)

        if cfg.early_stopping > 0 and patience >= cfg.early_stopping:
            print(f"early stopping after epoch {epoch + 1}")
            break

    print(f"\ndone. best val {best_val:.6f}\n  {best_path}\n  {last_path}")
    print("  score BOTH checkpoints.")
    logger.finish()
    return {"best_val": best_val, "best": best_path, "last": last_path}


def _save_resume(path, model, criterion, optimizer, scaler, scheduler, epoch,
                 best_val, patience, run_id, cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "patience": patience,
        "run_id": run_id,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, tmp)
    os.replace(tmp, path)       # never leave a half-written checkpoint


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    main_worker(cfg)


if __name__ == "__main__":
    main()
