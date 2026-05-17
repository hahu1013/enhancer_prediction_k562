"""Train the K562 enhancer prediction CNN.

``--init from_scratch`` uses random init; ``--init pretrained`` warm-starts
the backbone from puffin_D.pth via load_puffin_pretrained. Both runs share
all other training settings.

Loop: AdamW + ReduceLROnPlateau/CosineAnnealing, mixed-precision on GPU,
grad clipping at 1.0, masked BCEWithLogits with --pos-weight, auto-resume
from ``<out_dir>/last.pt``, per-epoch TSV log.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Path shim so the script can be run from anywhere.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    SamplerConfig,
    build_train_sampler,
    build_val_sampler,
)
from src.model import (  # noqa: E402
    EnhancerModel,
    EnhancerModelConfig,
    load_puffin_pretrained,
)


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_GENOME_PATH = f"{BASE}/data/reference/hg38.fa.gz"
DEFAULT_LABELS_DIR = f"{BASE}/projects/enhancer_prediction/data/labels"
DEFAULT_PUFFIN_WEIGHTS = f"{BASE}/projects/puffin/resources/puffin_D.pth"


# Split convention: train chr1-7 + chr11-20, val chr21, test chr22.
# chr8/9/10 excluded from modelling; chrX/chrY/chrM blacklisted.
TRAIN_CHROMS: Tuple[str, ...] = tuple(
    [f"chr{i}" for i in range(1, 8)]
    + [f"chr{i}" for i in range(11, 21)]
)
VAL_CHROMS: Tuple[str, ...] = ("chr21",)
TEST_CHROMS: Tuple[str, ...] = ("chr22",)
BLACKLIST_CHROMS: Tuple[str, ...] = ("chrY", "chrM")


@dataclass
class Config:
    # I/O paths
    output_dir: str
    init: str  # "from_scratch" | "pretrained"
    data_dir: str = DEFAULT_LABELS_DIR
    genome_path: str = DEFAULT_GENOME_PATH
    pretrained_path: str = DEFAULT_PUFFIN_WEIGHTS

    # Data split
    training_chroms: Tuple[str, ...] = field(default_factory=lambda: TRAIN_CHROMS)
    validation_chroms: Tuple[str, ...] = field(default_factory=lambda: VAL_CHROMS)
    test_chroms: Tuple[str, ...] = field(default_factory=lambda: TEST_CHROMS)
    blacklist_chroms: Tuple[str, ...] = field(default_factory=lambda: BLACKLIST_CHROMS)

    # Model + window
    sample_length: int = 100_000

    # Optimizer
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Schedule
    lr_schedule: str = "plateau"  # "plateau" | "cosine"
    lr_factor: float = 0.5
    lr_patience: int = 2          # epochs to wait without val-BCE improvement
    lr_min: float = 1e-6
    cosine_T_max: Optional[int] = None  # filled at runtime from total_steps

    # Loss
    pos_weight: float = 25.0

    # Sampler. Default 0.5 paired with uniform RandomPositions.
    min_callable_fraction: float = 0.5
    num_workers: int = 4

    # Batching / schedule
    batch_size: int = 4
    epochs: int = 20
    steps_per_epoch: int = 2_500    # ~50k total steps default; tunable.
    val_n_windows: int = 50         # mid-training val sweep size

    # Checkpointing / logging
    checkpoint_every: int = 1       # epochs
    log_every: int = 50             # steps (in-epoch train-loss heartbeat)

    # Misc
    seed: int = 42
    amp: bool = True                # mixed precision on GPU (auto-disabled on CPU)


log = logging.getLogger(__name__)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _capture_rng_states() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_states(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_bce_loss(
    logits: torch.Tensor,
    enhancer_label: torch.Tensor,
    callable_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Masked BCEWithLogits — only callable positions contribute."""
    loss_per_pos = F.binary_cross_entropy_with_logits(
        logits.squeeze(1),
        enhancer_label.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    return (loss_per_pos * callable_mask).sum() / callable_mask.sum().clamp(min=1)


def validate(
    model: torch.nn.Module,
    val_sampler,
    n_windows: int,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> dict:
    """Sweep ``n_windows`` val windows; report mean BCE, AUROC, AUPRC."""
    model.eval()
    logits_all = []
    labels_all = []
    masks_all = []

    losses = []
    with torch.no_grad():
        for _ in range(n_windows):
            seq_np, target_np = val_sampler.sample()
            seq = torch.from_numpy(seq_np[None]).to(device).float()
            enh = torch.from_numpy(target_np[0:1]).to(device).long()
            mask = torch.from_numpy(target_np[1:2]).to(device).float()

            logits = model(seq)
            loss = masked_bce_loss(logits, enh, mask, pos_weight).item()
            losses.append(loss)

            logits_all.append(logits.squeeze(1).cpu().numpy())
            labels_all.append(enh.cpu().numpy())
            masks_all.append(mask.cpu().numpy())

    logits_cat = np.concatenate(logits_all, axis=0).ravel().astype(np.float32)
    labels_cat = np.concatenate(labels_all, axis=0).ravel().astype(np.uint8)
    masks_cat = np.concatenate(masks_all, axis=0).ravel().astype(bool)

    n_valid = int(masks_cat.sum())
    valid_logits = logits_cat[masks_cat]
    valid_labels = labels_cat[masks_cat]

    auroc = float("nan")
    auprc = float("nan")
    pos_rate = float("nan")
    if n_valid > 0 and len(np.unique(valid_labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            valid_probs = 1.0 / (1.0 + np.exp(-valid_logits))
            auroc = float(roc_auc_score(valid_labels, valid_logits))
            auprc = float(average_precision_score(valid_labels, valid_probs))
            pos_rate = float(valid_labels.mean())
        except Exception as e:
            log.warning("sklearn metrics failed: %s", e)

    metrics = {
        "val/loss_bce": float(np.mean(losses)),
        "val/auroc_per_base": auroc,
        "val/auprc_per_base": auprc,
        "val/n_valid_positions": n_valid,
        "val/positive_rate": pos_rate,
    }
    model.train()
    return metrics


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    best_val_bce: float,
    best_val_epoch: int,
    rng_states: dict,
    cfg: Config,
    n_nan_batches: int = 0,
    n_skipped_grad_steps: int = 0,
) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict() if scaler is not None else None
        ),
        "best_val_bce": best_val_bce,
        "best_val_epoch": best_val_epoch,
        "rng_states": rng_states,
        "config": asdict(cfg),
        "n_nan_batches": n_nan_batches,
        "n_skipped_grad_steps": n_skipped_grad_steps,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def _save_or_check_config(cfg: Config, out_dir: Path) -> None:
    config_path = out_dir / "config.json"
    new = asdict(cfg)
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(new, f, indent=2, default=str)
        log.info("Saved config to %s", config_path)
    else:
        with open(config_path) as f:
            existing = json.load(f)
        if json.dumps(existing, sort_keys=True, default=str) != json.dumps(
            new, sort_keys=True, default=str
        ):
            log.warning(
                "Config differs from existing %s. This is OK if intentional "
                "(e.g. resuming with a different LR).",
                config_path,
            )


_TSV_HEADER = [
    "epoch", "step",
    "train_loss_mean", "val_loss_bce",
    "val_auroc_per_base", "val_auprc_per_base",
    "val_positive_rate", "val_n_valid_positions",
    "lr", "elapsed_s", "timestamp",
]


def _open_tsv_log(out_dir: Path) -> "csv.DictWriter":
    path = out_dir / "train_log.tsv"
    is_new = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_TSV_HEADER, delimiter="\t")
    if is_new:
        writer.writeheader()
        f.flush()
    return writer, f


def train(cfg: Config) -> None:
    _seed_all(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_or_check_config(cfg, out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    log.info("Device: %s (amp=%s)", device, use_amp)
    log.info("init           : %s", cfg.init)
    log.info("output_dir     : %s", out_dir)
    log.info("data_dir       : %s", cfg.data_dir)
    log.info("epochs         : %d  steps_per_epoch: %d  → total ~%d steps",
             cfg.epochs, cfg.steps_per_epoch, cfg.epochs * cfg.steps_per_epoch)
    log.info("batch_size     : %d", cfg.batch_size)
    log.info("lr             : %.2e  lr_schedule: %s  weight_decay: %.2e",
             cfg.lr, cfg.lr_schedule, cfg.weight_decay)
    log.info("pos_weight     : %.3f  (balanced ≈ %.1f given 1.24%% positive rate)",
             cfg.pos_weight, (1.0 - 0.0124) / 0.0124)
    log.info("min_callable_fraction: %.3f%s",
             cfg.min_callable_fraction,
             " (disabled)" if cfg.min_callable_fraction <= 0 else "")
    log.info("grad_clip      : %.2f  amp: %s", cfg.grad_clip, use_amp)
    log.info("train_chroms   : %s", list(cfg.training_chroms))
    log.info("val_chroms     : %s", list(cfg.validation_chroms))

    model = EnhancerModel(EnhancerModelConfig(seq_len=cfg.sample_length)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %.2f M", n_params / 1e6)

    if cfg.init == "pretrained":
        if not os.path.exists(cfg.pretrained_path):
            raise FileNotFoundError(
                f"--init pretrained but Puffin-D weights not found at "
                f"{cfg.pretrained_path}"
            )
        log.info("Warm-starting from %s ...", cfg.pretrained_path)
        model, info = load_puffin_pretrained(model, cfg.pretrained_path,
                                              map_location=device)
        log.info(
            "  loaded=%d  skipped_in_ckpt=%d  fresh_in_model=%d",
            info["n_loaded"], info["n_skipped_in_ckpt"], info["n_fresh_in_model"],
        )
        with open(out_dir / "warm_start_info.json", "w") as f:
            json.dump(info, f, indent=2)
    elif cfg.init == "from_scratch":
        log.info("Initializing from scratch (random PyTorch defaults).")
    else:
        raise ValueError(f"Unknown --init {cfg.init!r}; expected from_scratch | pretrained")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    if cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
        )
    elif cfg.lr_schedule == "cosine":
        T_max = cfg.cosine_T_max or cfg.epochs * cfg.steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=cfg.lr_min,
        )
    else:
        raise ValueError(f"Unknown --lr-schedule {cfg.lr_schedule!r}")

    scaler = None
    if use_amp:
        try:
            from torch.amp import GradScaler  # torch >= 2.4
            scaler = GradScaler()
        except ImportError:
            from torch.cuda.amp import GradScaler as _Old
            scaler = _Old()

    train_cfg = SamplerConfig(
        genome_path=cfg.genome_path,
        labels_dir=cfg.data_dir,
        training_chroms=cfg.training_chroms,
        validation_holdout=cfg.validation_chroms,
        test_holdout=cfg.test_chroms,
        sample_length=cfg.sample_length,
        seed=cfg.seed,
        blacklist_chroms=cfg.blacklist_chroms,
        min_callable_fraction=cfg.min_callable_fraction,
    )
    val_cfg = SamplerConfig(
        genome_path=cfg.genome_path,
        labels_dir=cfg.data_dir,
        training_chroms=cfg.training_chroms,
        validation_holdout=cfg.validation_chroms,
        test_holdout=cfg.test_chroms,
        sample_length=cfg.sample_length,
        seed=cfg.seed + 1,
        blacklist_chroms=cfg.blacklist_chroms,
        min_callable_fraction=cfg.min_callable_fraction,
    )

    train_sampler = build_train_sampler(train_cfg)
    val_sampler = build_val_sampler(val_cfg)

    from selene_mini.dataloaders import SamplerDataLoader

    loader = SamplerDataLoader(
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        pin_memory=(device.type == "cuda"),
    )

    last_path = out_dir / "last.pt"
    if last_path.exists():
        log.info("Resuming from %s", last_path)
        ckpt = load_checkpoint(
            str(last_path), model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
        )
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["step"])
        best_val_bce = float(ckpt["best_val_bce"])
        best_val_epoch = int(ckpt["best_val_epoch"])
        _restore_rng_states(ckpt["rng_states"])
        log.info("  resumed: start_epoch=%d global_step=%d best_val_bce=%.4f (epoch %d)",
                 start_epoch, global_step, best_val_bce, best_val_epoch)
    else:
        start_epoch = 0
        global_step = 0
        best_val_bce = float("inf")
        best_val_epoch = -1

    if start_epoch >= cfg.epochs:
        log.info("Already at/past epochs=%d — nothing to do.", cfg.epochs)
        return

    tsv_writer, tsv_file = _open_tsv_log(out_dir)

    pos_weight_t = torch.tensor(cfg.pos_weight, device=device)

    model.train()
    data_iter = iter(loader)
    loop_t0 = time.time()

    # Non-finite loss / gradient guards. AMP fp16 overflow inside
    # BCEWithLogitsLoss at high pos_weight can produce NaN losses; NaN
    # gradients reaching optimizer.step corrupt the weights via Adam's
    # moment buffers. Skip the batch in either case; abort after 100.
    n_nan_batches = 0
    n_skipped_grad_steps = 0
    if last_path.exists():
        try:
            ckpt_resumed = torch.load(last_path, map_location="cpu",
                                       weights_only=False)
            n_nan_batches = int(ckpt_resumed.get("n_nan_batches", 0))
            n_skipped_grad_steps = int(
                ckpt_resumed.get("n_skipped_grad_steps", 0)
            )
            del ckpt_resumed
        except Exception:
            pass

    for epoch in range(start_epoch, cfg.epochs):
        epoch_t0 = time.time()
        train_losses = []

        for step_in_epoch in range(cfg.steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            # batch = (seq (B, 4, L) float32, target (B, 2, L) uint8 with
            # channel 0 = enhancer_label, channel 1 = callable_mask).
            seq, target = batch
            seq = seq.to(device, non_blocking=True).float()
            enh = target[:, 0].to(device, non_blocking=True).long()
            mask = target[:, 1].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                from torch.amp import autocast
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(seq)
                # BCEWithLogitsLoss in fp32: with high pos_weight the
                # per-position loss can exceed fp16 range and produce NaNs.
                loss = masked_bce_loss(logits.float(), enh, mask, pos_weight_t)
            else:
                logits = model(seq)
                loss = masked_bce_loss(logits, enh, mask, pos_weight_t)

            # Skip BEFORE backward — a NaN loss would otherwise corrupt
            # the model state via backward + optimizer.step.
            if not torch.isfinite(loss):
                n_nan_batches += 1
                log.warning(
                    "Skipping non-finite training loss at epoch=%d step=%d "
                    "(cumulative count=%d)",
                    epoch, global_step, n_nan_batches,
                )
                if n_nan_batches > 100:
                    raise RuntimeError(
                        f"Too many non-finite losses ({n_nan_batches}); "
                        f"aborting."
                    )
                continue

            if use_amp:
                scaler.scale(loss).backward()
                # Unscale BEFORE the finiteness check below.
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # Inspect every grad explicitly: scaler.step alone is not
            # sufficient — transient inf/NaN can still slip through under
            # heavy class imbalance, and a single NaN gradient at
            # optimizer.step corrupts every weight via Adam moments.
            grads_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                n_skipped_grad_steps += 1
                log.warning(
                    "Non-finite gradient at epoch=%d step=%d "
                    "(cumulative count=%d); zeroing grads and skipping "
                    "optimizer step",
                    epoch, global_step, n_skipped_grad_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    # Tell scaler to decrease the loss scale next step.
                    scaler.update()
                if n_skipped_grad_steps > 100:
                    raise RuntimeError(
                        f"Too many non-finite gradient updates "
                        f"({n_skipped_grad_steps}); aborting."
                    )
                continue

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.grad_clip,
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            # Cosine scheduler steps per-step; plateau steps per-epoch.
            if cfg.lr_schedule == "cosine":
                scheduler.step()

            if (step_in_epoch + 1) % cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                recent_loss = float(np.mean(train_losses[-cfg.log_every:]))
                log.info(
                    "  epoch %d  step %d/%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                    epoch, step_in_epoch + 1, cfg.steps_per_epoch,
                    recent_loss, lr, time.time() - epoch_t0,
                )

        val_metrics = validate(
            model=model,
            val_sampler=val_sampler,
            n_windows=cfg.val_n_windows,
            pos_weight=pos_weight_t,
            device=device,
        )
        train_loss_mean = float(np.mean(train_losses))
        epoch_elapsed = time.time() - epoch_t0
        lr_now = optimizer.param_groups[0]["lr"]

        log.info(
            "[epoch %d] train_bce=%.4f  val_bce=%.4f  val_auroc=%.4f  "
            "val_auprc=%.4f  pos_rate=%.4f  lr=%.2e  epoch_time=%.1fs  "
            "nan_batches=%d  grad_skips=%d",
            epoch, train_loss_mean, val_metrics["val/loss_bce"],
            val_metrics["val/auroc_per_base"], val_metrics["val/auprc_per_base"],
            val_metrics["val/positive_rate"], lr_now, epoch_elapsed,
            n_nan_batches, n_skipped_grad_steps,
        )

        if cfg.lr_schedule == "plateau":
            scheduler.step(val_metrics["val/loss_bce"])

        tsv_writer.writerow({
            "epoch": epoch,
            "step": global_step,
            "train_loss_mean": train_loss_mean,
            "val_loss_bce": val_metrics["val/loss_bce"],
            "val_auroc_per_base": val_metrics["val/auroc_per_base"],
            "val_auprc_per_base": val_metrics["val/auprc_per_base"],
            "val_positive_rate": val_metrics["val/positive_rate"],
            "val_n_valid_positions": val_metrics["val/n_valid_positions"],
            "lr": lr_now,
            "elapsed_s": epoch_elapsed,
            "timestamp": timestamp(),
        })
        tsv_file.flush()

        if (epoch + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                str(out_dir / "last.pt"),
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, step=global_step,
                best_val_bce=best_val_bce, best_val_epoch=best_val_epoch,
                rng_states=_capture_rng_states(), cfg=cfg,
                n_nan_batches=n_nan_batches,
                n_skipped_grad_steps=n_skipped_grad_steps,
            )

        if val_metrics["val/loss_bce"] < best_val_bce:
            best_val_bce = float(val_metrics["val/loss_bce"])
            best_val_epoch = epoch
            save_checkpoint(
                str(out_dir / "best.pt"),
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, epoch=epoch, step=global_step,
                best_val_bce=best_val_bce, best_val_epoch=best_val_epoch,
                rng_states=_capture_rng_states(), cfg=cfg,
                n_nan_batches=n_nan_batches,
                n_skipped_grad_steps=n_skipped_grad_steps,
            )
            log.info("  NEW BEST val_bce=%.4f (epoch %d) — saved best.pt",
                     best_val_bce, best_val_epoch)

    tsv_file.close()
    final = {
        "completed_epochs": cfg.epochs,
        "final_global_step": global_step,
        "best_val_bce": best_val_bce,
        "best_val_epoch": best_val_epoch,
        "n_nan_batches": n_nan_batches,
        "n_skipped_grad_steps": n_skipped_grad_steps,
        "total_elapsed_s": time.time() - loop_t0,
        "config": asdict(cfg),
    }
    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    log.info("Training complete. best_val_bce=%.4f at epoch %d. Wrote %s",
             best_val_bce, best_val_epoch, out_dir / "final_summary.json")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--init", required=True,
                   choices=["from_scratch", "pretrained"],
                   help="Random init or Puffin-D warm-start.")
    p.add_argument("--output-dir", required=True,
                   help="Run dir; auto-resumes from <output-dir>/last.pt.")

    p.add_argument("--data-dir", default=DEFAULT_LABELS_DIR,
                   help="Directory with per-chrom label .npy files.")
    p.add_argument("--genome-path", default=DEFAULT_GENOME_PATH,
                   help="hg38 FASTA path.")
    p.add_argument("--pretrained-path", default=DEFAULT_PUFFIN_WEIGHTS,
                   help="Puffin-D weights for --init pretrained.")

    p.add_argument("--lr", type=float, default=2e-3,
                   help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Global gradient norm clip; 0 disables.")
    p.add_argument("--lr-schedule", default="plateau",
                   choices=["plateau", "cosine"],
                   help="ReduceLROnPlateau on val BCE, or CosineAnnealingLR.")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="Plateau LR multiplicative factor.")
    p.add_argument("--lr-patience", type=int, default=2,
                   help="Plateau epochs without improvement before LR drop.")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Floor LR.")

    p.add_argument("--pos-weight", type=float, default=25.0,
                   help="BCEWithLogits pos_weight (balanced ≈ 80 at 1.24%% pos).")

    p.add_argument("--min-callable-fraction", type=float, default=0.5,
                   help="Drop training windows below this callable fraction; "
                        "0 disables.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="SamplerDataLoader worker processes.")

    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size.")
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of training epochs.")
    p.add_argument("--steps-per-epoch", type=int, default=2_500,
                   help="Steps per epoch.")
    p.add_argument("--val-n-windows", type=int, default=50,
                   help="Val windows sampled per epoch.")

    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save last.pt every N epochs.")
    p.add_argument("--log-every", type=int, default=50,
                   help="In-epoch train-loss heartbeat every N steps.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    cfg = Config(
        output_dir=args.output_dir,
        init=args.init,
        data_dir=args.data_dir,
        genome_path=args.genome_path,
        pretrained_path=args.pretrained_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_schedule=args.lr_schedule,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        pos_weight=args.pos_weight,
        min_callable_fraction=args.min_callable_fraction,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_n_windows=args.val_n_windows,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        seed=args.seed,
        amp=not args.no_amp,
    )

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
