"""Plot training trajectories for both K562 enhancer runs.

Parses per-epoch lines from two SLURM logs and produces a two-panel
figure: train BCE on the left, val AUROC on the right. Best-checkpoint
epochs come from the "NEW BEST" lines.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_FROM_SCRATCH_LOG = (
    "/gpfs/data/zhou-lab/haoyanghu/logs/slurm.enh_train_from_scratch.10487856.log"
)
DEFAULT_PRETRAINED_LOG = (
    "/gpfs/data/zhou-lab/haoyanghu/logs/slurm.pretrained_enh_train.10492940.log"
)
DEFAULT_OUTPUT = "eval/loss_curves.png"

_EPOCH_RE = re.compile(
    r"""^\d\d:\d\d:\d\d\s+INFO\s+\[epoch\s+(?P<epoch>\d+)\]\s+
        train_bce=(?P<train_bce>[\d.]+)\s+
        val_bce=(?P<val_bce>[\d.]+)\s+
        val_auroc=(?P<val_auroc>[\d.]+)\s+
        val_auprc=(?P<val_auprc>[\d.]+)\s+
        pos_rate=(?P<pos_rate>[\d.]+)\s+
        lr=(?P<lr>[\d.eE+\-]+)\s+
        epoch_time=(?P<epoch_time>[\d.]+)s\s+
        nan_batches=(?P<nan_batches>\d+)\s+
        grad_skips=(?P<grad_skips>\d+)
    """,
    re.VERBOSE | re.MULTILINE,
)

_NEW_BEST_RE = re.compile(
    r"NEW BEST\s+val_bce=(?P<val_bce>[\d.]+)\s*\(epoch\s+(?P<epoch>\d+)\)"
)


log = logging.getLogger(__name__)


def parse_training_log(path: str | Path) -> list[dict]:
    """Parse a SLURM training log, returning one dict per epoch.

    Raises RuntimeError if zero epoch lines match (regex stale or log
    truncated).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training log not found: {p}")

    with open(p) as f:
        text = f.read()

    epochs: list[dict] = []
    for m in _EPOCH_RE.finditer(text):
        epochs.append({
            "epoch": int(m.group("epoch")),
            "train_bce": float(m.group("train_bce")),
            "val_bce": float(m.group("val_bce")),
            "val_auroc": float(m.group("val_auroc")),
            "val_auprc": float(m.group("val_auprc")),
            "pos_rate": float(m.group("pos_rate")),
            "lr": float(m.group("lr")),
            "epoch_time_s": float(m.group("epoch_time")),
            "nan_batches": int(m.group("nan_batches")),
            "grad_skips": int(m.group("grad_skips")),
            "is_new_best": False,
        })
    if not epochs:
        raise RuntimeError(
            f"Parsed zero epoch lines from {p} — regex broken or log truncated. "
            f"First 200 chars: {text[:200]!r}"
        )

    by_epoch = {e["epoch"]: e for e in epochs}
    n_new_best = 0
    for m in _NEW_BEST_RE.finditer(text):
        ep = int(m.group("epoch"))
        if ep in by_epoch:
            by_epoch[ep]["is_new_best"] = True
            n_new_best += 1
    log.info("  %s: %d epochs parsed, %d NEW BEST lines matched",
             p.name, len(epochs), n_new_best)
    return sorted(epochs, key=lambda e: e["epoch"])


def best_epoch(epochs: list[dict]) -> int | None:
    """Last 'NEW BEST' epoch (the val-selected best.pt), or argmin(val_bce)."""
    new_bests = [e for e in epochs if e["is_new_best"]]
    if new_bests:
        return new_bests[-1]["epoch"]
    return min(epochs, key=lambda e: e["val_bce"])["epoch"]


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figure(
    fs_epochs: list[dict],
    pt_epochs: list[dict],
    out_path: Path,
) -> None:
    """Two-panel figure: train BCE (left), val AUROC (right)."""
    plt = _matplotlib()

    fs_x = [e["epoch"] for e in fs_epochs]
    fs_train = [e["train_bce"] for e in fs_epochs]
    fs_auroc = [e["val_auroc"] for e in fs_epochs]
    pt_x = [e["epoch"] for e in pt_epochs]
    pt_train = [e["train_bce"] for e in pt_epochs]
    pt_auroc = [e["val_auroc"] for e in pt_epochs]

    fs_best = best_epoch(fs_epochs)
    pt_best = best_epoch(pt_epochs)

    fs_color = "tab:blue"
    pt_color = "tab:orange"

    max_x = max(fs_x + pt_x) if (fs_x or pt_x) else 0
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    axL.plot(fs_x, fs_train, "-o", color=fs_color, ms=4, lw=1.8,
             label=f"from_scratch (n={len(fs_x)} epochs)")
    axL.plot(pt_x, pt_train, "-o", color=pt_color, ms=4, lw=1.8,
             label=f"pretrained (n={len(pt_x)} epochs)")
    if fs_best is not None:
        axL.axvline(fs_best, ls="--", color=fs_color, alpha=0.5, lw=1)
        axL.annotate(f"from_scratch best @ ep {fs_best}",
                     xy=(fs_best, axL.get_ylim()[1]),
                     xytext=(fs_best + 0.5, axL.get_ylim()[1] * 0.95),
                     fontsize=8, color=fs_color, ha="left",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=fs_color, alpha=0.7))
    if pt_best is not None:
        axL.axvline(pt_best, ls="--", color=pt_color, alpha=0.5, lw=1)
        axL.annotate(f"pretrained best @ ep {pt_best}",
                     xy=(pt_best, axL.get_ylim()[1]),
                     xytext=(pt_best + 0.5, axL.get_ylim()[1] * 0.85),
                     fontsize=8, color=pt_color, ha="left",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=pt_color, alpha=0.7))
    axL.set_xlabel("Epoch")
    axL.set_ylabel("Training BCE loss")
    axL.set_title("Training loss")
    axL.set_xlim(-0.5, max_x + 0.5)
    axL.legend(loc="upper right", fontsize=9)
    axL.grid(True, alpha=0.3)

    axR.plot(fs_x, fs_auroc, "-o", color=fs_color, ms=4, lw=1.8,
             label="from_scratch")
    axR.plot(pt_x, pt_auroc, "-o", color=pt_color, ms=4, lw=1.8,
             label="pretrained")
    if fs_best is not None:
        axR.axvline(fs_best, ls="--", color=fs_color, alpha=0.5, lw=1)
        fs_best_auroc = next(
            (e["val_auroc"] for e in fs_epochs if e["epoch"] == fs_best), None,
        )
        if fs_best_auroc is not None:
            axR.plot([fs_best], [fs_best_auroc], "*", color=fs_color, ms=14,
                     zorder=5)
    if pt_best is not None:
        axR.axvline(pt_best, ls="--", color=pt_color, alpha=0.5, lw=1)
        pt_best_auroc = next(
            (e["val_auroc"] for e in pt_epochs if e["epoch"] == pt_best), None,
        )
        if pt_best_auroc is not None:
            axR.plot([pt_best], [pt_best_auroc], "*", color=pt_color, ms=14,
                     zorder=5)
    axR.set_xlabel("Epoch")
    axR.set_ylabel("Validation per-base AUROC")
    axR.set_title("Validation AUROC")
    axR.set_xlim(-0.5, max_x + 0.5)
    axR.set_ylim(0.5, 1.0)
    axR.legend(loc="lower right", fontsize=9)
    axR.grid(True, alpha=0.3)

    fig.suptitle("K562 enhancer prediction — training trajectories",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from-scratch-log", default=DEFAULT_FROM_SCRATCH_LOG,
                   help=f"SLURM log for from_scratch run "
                        f"(default {DEFAULT_FROM_SCRATCH_LOG}).")
    p.add_argument("--pretrained-log", default=DEFAULT_PRETRAINED_LOG,
                   help=f"SLURM log for pretrained run "
                        f"(default {DEFAULT_PRETRAINED_LOG}).")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output PNG path (default {DEFAULT_OUTPUT}).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    log.info("plot_loss_curves.py starting")
    log.info("  from_scratch_log : %s", args.from_scratch_log)
    log.info("  pretrained_log   : %s", args.pretrained_log)
    log.info("  output           : %s", args.output)

    fs = parse_training_log(args.from_scratch_log)
    pt = parse_training_log(args.pretrained_log)
    fs_best = best_epoch(fs)
    pt_best = best_epoch(pt)
    log.info("  from_scratch best epoch: %s (n_epochs=%d, last_train_bce=%.4f, "
             "last_val_auroc=%.4f)",
             fs_best, len(fs), fs[-1]["train_bce"], fs[-1]["val_auroc"])
    log.info("  pretrained   best epoch: %s (n_epochs=%d, last_train_bce=%.4f, "
             "last_val_auroc=%.4f)",
             pt_best, len(pt), pt[-1]["train_bce"], pt[-1]["val_auroc"])

    out_png = Path(args.output)
    make_figure(fs, pt, out_png)
    log.info("Wrote %s", out_png)

    sidecar = {
        "from_scratch": fs,
        "pretrained": pt,
        "from_scratch_best_epoch": fs_best,
        "pretrained_best_epoch": pt_best,
        "from_scratch_log": str(args.from_scratch_log),
        "pretrained_log": str(args.pretrained_log),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    json_path = out_png.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("Wrote %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
