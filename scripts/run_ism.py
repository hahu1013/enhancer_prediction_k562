"""Orchestrate ISM on the top-K positions of chr10.

Picks top-K spatially-deduplicated positions via select_top_positions,
runs compute_saliency, and writes saliency + reference base context to
a .npz for downstream motif extraction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# Path shim
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import PackbitsGenome  # noqa: E402
from src.ism import (  # noqa: E402
    select_top_positions,
    compute_saliency,
)
from src.model import EnhancerModel, EnhancerModelConfig  # noqa: E402


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_CHECKPOINT = f"{BASE}/projects/enhancer_prediction/runs/pretrained/best.pt"
DEFAULT_PREDICTIONS = (
    f"{BASE}/projects/enhancer_prediction/runs/pretrained/pred_chr10_best.npz"
)
DEFAULT_GENOME = f"{BASE}/data/reference/hg38.fa.gz"
DEFAULT_OUTPUT = (
    f"{BASE}/projects/enhancer_prediction/interpretation/ism_saliency_chr10.npz"
)


log = logging.getLogger(__name__)


# Model loader mirrors run_inference.load_model.
def load_model(ckpt_path: str, device: torch.device) -> EnhancerModel:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw
    model = EnhancerModel(EnhancerModelConfig()).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith(".num_batches_tracked")]
    real_unexpected = [k for k in unexpected
                        if not k.endswith(".num_batches_tracked")]
    if real_missing:
        log.warning("Non-BN missing keys: %s", real_missing)
    if real_unexpected:
        log.warning("Non-BN unexpected keys: %s", real_unexpected)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"Pretrained best.pt (default {DEFAULT_CHECKPOINT}).")
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                   help="Inference .npz used to pick top-K positions.")
    p.add_argument("--genome-path", default=DEFAULT_GENOME,
                   help=f"hg38 FASTA (default {DEFAULT_GENOME}).")
    p.add_argument("--chrom", default="chr10",
                   help="Chromosome (default chr10).")
    p.add_argument("--top-k", type=int, default=1000,
                   help="Number of focal positions to ISM (default 1000).")
    p.add_argument("--min-separation", type=int, default=200,
                   help="Minimum bp between selected positions (default 200).")
    p.add_argument("--window-half", type=int, default=25,
                   help="Half-width of the saliency window (default 25 → 51 bp).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Alt-sequence batch size per forward pass.")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output .npz path (default {DEFAULT_OUTPUT}).")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision inference.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info("run_ism.py starting")
    for k, v in vars(args).items():
        log.info("  %-16s = %s", k, v)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    log.info("device=%s amp=%s", device, use_amp)

    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %.2f M params", n_params / 1e6)

    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")
    data = np.load(args.predictions, allow_pickle=True)
    needed = {"enhancer_prob", "callable_mask"}
    if not needed.issubset(set(data.files)):
        raise ValueError(
            f"{args.predictions} missing keys: {sorted(needed - set(data.files))}"
        )
    probs = data["enhancer_prob"].astype(np.float32, copy=False)
    mask = data["callable_mask"].astype(np.uint8, copy=False)
    log.info("Loaded predictions: L=%d  positive_rate=%.6f  "
             "n_callable=%d (%.2f%%)  prob mean/median/max = %.4f/%.4f/%.4f",
             probs.shape[0],
             float((data["enhancer_label"][mask.astype(bool)] == 1).mean())
                if "enhancer_label" in data.files else float("nan"),
             int(mask.sum()),
             100.0 * mask.sum() / max(probs.shape[0], 1),
             float(probs.mean()), float(np.median(probs)), float(probs.max()))

    log.info("Opening genome at %s", args.genome_path)
    genome = PackbitsGenome(input_path=args.genome_path, storage="Packbits")
    chrom_lens = dict(genome.get_chr_lens())
    if args.chrom not in chrom_lens:
        raise ValueError(f"Chrom {args.chrom!r} not in genome")
    chrom_length = int(chrom_lens[args.chrom])
    log.info("[%s] length=%d", args.chrom, chrom_length)

    focals = select_top_positions(
        probs, mask, top_k=args.top_k, min_separation=args.min_separation,
    )
    if focals.size == 0:
        raise RuntimeError(
            "select_top_positions returned 0 positions — check mask + top_k."
        )

    t0 = time.time()
    focal_probs, saliency, ref_seqs = compute_saliency(
        model=model, genome=genome, chrom=args.chrom,
        chrom_length=chrom_length, focal_positions=focals,
        window_half=args.window_half, batch_size=args.batch_size,
        device=device, use_amp=use_amp,
    )
    elapsed = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": args.checkpoint,
        "predictions": args.predictions,
        "genome_path": args.genome_path,
        "chrom": args.chrom,
        "chrom_length": chrom_length,
        "top_k": int(args.top_k),
        "min_separation": int(args.min_separation),
        "window_half": int(args.window_half),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "amp": bool(use_amp),
        "seed": int(args.seed),
        "elapsed_seconds": float(elapsed),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_focals": int(len(focals)),
        "mean_focal_prob": float(focal_probs.mean()),
        "mean_saliency": float(saliency.mean()),
        "max_saliency": float(saliency.max()),
    }
    np.savez_compressed(
        out_path,
        focal_positions=focals.astype(np.int64, copy=False),
        focal_probs=focal_probs.astype(np.float32, copy=False),
        saliency=saliency.astype(np.float32, copy=False),
        sequences=ref_seqs.astype(np.uint8, copy=False),
        metadata=np.array(json.dumps(metadata), dtype=object),
    )
    sz_mb = out_path.stat().st_size / 1e6
    log.info("Wrote %s (%.1f MB)", out_path, sz_mb)
    log.info("=== ISM summary: K=%d  mean focal prob=%.4f  mean saliency=%.5f  "
             "max saliency=%.4f  runtime=%.1fs ===",
             len(focals), float(focal_probs.mean()), float(saliency.mean()),
             float(saliency.max()), elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
