"""Tiled inference for the K562 enhancer model.

Sweeps one chromosome with overlapping 100 kb windows, applies sigmoid,
averages predictions in overlap regions, and writes a per-base prediction
.npz with ``enhancer_prob``, ``enhancer_label``, ``callable_mask``, ``metadata``.
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


# Path shim so the script can be run from anywhere.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    PackbitsGenome,
    iter_chrom_windows,
)
from src.model import EnhancerModel, EnhancerModelConfig  # noqa: E402


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_GENOME_PATH = f"{BASE}/data/reference/hg38.fa.gz"
DEFAULT_LABELS_DIR = f"{BASE}/projects/enhancer_prediction/data/labels"


log = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> EnhancerModel:
    """Build EnhancerModel and load weights from ``checkpoint_path``.

    Accepts raw state_dict / ``{"model_state_dict": ...}`` /
    ``{"model": ...}``. Loads with strict=False; benign BN counter
    drift is filtered before warning.
    """
    log.info("Loading checkpoint: %s", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
        ckpt_meta = {k: raw.get(k) for k in ("epoch", "step", "best_val_bce",
                                              "best_val_epoch")
                     if k in raw}
    elif isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
        ckpt_meta = {}
    else:
        state = raw
        ckpt_meta = {}

    model = EnhancerModel(EnhancerModelConfig()).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Filter benign BN counter buffers; anything else is real.
    real_missing = [k for k in missing if not k.endswith(".num_batches_tracked")]
    real_unexpected = [
        k for k in unexpected if not k.endswith(".num_batches_tracked")
    ]
    if real_missing:
        log.warning("Non-BN missing keys (model has, ckpt lacks): %s", real_missing)
    if real_unexpected:
        log.warning("Non-BN unexpected keys (ckpt has, model lacks): %s",
                    real_unexpected)
    log.info("  loaded %d keys; %d BN counters skipped; checkpoint meta=%s",
             len(state.keys() & model.state_dict().keys()),
             len(missing) - len(real_missing),
             ckpt_meta)

    model.eval()
    return model, ckpt_meta


def fetch_padded_sequence(
    genome: PackbitsGenome,
    chrom: str,
    chrom_length: int,
    start: int,
    end: int,
) -> np.ndarray:
    """(4, end-start) float32 one-hot, N-padded past chrom edges."""
    L = end - start
    out = np.full((4, L), 0.25, dtype=np.float32)
    clip_start = max(start, 0)
    clip_end = min(end, chrom_length)
    if clip_start < clip_end:
        chunk = genome.get(chrom, clip_start, clip_end)  # (4, n_real)
        out[:, clip_start - start : clip_end - start] = chunk
    return out


def run_inference(
    model: EnhancerModel,
    genome: PackbitsGenome,
    chrom: str,
    chrom_length: int,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Tile ``chrom`` and return per-base P(enhancer), averaged over overlaps."""
    tiles = list(iter_chrom_windows(chrom_length, window=window_size, stride=stride))
    log.info("[%s] L=%d → %d tiles (window=%d, stride=%d, batch=%d)",
             chrom, chrom_length, len(tiles), window_size, stride, batch_size)

    sum_arr = np.zeros(chrom_length, dtype=np.float32)
    count_arr = np.zeros(chrom_length, dtype=np.int32)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else _NullCtx()
    )

    t0 = time.time()
    last_log_step = 0
    log_every = max(1, len(tiles) // 10)

    with torch.no_grad():
        for batch_start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[batch_start : batch_start + batch_size]
            seqs = np.stack(
                [fetch_padded_sequence(genome, chrom, chrom_length, s, e)
                 for s, e in batch_tiles],
                axis=0,
            )                                                        # (B, 4, win)
            seqs_t = torch.from_numpy(seqs).to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(seqs_t)
            probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()

            for i, (start, end) in enumerate(batch_tiles):
                write_end = min(end, chrom_length)
                n_keep = write_end - start
                if n_keep <= 0:
                    continue
                sum_arr[start:write_end] += probs[i, :n_keep]
                count_arr[start:write_end] += 1

            done = batch_start + len(batch_tiles)
            if done - last_log_step >= log_every or done == len(tiles):
                last_log_step = done
                log.info("  [%s] %d/%d tiles  elapsed=%.1fs",
                         chrom, done, len(tiles), time.time() - t0)

    # Cast denominator to float32 so the result stays float32 — otherwise
    # (float32 / int32) → float64 and doubles chrom-length memory.
    denom = np.maximum(count_arr, 1).astype(np.float32, copy=False)
    enhancer_prob = (sum_arr / denom).astype(np.float32, copy=False)
    n_uncovered = int((count_arr == 0).sum())
    if n_uncovered > 0:
        log.warning("%d positions had zero coverage (set to 0.5 prior)",
                    n_uncovered)
        enhancer_prob[count_arr == 0] = 0.5
    log.info("[%s] inference done in %.1fs", chrom, time.time() - t0)
    return enhancer_prob


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def load_chrom_labels(labels_dir: str, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``{labels_dir}/{chrom}.npy`` → ``(enhancer_label, callable_mask)``."""
    path = Path(labels_dir) / f"{chrom}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Label array not found: {path}")
    arr = np.load(path, mmap_mode=None)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.dtype != np.uint8:
        raise ValueError(
            f"Unexpected label array shape/dtype at {path}: "
            f"got shape={arr.shape}, dtype={arr.dtype}; expected (L, 2) uint8"
        )
    return arr[:, 0].copy(), arr[:, 1].copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt checkpoint to evaluate.")
    parser.add_argument("--output", required=True,
                        help="Path to write the prediction .npz.")
    parser.add_argument("--chrom", default="chr10",
                        help="Chromosome to run inference on.")
    parser.add_argument("--data-dir", default=DEFAULT_LABELS_DIR,
                        help="Phase-1 label directory.")
    parser.add_argument("--genome-path", default=DEFAULT_GENOME_PATH,
                        help="hg38 FASTA path.")
    parser.add_argument("--stride", type=int, default=50_000,
                        help="Window stride in bp (default 50000 = 50%% overlap).")
    parser.add_argument("--window-size", type=int, default=100_000,
                        help="Window size in bp (= model input length).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision inference.")
    parser.add_argument("--device", default=None,
                        help="Torch device override (default cuda/cpu).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = (device.type == "cuda") and not args.no_amp
    log.info("device=%s amp=%s", device, use_amp)
    log.info("checkpoint=%s", args.checkpoint)
    log.info("output=%s", args.output)
    log.info("chrom=%s stride=%d window_size=%d batch_size=%d",
             args.chrom, args.stride, args.window_size, args.batch_size)
    log.info("data_dir=%s", args.data_dir)
    log.info("genome_path=%s", args.genome_path)

    model, ckpt_meta = load_model(args.checkpoint, device)

    log.info("Opening genome at %s", args.genome_path)
    genome = PackbitsGenome(input_path=args.genome_path, storage="Packbits")
    chrom_lens = dict(genome.get_chr_lens())
    if args.chrom not in chrom_lens:
        raise ValueError(
            f"Chromosome {args.chrom!r} not found in genome at {args.genome_path}. "
            f"Available: {list(chrom_lens.keys())[:10]}..."
        )
    chrom_length = int(chrom_lens[args.chrom])

    enhancer_label, callable_mask = load_chrom_labels(args.data_dir, args.chrom)
    if enhancer_label.shape[0] != chrom_length:
        log.warning(
            "Label array length %d != genome length %d for %s — output "
            "alignment may be off. Saving what we have.",
            enhancer_label.shape[0], chrom_length, args.chrom,
        )
    log.info("[%s] label stats: positive=%d (%.4f%%) callable=%d (%.2f%%)",
             args.chrom, int(enhancer_label.sum()),
             100 * enhancer_label.sum() / max(chrom_length, 1),
             int(callable_mask.sum()),
             100 * callable_mask.sum() / max(chrom_length, 1))

    enhancer_prob = run_inference(
        model=model, genome=genome,
        chrom=args.chrom, chrom_length=chrom_length,
        window_size=args.window_size, stride=args.stride,
        batch_size=args.batch_size,
        device=device, use_amp=use_amp,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "chrom": args.chrom,
        "chrom_length": chrom_length,
        "stride": int(args.stride),
        "window_size": int(args.window_size),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "amp": bool(use_amp),
        "data_dir": str(args.data_dir),
        "genome_path": str(args.genome_path),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_meta": {k: (v if isinstance(v, (int, float, str, bool))
                                else str(v))
                            for k, v in ckpt_meta.items()},
    }
    # 0-d numpy object scalar so np.load preserves the dict.
    metadata_arr = np.array(json.dumps(metadata), dtype=object)
    np.savez_compressed(
        out_path,
        enhancer_prob=enhancer_prob.astype(np.float32, copy=False),
        enhancer_label=enhancer_label.astype(np.uint8, copy=False),
        callable_mask=callable_mask.astype(np.uint8, copy=False),
        metadata=metadata_arr,
    )
    sz_mb = out_path.stat().st_size / 1e6
    log.info("Wrote %s (%.1f MB)", out_path, sz_mb)

    mask_b = callable_mask.astype(bool)
    if mask_b.any():
        cp = enhancer_prob[mask_b]
        log.info("[%s] over callable positions: prob mean=%.4f  median=%.4f  "
                 ">0.5 frac=%.4f  positive rate=%.4f",
                 args.chrom, float(cp.mean()), float(np.median(cp)),
                 float((cp > 0.5).mean()),
                 float(enhancer_label[mask_b].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
