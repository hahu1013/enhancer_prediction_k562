"""In-silico mutagenesis (ISM) saliency for the K562 enhancer model.

Per-base saliency = mean over the 3 non-reference alts of
``max(0, p_ref - p_alt)``. The clamp keeps only constructive signal.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import torch


log = logging.getLogger(__name__)


BASES = ("A", "C", "G", "T")
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_BASE = {i: b for b, i in BASE_TO_IDX.items()}


def select_top_positions(
    probs: np.ndarray,
    mask: np.ndarray,
    top_k: int,
    min_separation: int,
) -> np.ndarray:
    """Greedy top-k callable positions, sorted by probs desc, with
    ``min_separation`` bp spacing between accepted positions.

    Returns a sorted int64 array — length may be < top_k if the chromosome
    runs out of widely-separated callable peaks.
    """
    if probs.ndim != 1 or mask.ndim != 1 or probs.shape != mask.shape:
        raise ValueError(
            f"probs and mask must be 1D arrays of equal length; "
            f"got probs={probs.shape}, mask={mask.shape}"
        )
    if top_k <= 0 or min_separation < 0:
        raise ValueError(
            f"top_k must be > 0 (got {top_k}) and min_separation >= 0 "
            f"(got {min_separation})"
        )

    callable_idx = np.where(mask.astype(bool))[0]
    if len(callable_idx) == 0:
        log.warning("select_top_positions: no callable positions")
        return np.empty(0, dtype=np.int64)
    sub_probs = probs[callable_idx]
    order = np.argsort(-sub_probs, kind="stable")
    candidates = callable_idx[order].astype(np.int64)

    accepted = np.empty(top_k, dtype=np.int64)
    n_accepted = 0
    sorted_accepted: list[int] = []  # always sorted ascending

    for pos in candidates:
        pos_int = int(pos)
        if not sorted_accepted:
            sorted_accepted.append(pos_int)
            accepted[n_accepted] = pos_int
            n_accepted += 1
            if n_accepted == top_k:
                break
            continue
        i = np.searchsorted(sorted_accepted, pos_int)
        too_close = False
        if i > 0 and pos_int - sorted_accepted[i - 1] < min_separation:
            too_close = True
        if (not too_close
                and i < len(sorted_accepted)
                and sorted_accepted[i] - pos_int < min_separation):
            too_close = True
        if too_close:
            continue
        sorted_accepted.insert(i, pos_int)
        accepted[n_accepted] = pos_int
        n_accepted += 1
        if n_accepted == top_k:
            break

    out = np.array(sorted(accepted[:n_accepted].tolist()), dtype=np.int64)
    log.info(
        "select_top_positions: requested top_k=%d min_separation=%d; "
        "accepted %d / %d candidate(s); range=[%d, %d]",
        top_k, min_separation, n_accepted, len(candidates),
        int(out.min()) if out.size else -1,
        int(out.max()) if out.size else -1,
    )
    return out


def _fetch_window(
    genome, chrom: str, chrom_length: int, start: int, end: int,
) -> np.ndarray:
    """(4, end-start) float32 one-hot, N-padded with 0.25 past chrom edges."""
    L = end - start
    out = np.full((4, L), 0.25, dtype=np.float32)
    clip_s = max(start, 0)
    clip_e = min(end, chrom_length)
    if clip_s < clip_e:
        chunk = genome.get(chrom, clip_s, clip_e)  # (4, n_real)
        out[:, clip_s - start : clip_e - start] = chunk
    return out


def _seq_to_int(seq_one_hot: np.ndarray) -> np.ndarray:
    """(4, L) float32 one-hot → (L,) uint8 base index. N maps to 0."""
    return np.argmax(seq_one_hot, axis=0).astype(np.uint8)


def compute_saliency(
    model: torch.nn.Module,
    genome,
    chrom: str,
    chrom_length: int,
    focal_positions: np.ndarray,
    window_half: int = 25,
    batch_size: int = 64,
    device: torch.device | None = None,
    use_amp: bool = True,
    window_size: int = 100_000,
    log_every: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-base saliency for each focal position.

    Returns ``(focal_probs, saliency, ref_seqs)`` where
        focal_probs : (N,) float32         — p_ref at each focal
        saliency    : (N, 2*window_half+1) float32
        ref_seqs    : (N, 2*window_half+1) uint8 base index (0=A..3=T)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    win = window_size
    half = window_half
    saliency_w = 2 * half + 1
    n_focal = len(focal_positions)
    focal_probs = np.zeros(n_focal, dtype=np.float32)
    saliency = np.zeros((n_focal, saliency_w), dtype=np.float32)
    ref_seqs = np.zeros((n_focal, saliency_w), dtype=np.uint8)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else _NullCtx()
    )

    t0 = time.time()
    with torch.no_grad():
        for fi, focal in enumerate(focal_positions):
            focal = int(focal)
            start = focal - win // 2
            end = start + win
            focal_in_window = focal - start

            ref_seq = _fetch_window(genome, chrom, chrom_length, start, end)

            seq_t = torch.from_numpy(ref_seq[None]).to(device).float()
            with autocast_ctx:
                logit_ref = model(seq_t)
            p_ref = float(torch.sigmoid(logit_ref.float())
                          .squeeze()[focal_in_window].item())
            focal_probs[fi] = p_ref

            sal_lo = focal_in_window - half
            sal_hi = focal_in_window + half + 1
            ref_idx = _seq_to_int(ref_seq[:, sal_lo:sal_hi])
            ref_seqs[fi] = ref_idx

            mut_seqs = np.empty(
                (3 * saliency_w, 4, win), dtype=np.float32,
            )
            sal_pos_index = np.empty(3 * saliency_w, dtype=np.int64)
            alt_index = np.empty(3 * saliency_w, dtype=np.int64)

            slot = 0
            for s in range(saliency_w):
                wi = sal_lo + s
                ref_b = int(ref_idx[s])
                k = 0
                for alt in range(4):
                    if alt == ref_b:
                        continue
                    mut = ref_seq.copy()
                    mut[:, wi] = 0.0
                    mut[alt, wi] = 1.0
                    mut_seqs[slot] = mut
                    sal_pos_index[slot] = s
                    alt_index[slot] = k
                    slot += 1
                    k += 1

            drops_sum = np.zeros(saliency_w, dtype=np.float64)
            for bs in range(0, mut_seqs.shape[0], batch_size):
                batch = mut_seqs[bs : bs + batch_size]
                batch_t = torch.from_numpy(batch).to(device, non_blocking=True)
                with autocast_ctx:
                    logits = model(batch_t)
                p_alt = (torch.sigmoid(logits.float())
                         .squeeze(1)[:, focal_in_window]
                         .cpu().numpy())
                drops = np.maximum(0.0, p_ref - p_alt.astype(np.float64))
                for k_in_batch in range(len(batch)):
                    s = int(sal_pos_index[bs + k_in_batch])
                    drops_sum[s] += drops[k_in_batch]

            saliency[fi] = (drops_sum / 3.0).astype(np.float32)

            if (fi + 1) % log_every == 0 or (fi + 1) == n_focal:
                elapsed = time.time() - t0
                eta = elapsed * (n_focal / (fi + 1) - 1)
                log.info("  ISM %d/%d  elapsed=%.1fs  eta=%.1fs  "
                         "mean_focal_prob=%.4f  mean_saliency=%.5f",
                         fi + 1, n_focal, elapsed, eta,
                         float(focal_probs[: fi + 1].mean()),
                         float(saliency[: fi + 1].mean()))

    log.info("compute_saliency: %d focals processed in %.1fs",
             n_focal, time.time() - t0)
    return focal_probs, saliency, ref_seqs


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


__all__ = [
    "select_top_positions",
    "compute_saliency",
    "BASES",
    "BASE_TO_IDX",
    "IDX_TO_BASE",
]


# Smoke tests — ``python -m src.ism``.
def _smoke() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- select_top_positions ----
    np.random.seed(0)
    L = 10_000
    probs = np.random.rand(L).astype(np.float32)
    mask = np.ones(L, dtype=np.uint8)
    mask[5000:5500] = 0  # mask a band
    top = select_top_positions(probs, mask, top_k=10, min_separation=200)
    assert len(top) <= 10
    assert all(mask[p] == 1 for p in top), "selected a masked position!"
    diffs = np.diff(top)
    assert (diffs >= 200).all(), f"min_separation violated: diffs={diffs}"
    print(f"select_top_positions: {len(top)} positions, min sep={diffs.min() if diffs.size else 'n/a'}")

    # Edge: top_k larger than callable count
    small_mask = np.zeros(L, dtype=np.uint8)
    small_mask[:5] = 1
    top2 = select_top_positions(probs, small_mask, top_k=100, min_separation=1)
    assert len(top2) == 5
    print(f"select_top_positions exhausts candidate list: got {len(top2)}/100")

    # ---- compute_saliency — mock model returning a fixed prob ----
    class ConstModel(torch.nn.Module):
        """Always returns logits → sigmoid = 0.5 (no signal)."""
        def __init__(self): super().__init__()
        def forward(self, x):
            B, _, L = x.shape
            return torch.zeros(B, 1, L)
    class FakeGenome:
        def __init__(self, L):
            self.L = L
            rng = np.random.default_rng(0)
            self._seq = np.zeros((4, L), dtype=np.float32)
            idx = rng.integers(0, 4, size=L)
            self._seq[idx, np.arange(L)] = 1.0
        def get(self, chrom, start, end, strand="+"):
            return self._seq[:, start:end]

    g = FakeGenome(L=200_000)
    model = ConstModel().eval()
    focals = np.array([100_000], dtype=np.int64)
    fp, sal, refs = compute_saliency(
        model, g, "chr_test", 200_000, focals,
        window_half=5, batch_size=8, device=torch.device("cpu"),
        use_amp=False, log_every=1,
    )
    assert fp.shape == (1,) and abs(fp[0] - 0.5) < 1e-5
    assert sal.shape == (1, 11)
    # All saliency contributions are p_ref - p_alt = 0 → mean drop = 0
    assert (sal == 0).all(), f"saliency should be zero everywhere for ConstModel; got max={sal.max()}"
    assert refs.shape == (1, 11) and refs.dtype == np.uint8
    print(f"compute_saliency (const model): saliency all zero (max={sal.max()})")

    # ---- compute_saliency — mock model that returns 1 iff focal base is A ----
    class FocalAModel(torch.nn.Module):
        """Returns large positive logit at every position iff that position's
        reference base is A (channel 0). Mutating an A at the focal to non-A
        drops p toward 0; mutating elsewhere doesn't affect the focal."""
        def __init__(self): super().__init__()
        def forward(self, x):
            # logit at position i = 10 * x[:, 0, i] - 5  (= +5 for A, -5 for non-A)
            return (10.0 * x[:, 0:1] - 5.0)

    # Build a genome where the focal position is 'A' (channel 0 = 1.0)
    g2 = FakeGenome(L=200_000)
    g2._seq[:, 100_000] = [1.0, 0.0, 0.0, 0.0]  # force A at focal
    # Make all surrounding bases non-A so mutating them won't affect focal.
    for i in range(99_995, 100_006):
        if i != 100_000:
            g2._seq[:, i] = [0.0, 1.0, 0.0, 0.0]  # C at the non-focal sal-window positions

    model2 = FocalAModel().eval()
    fp2, sal2, refs2 = compute_saliency(
        model2, g2, "chr_test", 200_000, np.array([100_000], dtype=np.int64),
        window_half=5, batch_size=8, device=torch.device("cpu"),
        use_amp=False, log_every=1,
    )
    # p_ref at focal: sigmoid(10*1 - 5) = sigmoid(5) ≈ 0.9933
    assert abs(fp2[0] - (1.0 / (1.0 + np.exp(-5)))) < 1e-5
    # Saliency at the focal position itself (index 5 in the 11-position
    # window): mutating A→C, A→G, A→T at the focal turns its channel-0
    # value from 1.0 to 0.0, so the model's logit at the focal becomes
    # sigmoid(-5) ≈ 0.0067 → drop = 0.9933 - 0.0067 ≈ 0.987 for all 3 alts.
    print(f"  focal saliency: {sal2[0, 5]:.4f} (expect ~0.987)")
    assert sal2[0, 5] > 0.95
    # Saliency at non-focal positions: mutating non-A bases anywhere doesn't
    # touch the focal's channel-0 value, so the focal logit is unchanged →
    # drop = 0.
    non_focal = np.concatenate([sal2[0, :5], sal2[0, 6:]])
    print(f"  non-focal saliency max: {non_focal.max():.6f} (expect ~0)")
    assert non_focal.max() < 1e-4
    print(f"compute_saliency (FocalA model): saliency localizes to focal as expected")

    print("\nISM SMOKE PASSED")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
