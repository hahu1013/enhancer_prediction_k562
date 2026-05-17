"""Realign each cluster member's sequence to its peak-saliency position,
rebuild the PWM, and re-plot vs JASPAR.

The first-pass cluster PWMs from extract_motifs.py average out the motif
when members carry it at different positions within the window —
realigning fixes that.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np


BASE = "/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction"
DEFAULT_ISM = f"{BASE}/interpretation/ism_saliency_chr10.npz"
DEFAULT_CLUSTER_PWMS = f"{BASE}/interpretation/cluster_pwms.npz"
DEFAULT_MATCHES = f"{BASE}/interpretation/jaspar_matches.json"
DEFAULT_MOTIF_CLUSTERS = f"{BASE}/interpretation/motif_clusters.json"
DEFAULT_JASPAR = (
    "/gpfs/data/zhou-lab/haoyanghu/data/jaspar/"
    "JASPAR2024_CORE_non-redundant_pfms_jaspar.txt"
)
DEFAULT_OUTPUT_DIR = f"{BASE}/interpretation"

ALPHABET = ["A", "C", "G", "T"]
ALPHABET_IDX = {b: i for i, b in enumerate(ALPHABET)}


log = logging.getLogger(__name__)


# JASPAR parsing — self-contained, mirrors plot_motif_matches.py.
_HEADER_RE = re.compile(r"^>\s*(\S+)\s+(\S.*?)\s*$")
_ROW_RE = re.compile(r"^([ACGT])\s*\[\s*([\d\s.]+?)\s*\]")


def parse_jaspar_pfm_file(path: str | Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m_hdr = _HEADER_RE.match(lines[i])
        if not m_hdr:
            i += 1
            continue
        matrix_id = m_hdr.group(1)
        name = m_hdr.group(2)
        rows: dict[str, list[float]] = {}
        for j in range(1, 5):
            if i + j >= len(lines):
                break
            m_row = _ROW_RE.match(lines[i + j])
            if not m_row:
                break
            base = m_row.group(1)
            counts = [float(x) for x in m_row.group(2).split()]
            rows[base] = counts
        if len(rows) == 4 and len({len(v) for v in rows.values()}) == 1:
            pcm = np.array([rows[b] for b in ALPHABET], dtype=np.float64)
            col_sums = pcm.sum(axis=0, keepdims=True) + 4 * 0.25
            pwm = (pcm + 0.25) / col_sums
            entries[name] = {
                "matrix_id": matrix_id, "name": name,
                "pcm": pcm, "pwm": pwm,
            }
        i += 5
    return entries


def build_pwm_from_sequences(seq_array: np.ndarray, pseudocount: float = 0.25
                              ) -> np.ndarray:
    """(N, W) base-index sequences (0=A..3=T) → (4, W) smoothed PWM."""
    N, W = seq_array.shape
    pcm = np.zeros((4, W), dtype=np.float64)
    for b in range(4):
        pcm[b] = (seq_array == b).sum(axis=0)
    col_sums = pcm.sum(axis=0, keepdims=True) + 4 * pseudocount
    pwm = (pcm + pseudocount) / col_sums
    return pwm


def pwm_information_content(pwm: np.ndarray) -> np.ndarray:
    """Per-column IC in bits."""
    eps = 1e-12
    H = -(pwm * np.log2(pwm + eps)).sum(axis=0)
    return 2.0 - H


def _pwm_to_information_matrix(pwm: np.ndarray) -> np.ndarray:
    """(4, W) PWM -> (W, 4) IC-weighted heights for logomaker.Logo()."""
    ic_per_pos = pwm_information_content(pwm)
    heights = (pwm * ic_per_pos[None, :]).T
    return heights


def realign_to_peak_saliency(
    sequences: np.ndarray,
    saliency: np.ndarray,
    member_idx: np.ndarray,
    align_half: int,
    min_saliency: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a (2*align_half+1) window centered on each member's
    peak-saliency position. Drops members whose peak is too close to the
    edge or below ``min_saliency``. Returns ``(aligned_seqs, kept_idx)``.
    """
    W = sequences.shape[1]
    aligned_seqs: list[np.ndarray] = []
    kept: list[int] = []
    for i in member_idx:
        sal = saliency[i]
        if min_saliency is not None and sal.max() < min_saliency:
            continue
        peak = int(np.argmax(sal))
        start = peak - align_half
        end = peak + align_half + 1
        if start < 0 or end > W:
            continue
        aligned_seqs.append(sequences[i, start:end])
        kept.append(int(i))
    if not aligned_seqs:
        return (np.zeros((0, 2 * align_half + 1), dtype=np.uint8),
                np.array([], dtype=np.int64))
    return (np.array(aligned_seqs, dtype=np.uint8),
            np.array(kept, dtype=np.int64))


def plot_logo_on_axis(ax, pwm: np.ndarray, title: str, fontsize: int = 11):
    import pandas as pd
    import logomaker
    heights = _pwm_to_information_matrix(pwm)
    df = pd.DataFrame(heights, columns=ALPHABET)
    df.index.name = "pos"
    logo = logomaker.Logo(
        df, ax=ax,
        color_scheme="classic",
        show_spines=False,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)
    ax.set_ylim(0, 2.05)
    ax.set_ylabel("bits", fontsize=fontsize - 1)
    ax.set_xlabel("position", fontsize=fontsize - 1)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)


def score_alignment(cluster_pwm: np.ndarray, jaspar_pwm: np.ndarray
                     ) -> tuple[float, int]:
    """Best IC-weighted Pearson alignment of two PWMs. Returns ``(score, offset)``."""
    Wc = cluster_pwm.shape[1]
    Wj = jaspar_pwm.shape[1]
    eps = 1e-12
    ic_j = pwm_information_content(jaspar_pwm)
    min_overlap = max(4, min(Wc, Wj) // 2)
    best_offset = 0
    best_score = -np.inf
    for offset in range(-(Wj - min_overlap), Wc - min_overlap + 1):
        j_starts = max(0, -offset)
        j_ends = min(Wj, Wc - offset)
        c_starts = max(0, offset)
        c_ends = c_starts + (j_ends - j_starts)
        if j_ends - j_starts < min_overlap:
            continue
        c_slice = cluster_pwm[:, c_starts:c_ends]
        j_slice = jaspar_pwm[:, j_starts:j_ends]
        ic_slice = ic_j[j_starts:j_ends]
        cv = c_slice.flatten() - c_slice.mean()
        jv = j_slice.flatten() - j_slice.mean()
        w = np.repeat(ic_slice, 4)
        num = (w * cv * jv).sum()
        den = np.sqrt((w * cv * cv).sum() * (w * jv * jv).sum() + eps)
        score = num / den
        if score > best_score:
            best_score = score
            best_offset = offset
    return float(best_score), int(best_offset)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ism-output", default=DEFAULT_ISM)
    p.add_argument("--motif-clusters-json", default=DEFAULT_MOTIF_CLUSTERS,
                   help="motif_clusters.json from extract_motifs.py.")
    p.add_argument("--matches-json", default=DEFAULT_MATCHES)
    p.add_argument("--jaspar-pfm-file", default=DEFAULT_JASPAR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--align-half", type=int, default=10,
                   help="Half-width of the realigned window (default 10).")
    p.add_argument("--min-saliency", type=float, default=None,
                   help="Drop members with max saliency below this.")
    p.add_argument("--min-cluster-size", type=int, default=20,
                   help="Skip clusters with fewer members after filtering.")
    p.add_argument("--min-score", type=float, default=0.55,
                   help="Skip clusters with initial K562 match below this.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info("realign_and_replot.py starting")
    for k, v in vars(args).items():
        log.info("  %-22s = %s", k, v)

    for path in (args.ism_output, args.motif_clusters_json, args.matches_json,
                 args.jaspar_pfm_file):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    ism = np.load(args.ism_output, allow_pickle=True)
    if "saliency" not in ism.files or "sequences" not in ism.files:
        raise ValueError(f"ISM .npz missing 'saliency' or 'sequences' "
                         f"(has {sorted(ism.files)})")
    saliency = ism["saliency"]  # (K, W)
    sequences = ism["sequences"]  # (K, W) ints in 0..3
    K, W = saliency.shape
    log.info("Loaded ISM: K=%d focals, W=%d window", K, W)

    with open(args.motif_clusters_json) as f:
        clusters_json = json.load(f)
    with open(args.matches_json) as f:
        matches_json = json.load(f)
    jaspar = parse_jaspar_pfm_file(args.jaspar_pfm_file)
    log.info("Loaded %d JASPAR entries (by name)", len(jaspar))

    # Map cluster member focal positions back to indices into the K axis.
    ism_focal_positions = ism["focal_positions"]
    pos_to_idx = {int(p): i for i, p in enumerate(ism_focal_positions)}

    cluster_to_match: dict[int, dict] = {}
    for entry in matches_json["per_cluster"]:
        top = entry.get("top_matches_k562") or []
        if not top:
            continue
        cluster_to_match[entry["cluster_id"]] = top[0]

    cluster_to_members: dict[int, list[int]] = {}
    for entry in clusters_json["clusters"]:
        cluster_to_members[entry["cluster_id"]] = entry["focal_positions"]

    qualifying: list[dict] = []
    for cid, focal_positions in cluster_to_members.items():
        if cid not in cluster_to_match:
            continue
        match = cluster_to_match[cid]
        size = len(focal_positions)
        if size < args.min_cluster_size:
            continue
        if match["score"] < args.min_score:
            continue
        if match["name"] not in jaspar:
            log.warning("Cluster %d: TF %r not in JASPAR by name; skipping",
                        cid, match["name"])
            continue
        member_idx = np.array(
            [pos_to_idx[p] for p in focal_positions if p in pos_to_idx],
            dtype=np.int64,
        )
        qualifying.append({
            "cluster_id": cid,
            "original_size": size,
            "tf_name": match["name"],
            "original_score": match["score"],
            "jaspar_pwm": jaspar[match["name"]]["pwm"],
            "jaspar_matrix_id": jaspar[match["name"]]["matrix_id"],
            "member_idx": member_idx,
        })

    if not qualifying:
        raise RuntimeError(
            "No clusters pass min-cluster-size + min-score filters."
        )
    log.info("Qualifying clusters (%d):", len(qualifying))
    for q in qualifying:
        log.info("  cluster %d (n=%d) -> %s (%s) score=%.3f",
                 q["cluster_id"], q["original_size"], q["tf_name"],
                 q["jaspar_matrix_id"], q["original_score"])

    for q in qualifying:
        aligned_seqs, kept_idx = realign_to_peak_saliency(
            sequences, saliency, q["member_idx"],
            align_half=args.align_half,
            min_saliency=args.min_saliency,
        )
        if len(aligned_seqs) < args.min_cluster_size:
            log.warning(
                "  cluster %d: only %d members survived realignment + filter "
                "(<%d); skipping this cluster from output.",
                q["cluster_id"], len(aligned_seqs), args.min_cluster_size,
            )
            q["skipped"] = True
            continue
        new_pwm = build_pwm_from_sequences(aligned_seqs)
        new_ic = pwm_information_content(new_pwm)
        new_score, new_offset = score_alignment(new_pwm, q["jaspar_pwm"])
        q["realigned_pwm"] = new_pwm
        q["n_kept"] = int(len(aligned_seqs))
        q["new_mean_ic"] = float(new_ic.mean())
        q["new_max_ic"] = float(new_ic.max())
        q["new_max_ic_pos"] = int(new_ic.argmax())
        q["new_score"] = float(new_score)
        q["new_offset"] = int(new_offset)
        q["skipped"] = False
        log.info(
            "  cluster %d realigned: n_kept=%d  mean IC=%.3f (was very low)  "
            "max IC=%.3f at pos %d  new match score=%.3f",
            q["cluster_id"], q["n_kept"], q["new_mean_ic"],
            q["new_max_ic"], q["new_max_ic_pos"], q["new_score"],
        )

    qualifying = [q for q in qualifying if not q["skipped"]]
    if not qualifying:
        raise RuntimeError("All clusters lost members after realignment.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in qualifying:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3))
        title_L = (
            f"Cluster {q['cluster_id']} realigned  "
            f"(n={q['n_kept']}/{q['original_size']}, max IC={q['new_max_ic']:.2f})"
        )
        title_R = (
            f"JASPAR {q['tf_name']} ({q['jaspar_matrix_id']})  "
            f"  match score: {q['original_score']:.3f} -> {q['new_score']:.3f}"
        )
        plot_logo_on_axis(axL, q["realigned_pwm"], title_L)
        plot_logo_on_axis(axR, q["jaspar_pwm"], title_R)
        fig.tight_layout()
        out_path = out_dir / (
            f"cluster_{q['cluster_id']:02d}_realigned_vs_{q['tf_name']}.png"
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)

    n = len(qualifying)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for row, q in enumerate(qualifying):
        title_L = (
            f"Cluster {q['cluster_id']} realigned to peak saliency  "
            f"(n={q['n_kept']}/{q['original_size']})"
        )
        title_R = (
            f"JASPAR {q['tf_name']} ({q['jaspar_matrix_id']})  "
            f"  score: {q['original_score']:.3f} -> {q['new_score']:.3f}"
        )
        plot_logo_on_axis(axes[row, 0], q["realigned_pwm"], title_L)
        plot_logo_on_axis(axes[row, 1], q["jaspar_pwm"], title_R)
    fig.suptitle(
        "K562 enhancer prediction — peak-realigned motifs vs JASPAR",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    grid_path = out_dir / "motif_matches_realigned_grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", grid_path)

    stats = {
        "ism_output": args.ism_output,
        "align_half": args.align_half,
        "min_saliency": args.min_saliency,
        "clusters": [
            {
                "cluster_id": q["cluster_id"],
                "tf_name": q["tf_name"],
                "jaspar_matrix_id": q["jaspar_matrix_id"],
                "original_size": q["original_size"],
                "n_kept": q["n_kept"],
                "original_match_score": q["original_score"],
                "realigned_match_score": q["new_score"],
                "realigned_mean_ic": q["new_mean_ic"],
                "realigned_max_ic": q["new_max_ic"],
                "realigned_max_ic_pos": q["new_max_ic_pos"],
            }
            for q in qualifying
        ],
    }
    stats_path = out_dir / "realigned_pwm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Wrote %s", stats_path)

    log.info("Done. %d cluster figure(s) + 1 grid figure.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
