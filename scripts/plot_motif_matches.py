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
DEFAULT_CLUSTER_PWMS = f"{BASE}/interpretation/cluster_pwms.npz"
DEFAULT_MATCHES = f"{BASE}/interpretation/jaspar_matches.json"
DEFAULT_JASPAR = (
    "/gpfs/data/zhou-lab/haoyanghu/data/jaspar/"
    "JASPAR2024_CORE_non-redundant_pfms_jaspar.txt"
)
DEFAULT_OUTPUT_DIR = f"{BASE}/interpretation"

ALPHABET = ["A", "C", "G", "T"]


log = logging.getLogger(__name__)


# JASPAR parsing — re-implemented here so the script is self-contained.
_HEADER_RE = re.compile(r"^>\s*(\S+)\s+(\S.*?)\s*$")
_ROW_RE = re.compile(r"^([ACGT])\s*\[\s*([\d\s.]+?)\s*\]")


def parse_jaspar_pfm_file(path: str | Path) -> dict[str, dict]:
    """Return ``{name: {matrix_id, name, pcm, pwm}}``."""
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
            # Column-normalized PWM with pseudocount 0.25 (matches build_pwm).
            col_sums = pcm.sum(axis=0, keepdims=True) + 4 * 0.25
            pwm = (pcm + 0.25) / col_sums
            entries[name] = {
                "matrix_id": matrix_id,
                "name": name,
                "pcm": pcm,
                "pwm": pwm,
            }
        i += 5

    return entries


def _import_logomaker():
    import logomaker
    return logomaker


def _pwm_to_information_matrix(pwm: np.ndarray) -> np.ndarray:
    """(4, W) PWM → (W, 4) IC-weighted heights for logomaker.Logo."""
    p = pwm.copy()
    eps = 1e-12
    H = -(p * np.log2(p + eps)).sum(axis=0)
    ic_per_pos = 2.0 - H
    heights = (p * ic_per_pos[None, :]).T
    return heights


def plot_logo_on_axis(ax, pwm: np.ndarray, title: str, fontsize: int = 11):
    """Plot an IC-weighted sequence logo on the given axis."""
    import pandas as pd
    logomaker = _import_logomaker()
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


def trim_cluster_to_match_region(
    cluster_pwm: np.ndarray,
    jaspar_pwm: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Best-offset alignment of cluster PWM to JASPAR PWM (IC-weighted
    Pearson). Returns ``(cluster_slice, start, end)``.
    """
    Wc = cluster_pwm.shape[1]
    Wj = jaspar_pwm.shape[1]

    eps = 1e-12
    H_j = -(jaspar_pwm * np.log2(jaspar_pwm + eps)).sum(axis=0)
    ic_j = 2.0 - H_j

    best_offset = 0
    best_score = -np.inf
    min_overlap = max(4, min(Wc, Wj) // 2)
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

    cluster_start = max(0, best_offset)
    cluster_end = min(Wc, best_offset + Wj)
    cluster_slice = cluster_pwm[:, cluster_start:cluster_end]
    return cluster_slice, cluster_start, cluster_end


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cluster-pwms", default=DEFAULT_CLUSTER_PWMS)
    p.add_argument("--matches-json", default=DEFAULT_MATCHES)
    p.add_argument("--jaspar-pfm-file", default=DEFAULT_JASPAR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--min-cluster-size", type=int, default=20,
                   help="Skip clusters smaller than this (default 20).")
    p.add_argument("--min-score", type=float, default=0.55,
                   help="Skip clusters with best K562 match score below this.")
    p.add_argument("--use-k562-match", action="store_true", default=True,
                   help="Use the top K562 match (default). False = all JASPAR.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info("plot_motif_matches.py starting")
    for k, v in vars(args).items():
        log.info("  %-20s = %s", k, v)

    if not os.path.exists(args.cluster_pwms):
        raise FileNotFoundError(args.cluster_pwms)
    if not os.path.exists(args.matches_json):
        raise FileNotFoundError(args.matches_json)
    if not os.path.exists(args.jaspar_pfm_file):
        raise FileNotFoundError(args.jaspar_pfm_file)

    bundle = np.load(args.cluster_pwms, allow_pickle=True)
    with open(args.matches_json) as f:
        matches = json.load(f)
    jaspar = parse_jaspar_pfm_file(args.jaspar_pfm_file)
    log.info("Loaded %d JASPAR entries (by name)", len(jaspar))

    qualifying: list[dict] = []
    match_field = "top_matches_k562" if args.use_k562_match else "top_matches_all"
    for entry in matches["per_cluster"]:
        cid = entry["cluster_id"]
        size = entry.get("size", 0) or 0
        top = entry.get(match_field) or []
        if not top:
            continue
        best = top[0]
        if size < args.min_cluster_size:
            continue
        if best["score"] < args.min_score:
            continue
        if best["name"] not in jaspar:
            log.warning("Cluster %d: JASPAR entry %r not found by name; "
                        "skipping", cid, best["name"])
            continue
        qualifying.append({
            "cluster_id": cid,
            "size": size,
            "tf_name": best["name"],
            "match_score": best["score"],
            "jaspar_pwm": jaspar[best["name"]]["pwm"],
            "jaspar_matrix_id": jaspar[best["name"]]["matrix_id"],
        })

    if not qualifying:
        raise RuntimeError(
            "No clusters pass min-cluster-size + min-score filters. "
            "Try lowering --min-score."
        )
    log.info("Qualifying clusters (%d):", len(qualifying))
    for q in qualifying:
        log.info("  cluster %d (n=%d) -> %s (%s) score=%.3f",
                 q["cluster_id"], q["size"], q["tf_name"],
                 q["jaspar_matrix_id"], q["match_score"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in qualifying:
        c_pwm = bundle[f"cluster_{q['cluster_id']}_pwm"]
        c_slice, c_start, c_end = trim_cluster_to_match_region(
            c_pwm, q["jaspar_pwm"],
        )
        q["cluster_slice"] = c_slice
        q["cluster_start"] = c_start
        q["cluster_end"] = c_end
        log.info("  cluster %d: trimmed to positions %d..%d "
                 "(slice width %d)",
                 q["cluster_id"], c_start, c_end, c_slice.shape[1])

    for q in qualifying:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3))
        title_L = (
            f"Cluster {q['cluster_id']} "
            f"(n={q['size']}, pos {q['cluster_start']}–{q['cluster_end']})"
        )
        title_R = (
            f"JASPAR {q['tf_name']} ({q['jaspar_matrix_id']})  "
            f"  match score = {q['match_score']:.3f}"
        )
        plot_logo_on_axis(axL, q["cluster_slice"], title_L)
        plot_logo_on_axis(axR, q["jaspar_pwm"], title_R)
        fig.tight_layout()
        out_path = out_dir / (
            f"cluster_{q['cluster_id']:02d}_vs_{q['tf_name']}.png"
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
            f"Cluster {q['cluster_id']} "
            f"(n={q['size']}, matched region pos {q['cluster_start']}–"
            f"{q['cluster_end']})"
        )
        title_R = (
            f"JASPAR {q['tf_name']} ({q['jaspar_matrix_id']})  "
            f"  score = {q['match_score']:.3f}"
        )
        plot_logo_on_axis(axes[row, 0], q["cluster_slice"], title_L)
        plot_logo_on_axis(axes[row, 1], q["jaspar_pwm"], title_R)
    fig.suptitle(
        "K562 enhancer prediction — learned motifs vs JASPAR (K562 TFs)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    grid_path = out_dir / "motif_matches_grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", grid_path)

    log.info("Done. Wrote %d per-cluster figures + 1 grid figure.",
             len(qualifying))
    return 0


if __name__ == "__main__":
    sys.exit(main())
