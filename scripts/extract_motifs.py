"""Cluster ISM saliency profiles and build per-cluster PWMs + logos."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


# Path shim
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.motifs import (  # noqa: E402
    cluster_saliency_profiles,
    build_pwm,
    pwm_logo,
    ALPHABET_ACGT,
)


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_ISM = (
    f"{BASE}/projects/enhancer_prediction/interpretation/ism_saliency_chr10.npz"
)
DEFAULT_OUTPUT_DIR = f"{BASE}/projects/enhancer_prediction/interpretation"


log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ism-output", default=DEFAULT_ISM,
                   help=f"Phase-6.1 saliency .npz (default {DEFAULT_ISM}).")
    p.add_argument("--n-clusters", type=int, default=10,
                   help="Target number of clusters (default 10).")
    p.add_argument("--min-cluster-size", type=int, default=20,
                   help="Clusters smaller than this are flagged 'below_min_size'.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Output dir (default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--linkage", default="average",
                   choices=["average", "complete", "single", "ward"],
                   help="AgglomerativeClustering linkage (ward needs euclidean).")
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
    log.info("extract_motifs.py starting")
    for k, v in vars(args).items():
        log.info("  %-16s = %s", k, v)

    np.random.seed(args.seed)

    if not os.path.exists(args.ism_output):
        raise FileNotFoundError(f"ISM .npz not found: {args.ism_output}")
    data = np.load(args.ism_output, allow_pickle=True)
    for k in ("focal_positions", "saliency", "sequences"):
        if k not in data.files:
            raise ValueError(
                f"{args.ism_output}: missing key {k!r} (has {sorted(data.files)})"
            )
    focal_positions = data["focal_positions"]
    saliency = data["saliency"].astype(np.float32, copy=False)
    sequences = data["sequences"].astype(np.uint8, copy=False)
    focal_probs = (data["focal_probs"].astype(np.float32, copy=False)
                   if "focal_probs" in data.files
                   else np.full(len(focal_positions), np.nan, dtype=np.float32))
    try:
        ism_meta = json.loads(data["metadata"].item()) if "metadata" in data.files else {}
    except Exception:
        ism_meta = {}
    K, W = saliency.shape
    log.info("Loaded ISM: K=%d focal positions, W=%d window", K, W)
    log.info("  mean focal prob=%.4f  mean saliency=%.5f",
             float(focal_probs.mean()) if focal_probs.size else float("nan"),
             float(saliency.mean()))

    labels = cluster_saliency_profiles(
        saliency, n_clusters=args.n_clusters, method="agglomerative",
        linkage=args.linkage,
    )
    label_counts = Counter(int(l) for l in labels)
    log.info("Cluster sizes: %s",
             {cid: c for cid, c in sorted(label_counts.items())})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_records: list[dict] = []
    pwm_npz_payload: dict[str, np.ndarray] = {}

    for cid in sorted(label_counts):
        member_idx = np.where(labels == cid)[0]
        n_members = len(member_idx)
        if n_members == 0:
            continue
        cluster_seqs = [sequences[i] for i in member_idx]
        pwm_dict = build_pwm(cluster_seqs)
        logo_name = f"cluster_{cid:02d}_logo.png"
        logo_path = out_dir / logo_name
        try:
            pwm_logo(pwm_dict, logo_path)
            log.info("[cluster %d] n=%d  consensus=%r  mean IC=%.3f  → %s",
                     cid, n_members, pwm_dict["consensus"],
                     float(pwm_dict["information_content"].mean()),
                     logo_path.name)
        except Exception as e:
            log.warning("[cluster %d] logo failed: %s", cid, e)

        cluster_records.append({
            "cluster_id": cid,
            "size": n_members,
            "below_min_size": n_members < args.min_cluster_size,
            "consensus": pwm_dict["consensus"],
            "mean_information_content": float(
                pwm_dict["information_content"].mean()
            ),
            "information_content": pwm_dict["information_content"].tolist(),
            "mean_focal_prob": (float(focal_probs[member_idx].mean())
                                if focal_probs.size else None),
            "mean_focal_saliency": float(saliency[member_idx].mean()),
            "focal_positions": focal_positions[member_idx].tolist(),
            "logo_path": str(logo_path),
            "pcm_path_key": f"cluster_{cid}_pcm",
            "pwm_path_key": f"cluster_{cid}_pwm",
            "ic_path_key":  f"cluster_{cid}_ic",
        })
        pwm_npz_payload[f"cluster_{cid}_pcm"] = pwm_dict["pcm"].astype(np.int64)
        pwm_npz_payload[f"cluster_{cid}_pwm"] = pwm_dict["pwm"].astype(np.float64)
        pwm_npz_payload[f"cluster_{cid}_ic"]  = pwm_dict["information_content"].astype(np.float64)

    summary = {
        "ism_output": args.ism_output,
        "n_focal_positions": int(K),
        "window_size": int(W),
        "n_clusters_requested": int(args.n_clusters),
        "n_clusters_emitted": len(cluster_records),
        "linkage": args.linkage,
        "min_cluster_size": int(args.min_cluster_size),
        "ism_metadata": ism_meta,
        "clusters": cluster_records,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    json_path = out_dir / "motif_clusters.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", json_path)

    pwm_npz_payload["_cluster_ids"] = np.array(
        [r["cluster_id"] for r in cluster_records], dtype=np.int64,
    )
    pwm_npz_payload["_cluster_sizes"] = np.array(
        [r["size"] for r in cluster_records], dtype=np.int64,
    )
    pwm_npz_payload["_metadata"] = np.array(
        json.dumps({
            "ism_output": args.ism_output,
            "n_clusters_emitted": len(cluster_records),
            "linkage": args.linkage,
            "window_size": int(W),
            "alphabet": ALPHABET_ACGT,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }),
        dtype=object,
    )
    npz_path = out_dir / "cluster_pwms.npz"
    np.savez_compressed(npz_path, **pwm_npz_payload)
    log.info("Wrote %s", npz_path)

    log.info("=" * 70)
    log.info("Cluster summary:")
    log.info("  %-3s %-8s %-15s %-10s %-12s",
             "id", "size", "consensus", "mean IC", "below_min")
    log.info("-" * 70)
    for r in cluster_records:
        log.info("  %-3d %-8d %-15s %-10.3f %-12s",
                 r["cluster_id"], r["size"],
                 r["consensus"][: min(15, len(r["consensus"]))],
                 r["mean_information_content"],
                 "*" if r["below_min_size"] else "")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
