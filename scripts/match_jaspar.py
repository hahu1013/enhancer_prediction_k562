"""Match each cluster PWM against JASPAR 2024 CORE vertebrates PFMs.

For each cluster, reports the top-N matches overall and the top-N
filtered to a K562-relevant TF allow-list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# Path shim
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.motifs import (  # noqa: E402
    parse_jaspar_pfm_file,
    match_cluster_against_jaspar,
)


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_CLUSTER_PWMS = (
    f"{BASE}/projects/enhancer_prediction/interpretation/cluster_pwms.npz"
)
DEFAULT_JASPAR = f"{BASE}/data/jaspar/JASPAR2024_CORE_vertebrates_nr.txt"
DEFAULT_OUTPUT = (
    f"{BASE}/projects/enhancer_prediction/interpretation/jaspar_matches.json"
)

# K562-relevant TFs for the filtered match table.
DEFAULT_K562_TFS = ",".join([
    "GATA1", "GATA2", "TAL1", "KLF1", "MYB",
    "NFY-A", "NFY-B", "NFYA", "NFYB", "SP1",
    "RUNX1", "MYC", "JUN", "FOS", "FOSL1", "FOSL2",
    "BACH1", "NRF1", "MAFK",
])


log = logging.getLogger(__name__)


def _normalize_tf_name(name: str) -> str:
    """Uppercase + strip ``-``/``_`` so 'NFY-A' matches JASPAR 'NFYA'."""
    return name.strip().upper().replace("-", "").replace("_", "")


def filter_k562(matches: list[dict], k562_set: set[str]) -> list[dict]:
    """Keep matches whose normalized JASPAR name is in the K562 allow-set."""
    k562_norm = {_normalize_tf_name(t) for t in k562_set}
    out: list[dict] = []
    for m in matches:
        name = _normalize_tf_name(m["name"])
        if name in k562_norm:
            out.append(m)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cluster-pwms", default=DEFAULT_CLUSTER_PWMS,
                   help=f"cluster_pwms.npz from extract_motifs.py "
                        f"(default {DEFAULT_CLUSTER_PWMS}).")
    p.add_argument("--jaspar-pfm-file", default=DEFAULT_JASPAR,
                   help=f"JASPAR PFM bundle (default {DEFAULT_JASPAR}).")
    p.add_argument("--k562-tf-list", default=DEFAULT_K562_TFS,
                   help="Comma-separated K562-relevant TF names.")
    p.add_argument("--top-n-matches", type=int, default=3,
                   help="Top-N JASPAR matches per cluster (default 3).")
    p.add_argument("--max-offset", type=int, default=5,
                   help="Maximum ±offset (bp) for PWM alignment.")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output JSON path (default {DEFAULT_OUTPUT}).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info("match_jaspar.py starting")
    for k, v in vars(args).items():
        log.info("  %-18s = %s", k, v)

    if not os.path.exists(args.cluster_pwms):
        raise FileNotFoundError(f"cluster_pwms.npz not found: {args.cluster_pwms}")
    bundle = np.load(args.cluster_pwms, allow_pickle=True)
    if "_cluster_ids" not in bundle.files:
        raise ValueError(
            f"{args.cluster_pwms}: missing '_cluster_ids' key; not a "
            f"cluster_pwms.npz from extract_motifs.py?"
        )
    cluster_ids = bundle["_cluster_ids"].tolist()
    cluster_sizes = (bundle["_cluster_sizes"].tolist()
                     if "_cluster_sizes" in bundle.files
                     else [None] * len(cluster_ids))
    log.info("Loaded %d cluster PWMs", len(cluster_ids))

    if not os.path.exists(args.jaspar_pfm_file):
        raise FileNotFoundError(
            f"JASPAR PFM file not found: {args.jaspar_pfm_file}"
        )
    jaspar_entries = parse_jaspar_pfm_file(args.jaspar_pfm_file)
    if not jaspar_entries:
        raise RuntimeError(
            f"Parsed zero entries from {args.jaspar_pfm_file} — file format "
            f"unexpected. Inspect the file head and adjust parse_jaspar_pfm_file."
        )
    log.info("Loaded %d JASPAR entries (e.g. %s)",
             len(jaspar_entries),
             [e["name"] for e in jaspar_entries[:5]])

    k562_set = {t.strip() for t in args.k562_tf_list.split(",") if t.strip()}
    log.info("K562 TF allow-set: %s", sorted(k562_set))

    per_cluster: list[dict] = []
    for cid, csize in zip(cluster_ids, cluster_sizes):
        pwm = bundle[f"cluster_{int(cid)}_pwm"]
        ic = bundle[f"cluster_{int(cid)}_ic"]
        top_all = match_cluster_against_jaspar(
            pwm, ic, jaspar_entries,
            top_n=args.top_n_matches, max_offset=args.max_offset,
        )
        # K562 subset: score against all entries, filter, then truncate.
        full_sorted = match_cluster_against_jaspar(
            pwm, ic, jaspar_entries,
            top_n=len(jaspar_entries), max_offset=args.max_offset,
        )
        top_k562 = filter_k562(full_sorted, k562_set)[: args.top_n_matches]
        per_cluster.append({
            "cluster_id": int(cid),
            "size": (int(csize) if csize is not None else None),
            "top_matches_all": top_all,
            "top_matches_k562": top_k562,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_pwms_path": args.cluster_pwms,
        "jaspar_pfm_file": args.jaspar_pfm_file,
        "n_jaspar_entries": len(jaspar_entries),
        "k562_tf_list": sorted(k562_set),
        "top_n_matches": int(args.top_n_matches),
        "max_offset": int(args.max_offset),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_cluster": per_cluster,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s", out_path)

    log.info("=" * 78)
    log.info("Cluster JASPAR matches (top match across all + top K562 match):")
    log.info("  %-3s %-8s %-25s %-10s %-25s %-10s",
             "id", "size", "top_all", "score", "top_k562", "score")
    log.info("-" * 78)
    for c in per_cluster:
        all_top = (c["top_matches_all"][0] if c["top_matches_all"] else None)
        k562_top = (c["top_matches_k562"][0] if c["top_matches_k562"] else None)
        def fmt(m, w):
            if m is None:
                return f"{'-':<{w}} {'-':>10}"
            name = m["name"][:w - 1]
            return f"{name:<{w}} {m['score']:>10.4f}"
        size_str = (str(c["size"]) if c["size"] is not None else "-")
        log.info("  %-3d %-8s %s %s",
                 c["cluster_id"], size_str,
                 fmt(all_top, 25), fmt(k562_top, 25))
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
