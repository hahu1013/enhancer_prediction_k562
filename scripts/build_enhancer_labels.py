"""Build per-chromosome enhancer-label arrays.

Writes ``{output_dir}/{chrom}.npy`` of shape (L, 2) uint8:
  arr[:, 0] = enhancer_label (1 = positive, 0 = negative)
  arr[:, 1] = callable_mask  (1 = contributes to loss, 0 = masked)

Positive rule: base is positive iff inside a pELS/dELS cCRE that
overlaps at least one K562 H3K27ac narrowPeak; the full cCRE is labeled,
not just the intersection.

Mask rule (union): cCRE PLS ± flank, CDS, RepeatMasker, FASTA Ns,
optional ENCODE blacklist. Mask wins over positive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


# Cluster paths — hardcoded constants for reproducibility.
BASE = "/gpfs/data/zhou-lab/haoyanghu"

HG38_FASTA = f"{BASE}/data/reference/hg38.fa.gz"
HG38_FAI = f"{BASE}/data/reference/hg38.fa.gz.fai"
HG38_PACKBITS = f"{BASE}/data/reference/hg38.fa.gz.packbits.npy"
HG38_PACKBITS_INDS = f"{BASE}/data/reference/hg38.fa.gz.packbits.inds.npy"
CHROM_SIZES = f"{BASE}/data/reference/hg38.chrom.sizes"

CCRE_BED = f"{BASE}/data/annotations/GRCh38-cCREs.bed"
H3K27AC_PEAKS = f"{BASE}/data/annotations/k562_h3k27ac/ENCFF532MMV.bed.gz"
GTF_PATH = f"{BASE}/data/annotations/Homo_sapiens.GRCh38.109.gtf.gz"
RMSK_PATH = f"{BASE}/data/annotations/rmsk.txt.gz"
BLACKLIST_BED_DEFAULT = (
    f"{BASE}/data/annotations/hg38-blacklist.v2.bed.gz"
)

DEFAULT_OUTPUT_DIR = (
    f"{BASE}/projects/enhancer_prediction/data/labels"
)

# chr1-22 + chrX; chrY skipped (K562 is female-derived).
DEFAULT_CHROMS = ",".join(
    [f"chr{i}" for i in range(1, 23)] + ["chrX"]
)

POSITIVE_CCRE_CLASSES = {"pELS", "dELS"}
PLS_CCRE_CLASS = "PLS"
N_BASE_CODES = frozenset(b"NnXx")


log = logging.getLogger(__name__)


# Annotation parsers return (starts, ends) int64 sorted by start, all
# coordinates 0-based half-open.
def _opener(path: str | os.PathLike):
    """Open transparently for gzip vs plain text."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_chrom_sizes(path: str) -> dict[str, int]:
    """Parse hg38.chrom.sizes (2-col TSV: chrom, length)."""
    sizes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise RuntimeError(f"No chromosome sizes parsed from {path}")
    return sizes


def load_ccre_bed(path: str, chrom: str):
    """Parse ENCODE cCRE V3 BED. Returns (starts, ends, classes) sorted."""
    starts: list[int] = []
    ends: list[int] = []
    classes: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[0] != chrom:
                continue
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))
            classes.append(parts[5])
    if not starts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
        )
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    c = np.asarray(classes, dtype=object)
    order = np.argsort(s, kind="stable")
    return s[order], e[order], c[order]


def load_h3k27ac_peaks(path: str, chrom: str):
    """Parse ENCODE narrowPeak.gz; returns sorted (starts, ends)."""
    starts: list[int] = []
    ends: list[int] = []
    with _opener(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def load_gtf_cds(path: str, chrom: str):
    """Parse Ensembl GTF CDS rows.

    Ensembl uses bare contig names (``1``, ``X``) so we strip the ``chr``
    prefix; GTF is 1-based closed → 0-based half-open.
    """
    bare = chrom.replace("chr", "")
    starts: list[int] = []
    ends: list[int] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5 or parts[0] != bare:
                continue
            if parts[2] != "CDS":
                continue
            starts.append(int(parts[3]) - 1)
            ends.append(int(parts[4]))
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def load_rmsk(path: str, chrom: str):
    """Parse UCSC ``rmsk.txt.gz`` (genoName=col5, genoStart=col6, genoEnd=col7)."""
    starts: list[int] = []
    ends: list[int] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            if parts[5] != chrom:
                continue
            starts.append(int(parts[6]))
            ends.append(int(parts[7]))
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def load_blacklist(path: str | None, chrom: str):
    """Parse the (possibly gzipped) blacklist BED; empty if path is falsy."""
    if not path:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts: list[int] = []
    ends: list[int] = []
    with _opener(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def load_n_positions(fasta_path: str, chrom: str, chrom_length: int) -> np.ndarray:
    """Return uint8 array of length ``chrom_length`` marking N-bases as 1."""
    try:
        from pyfaidx import Fasta
    except ImportError as e:
        raise RuntimeError(
            "pyfaidx is required for N-base masking. "
            "Install via `pip install pyfaidx`."
        ) from e

    fa = Fasta(fasta_path, sequence_always_upper=False, as_raw=True)
    if chrom not in fa:
        raise KeyError(
            f"Chromosome {chrom!r} not found in FASTA {fasta_path}. "
            f"First 5 keys: {list(fa.keys())[:5]}"
        )
    seq_str = str(fa[chrom][:])
    if len(seq_str) != chrom_length:
        log.warning(
            "FASTA length for %s (%d) != chrom_sizes (%d); using FASTA length",
            chrom, len(seq_str), chrom_length,
        )

    arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
    n_mask = np.zeros(len(arr), dtype=np.uint8)
    for code in N_BASE_CODES:
        n_mask |= (arr == code).astype(np.uint8)
    if len(n_mask) < chrom_length:
        out = np.zeros(chrom_length, dtype=np.uint8)
        out[: len(n_mask)] = n_mask
        return out
    if len(n_mask) > chrom_length:
        return n_mask[:chrom_length].copy()
    return n_mask


def paint_intervals(
    mask: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    flank: int = 0,
    chrom_length: int | None = None,
    value: int = 1,
) -> None:
    """In-place: ``mask[s-flank:e+flank] = value`` for each interval, clipped."""
    if chrom_length is None:
        chrom_length = len(mask)
    if len(starts) == 0:
        return
    s = np.clip(starts - flank, 0, chrom_length).astype(np.int64)
    e = np.clip(ends + flank, 0, chrom_length).astype(np.int64)
    for a, b in zip(s, e):
        if a < b:
            mask[a:b] = value


def interval_overlaps_any(
    cand_starts: np.ndarray,
    cand_ends: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """Bool array per candidate interval: True iff it overlaps any covered base.

    Vectorized via cumsum of coverage — O(L + n).
    """
    if len(cand_starts) == 0:
        return np.zeros(0, dtype=bool)
    L = len(coverage)
    cumsum = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(coverage.astype(np.int64), out=cumsum[1:])
    s = np.clip(cand_starts, 0, L).astype(np.int64)
    e = np.clip(cand_ends, 0, L).astype(np.int64)
    return (cumsum[e] - cumsum[s]) > 0


def longest_positive_stretches(
    enhancer_label: np.ndarray, top_n: int = 5,
) -> list[dict]:
    """Top-N longest contiguous positive runs, each ``{start, end, length}``."""
    if enhancer_label.size == 0:
        return []
    pad = np.concatenate(([0], enhancer_label.astype(np.int8), [0]))
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return []
    lengths = ends - starts
    order = np.argsort(-lengths, kind="stable")[:top_n]
    return [
        {
            "start": int(starts[i]),
            "end": int(ends[i]),
            "length": int(lengths[i]),
        }
        for i in order
    ]


def build_one_chrom(chrom: str, chrom_length: int, args, paths: dict) -> dict:
    """Build labels for one chromosome. Returns a stats dict for JSON sidecar."""
    t0 = time.time()
    log.info("[%s] L=%d — loading annotations ...", chrom, chrom_length)

    ccre_s, ccre_e, ccre_c = load_ccre_bed(paths["ccre_bed"], chrom)
    log.info("  cCRE: %d intervals", len(ccre_s))

    peak_s, peak_e = load_h3k27ac_peaks(paths["h3k27ac_peaks"], chrom)
    log.info("  H3K27ac peaks: %d", len(peak_s))

    cds_s, cds_e = load_gtf_cds(paths["gtf"], chrom)
    log.info("  CDS intervals: %d", len(cds_s))

    rmsk_s, rmsk_e = load_rmsk(paths["rmsk"], chrom)
    log.info("  RepeatMasker intervals: %d", len(rmsk_s))

    bl_s, bl_e = load_blacklist(args.blacklist_bed, chrom)
    log.info("  Blacklist intervals: %d", len(bl_s))

    log.info("  Reading N positions from FASTA ...")
    n_mask = load_n_positions(paths["fasta"], chrom, chrom_length)
    n_n_bases = int(n_mask.sum())
    log.info("  N bases: %d", n_n_bases)

    peak_coverage = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(peak_coverage, peak_s, peak_e,
                    chrom_length=chrom_length, value=1)

    is_positive_class = np.isin(
        ccre_c, np.array(list(POSITIVE_CCRE_CLASSES), dtype=object),
    )
    cand_s = ccre_s[is_positive_class]
    cand_e = ccre_e[is_positive_class]
    cand_class = ccre_c[is_positive_class]
    overlap = interval_overlaps_any(cand_s, cand_e, peak_coverage)
    pos_s = cand_s[overlap]
    pos_e = cand_e[overlap]
    pos_class = cand_class[overlap]
    log.info(
        "  Positive cCREs: %d / %d candidate (pELS+dELS) "
        "(pELS=%d, dELS=%d kept)",
        int(overlap.sum()), len(cand_s),
        int((pos_class == "pELS").sum()), int((pos_class == "dELS").sum()),
    )

    enhancer_label = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(enhancer_label, pos_s, pos_e,
                    chrom_length=chrom_length, value=1)
    n_positive_pre_mask = int(enhancer_label.sum())

    pls_s = ccre_s[ccre_c == PLS_CCRE_CLASS]
    pls_e = ccre_e[ccre_c == PLS_CCRE_CLASS]
    mask_pls = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(mask_pls, pls_s, pls_e, flank=args.pls_flank,
                    chrom_length=chrom_length, value=1)

    mask_cds = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(mask_cds, cds_s, cds_e,
                    chrom_length=chrom_length, value=1)

    mask_rmsk = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(mask_rmsk, rmsk_s, rmsk_e,
                    chrom_length=chrom_length, value=1)

    mask_blacklist = np.zeros(chrom_length, dtype=np.uint8)
    paint_intervals(mask_blacklist, bl_s, bl_e,
                    chrom_length=chrom_length, value=1)

    union_mask = (
        mask_pls | mask_cds | mask_rmsk | n_mask | mask_blacklist
    )
    callable_mask = (1 - union_mask).astype(np.uint8)

    # Mask wins over positive.
    enhancer_label[callable_mask == 0] = 0

    n_pls = int(mask_pls.sum())
    n_cds = int(mask_cds.sum())
    n_rmsk = int(mask_rmsk.sum())
    n_blacklist = int(mask_blacklist.sum())
    n_n = int(n_mask.sum())
    n_union = int(union_mask.sum())

    # Exclusive contribution: bases masked by `src` AND by NO sibling source.
    # OR the four *sibling* sources (exclude `src` by identity), then
    # `src & ~siblings` — do not AND with `~src` (would zero out overlaps).
    all_sources = (mask_pls, mask_cds, mask_rmsk, n_mask, mask_blacklist)

    def _exclusive(src):
        src_b = src.astype(bool)
        others_union = np.zeros_like(src_b, dtype=bool)
        for m in all_sources:
            if m is src:
                continue
            others_union |= m.astype(bool)
        return int((src_b & ~others_union).sum())

    excl_pls = _exclusive(mask_pls)
    excl_cds = _exclusive(mask_cds)
    excl_rmsk = _exclusive(mask_rmsk)
    excl_blacklist = _exclusive(mask_blacklist)
    excl_n = _exclusive(n_mask)

    excl_sum = excl_pls + excl_cds + excl_rmsk + excl_blacklist + excl_n
    assert excl_sum <= n_union, (
        f"[{chrom}] exclusive counts sum {excl_sum} exceeds union total "
        f"{n_union} — _exclusive() logic broken"
    )

    n_callable = int(callable_mask.sum())
    n_positive_post_mask = int(enhancer_label.sum())

    arr = np.stack([enhancer_label, callable_mask], axis=1)
    assert arr.shape == (chrom_length, 2) and arr.dtype == np.uint8

    out_path = os.path.join(args.output_dir, f"{chrom}.npy")
    np.save(out_path, arr)
    file_mb = os.path.getsize(out_path) / 1e6
    log.info("  Wrote %s (%.1f MB)", out_path, file_mb)

    top_pos = longest_positive_stretches(enhancer_label, top_n=5)
    if top_pos:
        log.info("  Top-5 longest positive stretches:")
        for r in top_pos:
            log.info("    %s:%d-%d  length=%d",
                     chrom, r["start"], r["end"], r["length"])

    elapsed = time.time() - t0
    log.info(
        "[%s] callable=%d (%.2f%% of L); positive=%d (%.2f%% of callable); %.1fs",
        chrom, n_callable, 100 * n_callable / chrom_length,
        n_positive_post_mask,
        100 * n_positive_post_mask / max(n_callable, 1),
        elapsed,
    )

    return {
        "chrom": chrom,
        "chrom_length": int(chrom_length),
        "n_callable": n_callable,
        "callable_fraction": float(n_callable / chrom_length),
        "n_positive_post_mask": n_positive_post_mask,
        "positive_fraction_of_callable": (
            float(n_positive_post_mask / max(n_callable, 1))
        ),
        "n_positive_pre_mask": n_positive_pre_mask,
        "positive_ccre_intervals_kept": int(overlap.sum()),
        "positive_ccre_pels": int((pos_class == "pELS").sum()),
        "positive_ccre_dels": int((pos_class == "dELS").sum()),
        "candidate_ccre_intervals": int(len(cand_s)),
        "mask_sources": {
            "pls_plus_flank": {"total": n_pls, "exclusive": excl_pls,
                               "n_intervals": int(len(pls_s)),
                               "flank_bp": int(args.pls_flank)},
            "cds": {"total": n_cds, "exclusive": excl_cds,
                    "n_intervals": int(len(cds_s))},
            "rmsk": {"total": n_rmsk, "exclusive": excl_rmsk,
                     "n_intervals": int(len(rmsk_s))},
            "n_bases": {"total": n_n, "exclusive": excl_n},
            "blacklist": {"total": n_blacklist, "exclusive": excl_blacklist,
                          "n_intervals": int(len(bl_s)),
                          "path": args.blacklist_bed or None},
            "union": n_union,
        },
        "h3k27ac_peaks_in_chrom": int(len(peak_s)),
        "top_5_positive_stretches": top_pos,
        "output_path": str(out_path),
        "elapsed_seconds": float(elapsed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chroms", default=DEFAULT_CHROMS,
        help="Comma-separated chromosomes (default chr1..chr22 + chrX).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Where to write {chrom}.npy and build_summary.json.",
    )
    parser.add_argument(
        "--pls-flank", type=int, default=500,
        help="bp flank added to each PLS interval when masking (default 500).",
    )
    parser.add_argument(
        "--blacklist-bed", default=BLACKLIST_BED_DEFAULT,
        help="ENCODE blacklist BED (pass empty string to disable).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default INFO).",
    )
    parser.add_argument(
        "--fasta", default=HG38_FASTA,
        help=f"hg38 FASTA path (default {HG38_FASTA}).",
    )
    parser.add_argument(
        "--chrom-sizes", default=CHROM_SIZES,
        help=f"hg38.chrom.sizes path (default {CHROM_SIZES}).",
    )
    parser.add_argument(
        "--ccre-bed", default=CCRE_BED,
        help=f"ENCODE cCRE V3 BED (default {CCRE_BED}).",
    )
    parser.add_argument(
        "--h3k27ac-peaks", default=H3K27AC_PEAKS,
        help=f"K562 H3K27ac narrowPeak gz (default {H3K27AC_PEAKS}).",
    )
    parser.add_argument(
        "--gtf", default=GTF_PATH,
        help=f"Ensembl GTF gz (default {GTF_PATH}).",
    )
    parser.add_argument(
        "--rmsk", default=RMSK_PATH,
        help=f"UCSC rmsk.txt.gz (default {RMSK_PATH}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    log.info("build_enhancer_labels.py starting")
    log.info("  chroms        : %s", args.chroms)
    log.info("  output_dir    : %s", args.output_dir)
    log.info("  pls_flank     : %d bp", args.pls_flank)
    log.info("  blacklist_bed : %s",
             args.blacklist_bed if args.blacklist_bed else "(disabled)")
    log.info("  fasta         : %s", args.fasta)
    log.info("  chrom_sizes   : %s", args.chrom_sizes)
    log.info("  ccre_bed      : %s", args.ccre_bed)
    log.info("  h3k27ac_peaks : %s", args.h3k27ac_peaks)
    log.info("  gtf           : %s", args.gtf)
    log.info("  rmsk          : %s", args.rmsk)

    paths = {
        "fasta": args.fasta,
        "chrom_sizes": args.chrom_sizes,
        "ccre_bed": args.ccre_bed,
        "h3k27ac_peaks": args.h3k27ac_peaks,
        "gtf": args.gtf,
        "rmsk": args.rmsk,
    }
    missing = [(k, v) for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        for k, v in missing:
            log.error("Missing input: --%s %s", k.replace("_", "-"), v)
        return 2
    if args.blacklist_bed and not os.path.exists(args.blacklist_bed):
        log.error("Missing blacklist BED: %s", args.blacklist_bed)
        return 2

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    unknown = [c for c in chroms if c not in chrom_sizes]
    if unknown:
        log.error(
            "Unknown chromosomes (not in %s): %s",
            args.chrom_sizes, unknown,
        )
        return 2

    per_chrom_stats = []
    t_start = time.time()
    for chrom in chroms:
        stats = build_one_chrom(chrom, chrom_sizes[chrom], args, paths)
        per_chrom_stats.append(stats)

    # Build summary.
    summary = {
        "script": "build_enhancer_labels.py",
        "version": "phase1",
        "args": {
            "chroms": chroms,
            "output_dir": str(args.output_dir),
            "pls_flank": int(args.pls_flank),
            "blacklist_bed": args.blacklist_bed or None,
        },
        "inputs": paths,
        "split_convention": {
            "train": [f"chr{i}" for i in list(range(1, 8)) + list(range(11, 21))],
            "validation": ["chr21"],
            "test": ["chr22"],
            "excluded_from_modelling": ["chr8", "chr9", "chr10"],
            "note": "chrX built for completeness only; chrY skipped (K562 female).",
        },
        "per_chrom": per_chrom_stats,
        "total_elapsed_seconds": float(time.time() - t_start),
    }
    summary_path = os.path.join(args.output_dir, "build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary: %s", summary_path)
    log.info(
        "All chromosomes done in %.1fs",
        summary["total_elapsed_seconds"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
