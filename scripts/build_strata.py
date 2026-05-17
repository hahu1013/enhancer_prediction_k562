"""Build a per-position cCRE-category int8 array for stratified-AUROC.

Encoding: 0=neither, 1=pELS, 2=dELS. pELS wins if a position is somehow
covered by both. Aborts if either class is empty on the chromosome (BED
likely not cCRE V3).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_CCRE_BED = f"{BASE}/data/annotations/GRCh38-cCREs.bed"
DEFAULT_CHROM_SIZES = f"{BASE}/data/reference/hg38.chrom.sizes"
DEFAULT_OUTPUT_DIR = f"{BASE}/projects/enhancer_prediction/data/labels"

PELS_CODE = np.int8(1)
DELS_CODE = np.int8(2)
NEITHER_CODE = np.int8(0)


log = logging.getLogger(__name__)


def load_chrom_sizes(path: str) -> dict[str, int]:
    """Read ``hg38.chrom.sizes`` (2-col TSV: chrom, length)."""
    sizes: dict[str, int] = {}
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


def load_ccre_for_chrom(
    path: str, chrom: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Parse cCRE V3 BED, returning ``(pELS_intervals, dELS_intervals)``.

    Other classifications are ignored.
    """
    pels: list[tuple[int, int]] = []
    dels: list[tuple[int, int]] = []
    n_other = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[0] != chrom:
                continue
            cls = parts[5]
            s = int(parts[1])
            e = int(parts[2])
            if cls == "pELS":
                pels.append((s, e))
            elif cls == "dELS":
                dels.append((s, e))
            else:
                n_other += 1
    log.info("  %s: pELS=%d intervals, dELS=%d intervals, other=%d skipped",
             chrom, len(pels), len(dels), n_other)
    return pels, dels


def build_strata_array(
    chrom_length: int,
    pels_intervals: list[tuple[int, int]],
    dels_intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    """Paint dELS first, then pELS so pELS wins on overlap.

    Returns ``(strata, n_conflicts)`` — n_conflicts counts dELS positions
    overwritten by pELS.
    """
    strata = np.zeros(chrom_length, dtype=np.int8)

    for s, e in dels_intervals:
        a = max(0, s)
        b = min(chrom_length, e)
        if a < b:
            strata[a:b] = DELS_CODE

    n_conflicts = 0
    for s, e in pels_intervals:
        a = max(0, s)
        b = min(chrom_length, e)
        if a < b:
            conflicts_here = int((strata[a:b] == DELS_CODE).sum())
            n_conflicts += conflicts_here
            strata[a:b] = PELS_CODE

    return strata, n_conflicts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--chrom", default="chr10",
                   help="Chromosome to build strata for (default chr10).")
    p.add_argument("--ccre-bed", default=DEFAULT_CCRE_BED,
                   help=f"ENCODE cCRE V3 BED (default {DEFAULT_CCRE_BED}).")
    p.add_argument("--chrom-sizes", default=DEFAULT_CHROM_SIZES,
                   help=f"hg38.chrom.sizes path (default {DEFAULT_CHROM_SIZES}).")
    p.add_argument("--output", default=None,
                   help="Output .npy path (default "
                        f"{DEFAULT_OUTPUT_DIR}/{{chrom}}.strata.npy).")
    p.add_argument("--output-json", default=None,
                   help="JSON sidecar path (default <output>.replace('.npy', "
                        "'.build.json')).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.output is None:
        args.output = os.path.join(
            DEFAULT_OUTPUT_DIR, f"{args.chrom}.strata.npy",
        )
    if args.output_json is None:
        args.output_json = args.output.replace(".npy", ".build.json")
        if args.output_json == args.output:
            args.output_json = args.output + ".build.json"

    log.info("build_strata.py starting")
    log.info("  chrom        : %s", args.chrom)
    log.info("  ccre_bed     : %s", args.ccre_bed)
    log.info("  chrom_sizes  : %s", args.chrom_sizes)
    log.info("  output       : %s", args.output)
    log.info("  output_json  : %s", args.output_json)

    for k, v in (("--ccre-bed", args.ccre_bed),
                 ("--chrom-sizes", args.chrom_sizes)):
        if not os.path.exists(v):
            log.error("Missing input %s: %s", k, v)
            return 2

    chrom_lens = load_chrom_sizes(args.chrom_sizes)
    if args.chrom not in chrom_lens:
        log.error("Chromosome %s not in %s (first 5 keys: %s)",
                  args.chrom, args.chrom_sizes,
                  list(chrom_lens.keys())[:5])
        return 2
    L = chrom_lens[args.chrom]
    log.info("  %s length: %d", args.chrom, L)

    t0 = time.time()
    log.info("Reading cCRE BED ...")
    pels, dels = load_ccre_for_chrom(args.ccre_bed, args.chrom)

    if not pels or not dels:
        log.error(
            "Zero pELS (%d) or dELS (%d) intervals on %s — cCRE BED likely "
            "isn't V3 schema (expected exact 'pELS'/'dELS' spellings). "
            "Aborting — refusing to write an all-zero strata file.",
            len(pels), len(dels), args.chrom,
        )
        return 3

    log.info("Building strata array (L=%d) ...", L)
    strata, n_conflicts = build_strata_array(L, pels, dels)
    elapsed = time.time() - t0

    n_pels_pos = int((strata == PELS_CODE).sum())
    n_dels_pos = int((strata == DELS_CODE).sum())
    n_neither = int((strata == NEITHER_CODE).sum())
    log.info(
        "  pELS positions: %d (%.4f%% of L)",
        n_pels_pos, 100.0 * n_pels_pos / L,
    )
    log.info(
        "  dELS positions: %d (%.4f%% of L)",
        n_dels_pos, 100.0 * n_dels_pos / L,
    )
    log.info("  neither       : %d (%.4f%% of L)",
             n_neither, 100.0 * n_neither / L)
    if n_conflicts > 0:
        log.warning("  %d positions were originally dELS but overwritten by "
                    "pELS (pELS wins on conflict)", n_conflicts)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, strata)
    sz_mb = out_path.stat().st_size / 1e6
    log.info("Wrote %s (%.1f MB)", out_path, sz_mb)

    sidecar = {
        "script": "build_strata.py",
        "chrom": args.chrom,
        "chrom_length": int(L),
        "input_ccre_bed": args.ccre_bed,
        "input_chrom_sizes": args.chrom_sizes,
        "output": str(out_path),
        "n_pELS_intervals": len(pels),
        "n_dELS_intervals": len(dels),
        "n_pELS_positions": n_pels_pos,
        "n_dELS_positions": n_dels_pos,
        "n_neither_positions": n_neither,
        "n_conflicts_dELS_overwritten_by_pELS": n_conflicts,
        "encoding": {"0": "neither", "1": "pELS", "2": "dELS"},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(elapsed),
    }
    sidecar_path = Path(args.output_json)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("Wrote sidecar %s", sidecar_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
