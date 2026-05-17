"""Evaluation orchestrator: load prediction .npz files + strata, run all
metrics from src/eval_metrics.py, write summary JSON + ROC/PR/stratified
figures, and print a comparison table.

Accepts N predictions via ``--predictions name=path.npz`` pairs.
``enhancer_label`` and ``callable_mask`` are asserted byte-identical
across inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


# Path shim so the script can be run from anywhere.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_metrics import (  # noqa: E402
    per_base_auroc,
    per_base_auprc,
    per_base_curve_points,
    per_base_threshold_sweep,
    peak_level_recall_precision,
    stratified_auroc,
)


BASE = "/gpfs/data/zhou-lab/haoyanghu"
DEFAULT_OUTPUT_DIR = f"{BASE}/projects/enhancer_prediction/eval"
DEFAULT_STRATA = f"{BASE}/projects/enhancer_prediction/data/labels/chr10.strata.npy"

DEFAULT_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
MATCH_RULES = ("anyoverlap", "reciprocal50")


log = logging.getLogger(__name__)


@dataclass
class Prediction:
    name: str
    path: str
    enhancer_prob: np.ndarray   # float32 (L,)
    enhancer_label: np.ndarray  # uint8   (L,)
    callable_mask: np.ndarray   # uint8   (L,)
    metadata: dict


def parse_name_path(arg: str) -> tuple[str, str]:
    """Split a single ``name=path`` argument on the FIRST ``=`` only."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"expected name=path; got {arg!r} (no '=' found)"
        )
    name, _, path = arg.partition("=")
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"empty name in {arg!r}")
    if not path:
        raise argparse.ArgumentTypeError(f"empty path in {arg!r}")
    return name, path


def load_prediction(name: str, path: str) -> Prediction:
    """Load + validate one prediction ``.npz``. Fails loudly on schema drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] prediction file not found: {path}")
    log.info("Loading %s ← %s", name, path)
    data = np.load(path, allow_pickle=True)
    expected = {"enhancer_prob", "enhancer_label", "callable_mask", "metadata"}
    missing = expected - set(data.files)
    if missing:
        raise ValueError(
            f"[{name}] {path} missing required keys: {sorted(missing)} "
            f"(has {sorted(data.files)})"
        )
    prob = data["enhancer_prob"]
    label = data["enhancer_label"]
    mask = data["callable_mask"]
    if prob.dtype != np.float32:
        raise ValueError(
            f"[{name}] enhancer_prob dtype is {prob.dtype}, expected float32"
        )
    if label.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError(
            f"[{name}] label/mask dtype is "
            f"{label.dtype}/{mask.dtype}, expected uint8/uint8"
        )
    if not (prob.shape == label.shape == mask.shape) or prob.ndim != 1:
        raise ValueError(
            f"[{name}] shape mismatch / not 1D: prob={prob.shape}, "
            f"label={label.shape}, mask={mask.shape}"
        )
    # metadata is a 0-d object scalar — JSON-decode if possible.
    meta_raw = data["metadata"]
    try:
        meta = json.loads(meta_raw.item())
        if not isinstance(meta, dict):
            meta = {"raw": meta}
    except Exception:
        meta = {"raw": str(meta_raw)}
    return Prediction(
        name=name, path=str(path),
        enhancer_prob=prob, enhancer_label=label,
        callable_mask=mask, metadata=meta,
    )


def assert_labels_consistent(preds: list[Prediction]) -> None:
    """Assert all predictions share byte-identical labels + callable_mask."""
    if len(preds) <= 1:
        return
    ref = preds[0]
    for p in preds[1:]:
        if p.enhancer_label.shape != ref.enhancer_label.shape:
            raise ValueError(
                f"label shape mismatch between {ref.name} "
                f"{ref.enhancer_label.shape} and {p.name} "
                f"{p.enhancer_label.shape}"
            )
        if not np.array_equal(p.enhancer_label, ref.enhancer_label):
            n_diff = int((p.enhancer_label != ref.enhancer_label).sum())
            raise ValueError(
                f"enhancer_label differs between {ref.name} and {p.name} "
                f"at {n_diff} positions — not from the same Phase-1 build"
            )
        if not np.array_equal(p.callable_mask, ref.callable_mask):
            n_diff = int((p.callable_mask != ref.callable_mask).sum())
            raise ValueError(
                f"callable_mask differs between {ref.name} and {p.name} "
                f"at {n_diff} positions — not from the same Phase-1 build"
            )
    log.info("labels + callable_mask byte-identical across %d predictions",
             len(preds))


def compute_all(
    pred: Prediction,
    strata: np.ndarray,
    thresholds: tuple[float, ...],
    curve_points: int,
) -> dict:
    """Run every metric for one prediction. Returns a JSON-ready dict."""
    log.info("[%s] computing metrics ...", pred.name)
    probs = pred.enhancer_prob
    labels = pred.enhancer_label
    mask = pred.callable_mask

    auroc = per_base_auroc(probs, labels, mask)
    auprc = per_base_auprc(probs, labels, mask)
    log.info("[%s]   per-base AUROC=%.4f  AUPRC=%.4f", pred.name, auroc, auprc)

    sweep = per_base_threshold_sweep(probs, labels, mask, thresholds)
    for row in sweep:
        log.info("[%s]   per-base @t=%.2f  prec=%.4f  rec=%.4f  f1=%.4f  "
                 "n_pred_pos=%d",
                 pred.name, row["threshold"], row["precision"], row["recall"],
                 row["f1"], row["n_pred_positive"])

    peak = peak_level_recall_precision(
        probs, labels, mask, thresholds, match_rules=MATCH_RULES,
    )
    for t in thresholds:
        diag = peak[t]["diagnostics"]
        ao = peak[t]["anyoverlap"]
        r50 = peak[t]["reciprocal50"]
        log.info("[%s]   peak @t=%.2f  n_pred=%d n_gt=%d  "
                 "mean_width=%.1f median_width=%.1f  frac_above=%.4f",
                 pred.name, t, diag["n_pred_peaks"], diag["n_gt_peaks"],
                 diag["pred_peak_mean_width"], diag["pred_peak_median_width"],
                 diag["frac_callable_above_threshold"])
        log.info("[%s]   peak @t=%.2f  anyoverlap   rec=%.4f prec=%.4f",
                 pred.name, t, ao["recall"], ao["precision"])
        log.info("[%s]   peak @t=%.2f  reciprocal50 rec=%.4f prec=%.4f",
                 pred.name, t, r50["recall"], r50["precision"])

    strat = stratified_auroc(probs, labels, mask, strata)
    log.info("[%s]   stratified  pELS=%.4f  dELS=%.4f",
             pred.name, strat["pELS"], strat["dELS"])

    curve = per_base_curve_points(probs, labels, mask, n_subsample=curve_points)

    return {
        "name": pred.name,
        "path": pred.path,
        "metadata": pred.metadata,
        "per_base": {
            "auroc": auroc,
            "auprc": auprc,
            "threshold_sweep": sweep,
        },
        "peak_level": {
            str(t): peak[t] for t in thresholds  # str keys for JSON
        },
        "stratified_auroc": strat,
        "curve_points": {
            "roc": {
                "fpr": curve["roc"]["fpr"].tolist(),
                "tpr": curve["roc"]["tpr"].tolist(),
                "thresholds": curve["roc"]["thresholds"].tolist(),
            },
            "pr": {
                "precision": curve["pr"]["precision"].tolist(),
                "recall": curve["pr"]["recall"].tolist(),
                "thresholds": curve["pr"]["thresholds"].tolist(),
            },
        },
    }


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    return plt


def plot_roc(
    per_pred: dict, chance_auroc: float, out_path: Path, chrom: str,
) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in per_pred.items():
        fpr = np.asarray(result["curve_points"]["roc"]["fpr"])
        tpr = np.asarray(result["curve_points"]["roc"]["tpr"])
        auroc = result["per_base"]["auroc"]
        if fpr.size == 0:
            continue
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1,
            label=f"chance (AUROC={chance_auroc:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-base ROC — {chrom}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(
    per_pred: dict, positive_rate: float, out_path: Path, chrom: str,
) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in per_pred.items():
        prec = np.asarray(result["curve_points"]["pr"]["precision"])
        rec = np.asarray(result["curve_points"]["pr"]["recall"])
        auprc = result["per_base"]["auprc"]
        if prec.size == 0:
            continue
        ax.plot(rec, prec, lw=2, label=f"{name} (AUPRC={auprc:.4f})")
    ax.axhline(positive_rate, ls="--", color="grey", lw=1,
               label=f"chance (positive rate={positive_rate:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Per-base PR — {chrom}  (imbalanced; chance = positive rate)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stratified(
    per_pred: dict, out_path: Path, chrom: str,
) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(per_pred.keys())
    x = np.arange(2)
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        s = per_pred[name]["stratified_auroc"]
        bar_y = [s.get("pELS", float("nan")), s.get("dELS", float("nan"))]
        ax.bar(x + i * width - 0.4 + width / 2, bar_y,
               width=width, label=name)
        for xi, yi in zip(x + i * width - 0.4 + width / 2, bar_y):
            if yi == yi:
                ax.text(xi, yi + 0.005, f"{yi:.3f}",
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", color="grey", lw=1, label="chance (AUROC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(["pELS", "dELS"])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Per-base AUROC")
    ax.set_title(f"Stratified per-base AUROC — {chrom}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_comparison_table(per_pred: dict, log_=log) -> None:
    """Print the headline AUROC/AUPRC/peak comparison table."""
    header = (
        f"{'name':<22} {'AUROC':>7} {'AUPRC':>7} "
        f"{'pb_P@.5':>7} {'pb_R@.5':>7} "
        f"{'pk_R@.5(any)':>13} {'pk_P@.5(any)':>13} "
        f"{'pk_R@.5(r50)':>13} "
        f"{'AUROC_pELS':>11} {'AUROC_dELS':>11}"
    )
    sep = "-" * len(header)
    log_.info(sep)
    log_.info(header)
    log_.info(sep)
    for name, r in per_pred.items():
        pb = r["per_base"]
        pb_05 = next((row for row in pb["threshold_sweep"]
                      if abs(row["threshold"] - 0.5) < 1e-9), None)
        pk_05 = r["peak_level"].get("0.5", {})
        pk_05_ao = pk_05.get("anyoverlap", {}) if pk_05 else {}
        pk_05_r50 = pk_05.get("reciprocal50", {}) if pk_05 else {}
        s = r["stratified_auroc"]

        def fmt(v, w=7, prec=4):
            if v is None or (isinstance(v, float) and v != v):
                return f"{'nan':>{w}}"
            return f"{v:>{w}.{prec}f}"

        log_.info(
            f"{name:<22} {fmt(pb['auroc'])} {fmt(pb['auprc'])} "
            f"{fmt(pb_05['precision'] if pb_05 else None)} "
            f"{fmt(pb_05['recall'] if pb_05 else None)} "
            f"{fmt(pk_05_ao.get('recall'), w=13)} "
            f"{fmt(pk_05_ao.get('precision'), w=13)} "
            f"{fmt(pk_05_r50.get('recall'), w=13)} "
            f"{fmt(s.get('pELS'), w=11)} "
            f"{fmt(s.get('dELS'), w=11)}"
        )
    log_.info(sep)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions", nargs="+", required=True, metavar="name=path",
                   help="One or more name=path.npz pairs.")
    p.add_argument("--strata", default=DEFAULT_STRATA,
                   help=f"Path to chr10 cCRE strata side file from "
                        f"build_strata.py (default {DEFAULT_STRATA}).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory for JSON + PNGs "
                        f"(default {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                   help="Comma-separated probability thresholds.")
    p.add_argument("--curve-points", type=int, default=2000,
                   help="Subsample ROC/PR curves to ~N points (default 2000).")
    p.add_argument("--chrom", default="chr10",
                   help="Chromosome label for figure titles / filenames.")
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

    log.info("evaluate.py starting")
    log.info("  predictions  : %s", args.predictions)
    log.info("  strata       : %s", args.strata)
    log.info("  output_dir   : %s", args.output_dir)
    log.info("  thresholds   : %s", args.thresholds)
    log.info("  curve_points : %d", args.curve_points)
    log.info("  chrom        : %s", args.chrom)
    log.info("  seed         : %d", args.seed)

    parsed = [parse_name_path(arg) for arg in args.predictions]
    names = [n for n, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate names in --predictions: {names}")
    preds: list[Prediction] = [load_prediction(n, p) for n, p in parsed]
    assert_labels_consistent(preds)

    if not os.path.exists(args.strata):
        raise FileNotFoundError(
            f"Strata file not found: {args.strata}. Run scripts/build_strata.py first."
        )
    strata = np.load(args.strata)
    log.info("Loaded strata %s  shape=%s dtype=%s  "
             "pELS=%d dELS=%d neither=%d",
             args.strata, strata.shape, strata.dtype,
             int((strata == 1).sum()), int((strata == 2).sum()),
             int((strata == 0).sum()))
    if strata.shape != preds[0].enhancer_label.shape:
        raise ValueError(
            f"strata shape {strata.shape} != predictions length "
            f"{preds[0].enhancer_label.shape}"
        )

    thresholds = tuple(float(s) for s in args.thresholds.split(","))

    # Chance AUPRC = positive rate; identical across predictions per assertion.
    ref = preds[0]
    mask_b = ref.callable_mask.astype(bool)
    n_callable = int(mask_b.sum())
    positive_rate = float(ref.enhancer_label[mask_b].mean()) if n_callable else 0.0
    log.info("chance reference: AUROC=0.5  AUPRC=positive_rate=%.6f "
             "(n_callable=%d, n_positive=%d)",
             positive_rate, n_callable,
             int(ref.enhancer_label[mask_b].sum()))

    per_pred: dict = {}
    for pred in preds:
        per_pred[pred.name] = compute_all(
            pred, strata, thresholds, args.curve_points,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_comparison_table(per_pred)

    summary = {
        "meta": {
            "chrom": args.chrom,
            "thresholds": list(thresholds),
            "match_rules": list(MATCH_RULES),
            "curve_points": int(args.curve_points),
            "seed": int(args.seed),
            "n_callable": n_callable,
            "positive_rate": positive_rate,
            "input_predictions": {n: p for n, p in parsed},
            "strata_path": str(args.strata),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "chance": {
            "auroc": 0.5,
            "auprc": positive_rate,
        },
        "predictions": per_pred,
    }
    summary_path = out_dir / f"{args.chrom}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote %s", summary_path)

    plot_roc(per_pred, chance_auroc=0.5,
             out_path=out_dir / f"{args.chrom}_roc.png", chrom=args.chrom)
    log.info("Wrote %s", out_dir / f"{args.chrom}_roc.png")

    plot_pr(per_pred, positive_rate=positive_rate,
            out_path=out_dir / f"{args.chrom}_pr.png", chrom=args.chrom)
    log.info("Wrote %s", out_dir / f"{args.chrom}_pr.png")

    plot_stratified(
        per_pred, out_path=out_dir / f"{args.chrom}_stratified_auroc.png",
        chrom=args.chrom,
    )
    log.info("Wrote %s", out_dir / f"{args.chrom}_stratified_auroc.png")

    log.info("evaluate.py done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
