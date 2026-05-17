"""Plot per-base precision + recall vs probability threshold from the
chr10 summary JSON. Optional second figure shows peak-level recall by
matching rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


DEFAULT_SUMMARY = "eval/chr10_summary.json"
DEFAULT_OUTPUT = "eval/threshold_sweep.png"
DEFAULT_OUTPUT_PEAK = "eval/peak_recall_by_rule.png"

COLOR_FROM_SCRATCH = "tab:blue"
COLOR_PRETRAINED = "tab:orange"
COLOR_CHANCE = "grey"


log = logging.getLogger(__name__)


def load_summary(path: str | Path) -> dict:
    """Load the chr10 summary JSON and assert required keys are present."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Summary JSON not found: {p}")
    with open(p) as f:
        s = json.load(f)
    expected_top = {"meta", "chance", "predictions"}
    missing = expected_top - set(s.keys())
    if missing:
        raise ValueError(
            f"{p}: missing top-level keys {sorted(missing)}; "
            f"got {sorted(s.keys())}"
        )
    if not s["predictions"]:
        raise ValueError(f"{p}: 'predictions' is empty")
    for name, block in s["predictions"].items():
        if "per_base" not in block:
            raise ValueError(
                f"{p}: prediction '{name}' missing 'per_base' "
                f"(has {sorted(block.keys())})"
            )
        if "threshold_sweep" not in block["per_base"]:
            raise ValueError(
                f"{p}: prediction '{name}' missing 'per_base.threshold_sweep' "
                f"(has {sorted(block['per_base'].keys())})"
            )
    return s


def pick_color(name: str) -> str:
    """Color-by-name convention so plots stay consistent across figures."""
    lname = name.lower()
    if "pretrained" in lname:
        return COLOR_PRETRAINED
    if "from_scratch" in lname or "scratch" in lname:
        return COLOR_FROM_SCRATCH
    return None


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _extract_sweep(prediction: dict) -> tuple[list[float], list[float], list[float]]:
    """Return ``(thresholds, precisions, recalls)`` from one prediction block."""
    sweep = prediction["per_base"]["threshold_sweep"]
    sweep_sorted = sorted(sweep, key=lambda r: r["threshold"])
    thrs = [r["threshold"] for r in sweep_sorted]
    precs = [r.get("precision") if r.get("precision") is not None
             else float("nan") for r in sweep_sorted]
    recs = [r.get("recall") if r.get("recall") is not None
            else float("nan") for r in sweep_sorted]
    return thrs, precs, recs


def make_main_figure(
    summary: dict, out_path: Path,
) -> dict:
    """Two-panel: per-base precision + recall vs threshold."""
    plt = _matplotlib()

    predictions = summary["predictions"]
    positive_rate = float(summary["meta"]["positive_rate"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    def _name_order(n):
        ln = n.lower()
        if "from_scratch" in ln or "scratch" in ln:
            return (0, ln)
        if "pretrained" in ln:
            return (1, ln)
        return (2, ln)

    names = sorted(predictions.keys(), key=_name_order)

    series: dict[str, dict] = {}
    for name in names:
        thrs, precs, recs = _extract_sweep(predictions[name])
        series[name] = {"thresholds": thrs, "precision": precs, "recall": recs,
                         "color": pick_color(name)}

    ref_thrs = series[names[0]]["thresholds"] if names else []

    # Shade between the two lines when two predictions share the threshold grid.
    if len(names) == 2 and series[names[0]]["thresholds"] == series[names[1]]["thresholds"]:
        ya = np.array(series[names[0]]["precision"], dtype=float)
        yb = np.array(series[names[1]]["precision"], dtype=float)
        ymin = np.minimum(ya, yb)
        ymax = np.maximum(ya, yb)
        axL.fill_between(ref_thrs, ymin, ymax, color="grey", alpha=0.10,
                         label="gap (precision)")
    for name in names:
        s = series[name]
        kwargs = {"lw": 2, "marker": "o", "ms": 7, "label": name}
        if s["color"] is not None:
            kwargs["color"] = s["color"]
        axL.plot(s["thresholds"], s["precision"], **kwargs)
    axL.axhline(positive_rate, ls="--", color=COLOR_CHANCE, lw=1,
                label=f"chance (positive rate={positive_rate:.4f})")
    axL.set_xlabel("Probability threshold")
    axL.set_ylabel("Per-base precision")
    axL.set_title("Precision vs threshold")
    axL.set_ylim(0, max(1.0, max(
        (max(s["precision"]) for s in series.values()), default=1.0,
    ) * 1.05))
    axL.grid(True, alpha=0.3)
    axL.legend(loc="best", fontsize=9)

    if len(names) == 2 and series[names[0]]["thresholds"] == series[names[1]]["thresholds"]:
        ya = np.array(series[names[0]]["recall"], dtype=float)
        yb = np.array(series[names[1]]["recall"], dtype=float)
        ymin = np.minimum(ya, yb)
        ymax = np.maximum(ya, yb)
        axR.fill_between(ref_thrs, ymin, ymax, color="grey", alpha=0.10,
                         label="gap (recall)")
    for name in names:
        s = series[name]
        kwargs = {"lw": 2, "marker": "o", "ms": 7, "label": name}
        if s["color"] is not None:
            kwargs["color"] = s["color"]
        axR.plot(s["thresholds"], s["recall"], **kwargs)
    axR.set_xlabel("Probability threshold")
    axR.set_ylabel("Per-base recall")
    axR.set_title("Recall vs threshold")
    axR.set_ylim(0, 1.05)
    axR.grid(True, alpha=0.3)
    axR.legend(loc="best", fontsize=9)

    fig.suptitle("K562 enhancer prediction — operating-point comparison on chr10",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {name: {
        "thresholds": series[name]["thresholds"],
        "precision": series[name]["precision"],
        "recall": series[name]["recall"],
    } for name in names}


def make_peak_figure(summary: dict, out_path: Path) -> None:
    """Peak-level recall vs threshold, one panel per match rule."""
    plt = _matplotlib()
    predictions = summary["predictions"]
    rules = ("anyoverlap", "reciprocal50")

    first_name = next(iter(predictions))
    first_keys = predictions[first_name]["peak_level"].keys()
    thrs = sorted(float(t) for t in first_keys)

    def _name_order(n):
        ln = n.lower()
        if "from_scratch" in ln or "scratch" in ln:
            return (0, ln)
        if "pretrained" in ln:
            return (1, ln)
        return (2, ln)
    names = sorted(predictions.keys(), key=_name_order)

    fig, axes = plt.subplots(1, len(rules), figsize=(13, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, rule in zip(axes, rules):
        for name in names:
            recalls = []
            for t in thrs:
                # peak_level keys are stringified floats — try both forms.
                blk = (predictions[name]["peak_level"].get(str(t))
                       or predictions[name]["peak_level"].get(repr(t)))
                if blk is None:
                    recalls.append(float("nan"))
                    continue
                rule_blk = blk.get(rule, {})
                r = rule_blk.get("recall")
                recalls.append(float(r) if r is not None else float("nan"))
            color = pick_color(name)
            kw = {"lw": 2, "marker": "o", "ms": 7, "label": name}
            if color is not None:
                kw["color"] = color
            ax.plot(thrs, recalls, **kw)
        ax.set_xlabel("Probability threshold")
        ax.set_ylabel("Peak-level recall")
        ax.set_title(f"{rule}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("K562 enhancer prediction — peak-level recall by matching rule (chr10)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--summary-json", default=DEFAULT_SUMMARY,
                   help=f"Path to chr10_summary.json (default {DEFAULT_SUMMARY}).")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output PNG (precision + recall) "
                        f"(default {DEFAULT_OUTPUT}).")
    p.add_argument("--output-peak", default=DEFAULT_OUTPUT_PEAK,
                   help=f"Output PNG for the optional peak-level figure "
                        f"(default {DEFAULT_OUTPUT_PEAK}).")
    p.add_argument("--skip-peak-figure", action="store_true",
                   help="Skip the optional peak-level recall figure.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    log.info("plot_threshold_sweep.py starting")
    log.info("  summary_json : %s", args.summary_json)
    log.info("  output       : %s", args.output)
    log.info("  output_peak  : %s%s", args.output_peak,
             " (skipped)" if args.skip_peak_figure else "")

    summary = load_summary(args.summary_json)
    positive_rate = float(summary["meta"]["positive_rate"])
    log.info("  positive_rate: %.6f (chance precision)", positive_rate)
    log.info("  predictions  : %s", list(summary["predictions"].keys()))

    main_out = Path(args.output)
    series = make_main_figure(summary, main_out)
    log.info("Wrote %s", main_out)

    if not args.skip_peak_figure:
        try:
            peak_out = Path(args.output_peak)
            make_peak_figure(summary, peak_out)
            log.info("Wrote %s", peak_out)
        except Exception as e:
            log.warning("Failed to write peak-level figure: %s", e)

    header = f"{'model':<25} {'threshold':>10} {'precision':>10} {'recall':>10}"
    log.info("-" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for name, s in series.items():
        for t, pr, rc in zip(s["thresholds"], s["precision"], s["recall"]):
            log.info("%-25s %10.2f %10.4f %10.4f", name, t,
                     pr if pr == pr else float("nan"),
                     rc if rc == rc else float("nan"))
    log.info("-" * len(header))

    return 0


if __name__ == "__main__":
    sys.exit(main())
