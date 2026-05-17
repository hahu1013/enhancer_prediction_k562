"""Mask-aware evaluation metrics for the K562 enhancer model.

All metrics drop ``mask == 0`` positions as their first operation — never
zero them, never down-weight them.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np


log = logging.getLogger(__name__)


def _apply_mask(
    probs: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop masked positions; return ``(probs_m, labels_m, n_callable)``.

    Validates that all three arrays are 1D with matching length. Raises
    ``ValueError`` on shape / length mismatch (no silent failures).
    """
    if probs.ndim != 1 or labels.ndim != 1 or mask.ndim != 1:
        raise ValueError(
            f"all inputs must be 1D; got probs.ndim={probs.ndim}, "
            f"labels.ndim={labels.ndim}, mask.ndim={mask.ndim}"
        )
    if not (probs.shape[0] == labels.shape[0] == mask.shape[0]):
        raise ValueError(
            f"length mismatch: probs={probs.shape[0]}, "
            f"labels={labels.shape[0]}, mask={mask.shape[0]}"
        )
    mask_b = mask.astype(bool)
    n_callable = int(mask_b.sum())
    return probs[mask_b], labels[mask_b], n_callable


def per_base_auroc(
    probs: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> float:
    """Mask-aware per-base AUROC. Returns nan if only one class is present."""
    from sklearn.metrics import roc_auc_score

    p, l, n_callable = _apply_mask(probs, labels, mask)
    if n_callable == 0:
        log.warning("per_base_auroc: zero callable positions; returning nan")
        return float("nan")
    if len(np.unique(l)) < 2:
        log.warning(
            "per_base_auroc: only one class present after masking "
            "(n_callable=%d, pos=%d); returning nan",
            n_callable, int(l.sum()),
        )
        return float("nan")
    return float(roc_auc_score(l, p))


def per_base_auprc(
    probs: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> float:
    """Mask-aware per-base AUPRC (sklearn ``average_precision_score``)."""
    from sklearn.metrics import average_precision_score

    p, l, n_callable = _apply_mask(probs, labels, mask)
    if n_callable == 0:
        log.warning("per_base_auprc: zero callable positions; returning nan")
        return float("nan")
    if int(l.sum()) == 0:
        log.warning(
            "per_base_auprc: zero positive positions after masking "
            "(n_callable=%d); returning nan",
            n_callable,
        )
        return float("nan")
    return float(average_precision_score(l, p))


def _subsample_aligned(arrs: Sequence[np.ndarray], n: int) -> list[np.ndarray]:
    """Subsample N points uniformly, preserving first and last."""
    N = arrs[0].shape[0]
    if N <= n or n <= 0:
        return [a.copy() for a in arrs]
    idx = np.unique(np.linspace(0, N - 1, n).round().astype(np.int64))
    return [a[idx] for a in arrs]


def per_base_curve_points(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_subsample: int = 2000,
) -> dict:
    """Mask-aware ROC + PR curve points, subsampled for plotting.

    Returns ``{'roc': {fpr, tpr, thresholds}, 'pr': {precision, recall, thresholds}}``.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    p, l, n_callable = _apply_mask(probs, labels, mask)
    if n_callable == 0 or len(np.unique(l)) < 2:
        log.warning("per_base_curve_points: insufficient data; returning empties")
        empty = np.empty(0, dtype=np.float64)
        return {
            "roc": {"fpr": empty, "tpr": empty, "thresholds": empty},
            "pr":  {"precision": empty, "recall": empty, "thresholds": empty},
        }

    fpr, tpr, roc_thr = roc_curve(l, p)
    prec, rec, pr_thr = precision_recall_curve(l, p)
    # precision_recall_curve returns one fewer threshold than precision/recall.
    pr_thr_padded = np.concatenate([pr_thr, [np.nan]])

    fpr_s, tpr_s, roc_thr_s = _subsample_aligned([fpr, tpr, roc_thr], n_subsample)
    prec_s, rec_s, pr_thr_s = _subsample_aligned(
        [prec, rec, pr_thr_padded], n_subsample,
    )
    return {
        "roc": {"fpr": fpr_s, "tpr": tpr_s, "thresholds": roc_thr_s},
        "pr":  {"precision": prec_s, "recall": rec_s, "thresholds": pr_thr_s},
    }


def per_base_threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict]:
    """Per-base precision/recall/F1 at each threshold (no peak grouping)."""
    p, l, n_callable = _apply_mask(probs, labels, mask)
    out: list[dict] = []
    if n_callable == 0:
        log.warning("per_base_threshold_sweep: zero callable; returning nan rows")
        for t in thresholds:
            out.append({
                "threshold": float(t), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "n_pred_positive": 0,
            })
        return out

    pos_total = int(l.sum())
    for t in thresholds:
        pred = p >= float(t)
        tp = int((pred & (l == 1)).sum())
        fp = int((pred & (l == 0)).sum())
        fn = pos_total - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        rec = float(tp / pos_total) if pos_total > 0 else float("nan")
        if prec != prec or rec != rec or (prec + rec) == 0:
            f1 = float("nan")
        else:
            f1 = float(2 * prec * rec / (prec + rec))
        out.append({
            "threshold": float(t),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_pred_positive": int(pred.sum()),
        })
    return out


def extract_runs(binary_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maximal contiguous runs of ``(binary_arr AND mask)`` as (n, 2) int64
    half-open intervals. Masked positions split runs.
    """
    if binary_arr.ndim != 1 or mask.ndim != 1:
        raise ValueError("binary_arr and mask must be 1D")
    if binary_arr.shape[0] != mask.shape[0]:
        raise ValueError(
            f"length mismatch: binary_arr={binary_arr.shape[0]}, "
            f"mask={mask.shape[0]}"
        )
    arr = np.logical_and(binary_arr.astype(bool), mask.astype(bool))
    if not arr.any():
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0].astype(np.int64)
    ends = np.where(diff == -1)[0].astype(np.int64)
    return np.column_stack([starts, ends])


def match_intervals(
    pred_intervals: np.ndarray,
    gt_intervals: np.ndarray,
    rule: str = "anyoverlap",
) -> tuple[np.ndarray, np.ndarray]:
    """Match predicted intervals against ground-truth intervals.

    rule = ``anyoverlap`` (overlap ≥ 1 bp) or ``reciprocal50``
    (overlap ≥ 50 % of both |P| and |G|).
    Returns ``(matched_pred, matched_gt)`` bool arrays indicating, per
    interval, whether at least one match was found.
    """
    if rule not in {"anyoverlap", "reciprocal50"}:
        raise ValueError(f"unknown rule {rule!r}; "
                         f"expected 'anyoverlap' or 'reciprocal50'")
    pred_arr = np.asarray(pred_intervals, dtype=np.int64).reshape(-1, 2)
    gt_arr = np.asarray(gt_intervals, dtype=np.int64).reshape(-1, 2)
    n_p = pred_arr.shape[0]
    n_g = gt_arr.shape[0]
    matched_pred = np.zeros(n_p, dtype=bool)
    matched_gt = np.zeros(n_g, dtype=bool)
    if n_p == 0 or n_g == 0:
        return matched_pred, matched_gt

    pred_order = np.argsort(pred_arr[:, 0], kind="stable")
    gt_order = np.argsort(gt_arr[:, 0], kind="stable")
    pred_sorted = pred_arr[pred_order]
    gt_sorted = gt_arr[gt_order]

    left = 0
    for i in range(n_p):
        ps = int(pred_sorted[i, 0])
        pe = int(pred_sorted[i, 1])
        while left < n_g and int(gt_sorted[left, 1]) <= ps:
            left += 1
        if left == n_g:
            break
        j = left
        while j < n_g and int(gt_sorted[j, 0]) < pe:
            gs = int(gt_sorted[j, 0])
            ge = int(gt_sorted[j, 1])
            overlap = min(pe, ge) - max(ps, gs)
            if overlap > 0:
                if rule == "anyoverlap":
                    matched_pred[pred_order[i]] = True
                    matched_gt[gt_order[j]] = True
                else:
                    pw = pe - ps
                    gw = ge - gs
                    # Compare 2 * overlap vs width to avoid floating-point.
                    if overlap * 2 >= pw and overlap * 2 >= gw:
                        matched_pred[pred_order[i]] = True
                        matched_gt[gt_order[j]] = True
            j += 1
    return matched_pred, matched_gt


def peak_level_recall_precision(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    thresholds: Sequence[float],
    match_rules: Sequence[str] = ("anyoverlap", "reciprocal50"),
) -> dict:
    """Peak-level recall + precision at every (threshold × rule) combo.

    Returns a dict keyed by threshold; each value contains one sub-dict
    per rule (recall, precision, counts) plus a ``"diagnostics"`` block
    with predicted-peak widths and fraction-callable-above-threshold —
    the latter flags the "one giant blob" failure mode.
    """
    mask_b = mask.astype(bool)
    n_callable = int(mask_b.sum())
    gt_intervals = extract_runs(labels.astype(bool), mask_b)
    n_gt = int(gt_intervals.shape[0])

    out: dict = {}
    for t in thresholds:
        pred_b = probs >= float(t)
        pred_intervals = extract_runs(pred_b, mask_b)
        n_pred = int(pred_intervals.shape[0])

        n_above = int(np.logical_and(pred_b, mask_b).sum())
        if n_pred > 0:
            widths = pred_intervals[:, 1] - pred_intervals[:, 0]
            mean_w = float(widths.mean())
            median_w = float(np.median(widths))
        else:
            mean_w = 0.0
            median_w = 0.0

        per_t: dict = {
            "diagnostics": {
                "n_pred_peaks": n_pred,
                "n_gt_peaks": n_gt,
                "pred_peak_mean_width": mean_w,
                "pred_peak_median_width": median_w,
                "frac_callable_above_threshold": (
                    float(n_above / n_callable) if n_callable > 0
                    else float("nan")
                ),
                "n_callable_above_threshold": n_above,
                "n_callable": n_callable,
            }
        }
        for rule in match_rules:
            mp, mg = match_intervals(pred_intervals, gt_intervals, rule=rule)
            recall = float(mg.sum() / n_gt) if n_gt > 0 else float("nan")
            precision = float(mp.sum() / n_pred) if n_pred > 0 else float("nan")
            per_t[rule] = {
                "recall": recall,
                "precision": precision,
                "matched_pred": int(mp.sum()),
                "matched_gt": int(mg.sum()),
                "n_pred": n_pred,
                "n_gt": n_gt,
            }
        out[float(t)] = per_t

    return out


def stratified_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    strata: np.ndarray,
) -> dict:
    """Per-base AUROC computed separately for pELS (strata==1) and dELS (==2)."""
    if strata.ndim != 1 or strata.shape[0] != probs.shape[0]:
        raise ValueError(
            f"strata shape {strata.shape} does not match probs shape "
            f"{probs.shape}"
        )
    mask_b = mask.astype(bool)
    out: dict = {}
    for code, name in ((1, "pELS"), (2, "dELS")):
        sub_mask = mask_b & (strata == code)
        out[name] = per_base_auroc(
            probs, labels, sub_mask.astype(np.uint8),
        )
    return out


__all__ = [
    "per_base_auroc",
    "per_base_auprc",
    "per_base_curve_points",
    "per_base_threshold_sweep",
    "extract_runs",
    "match_intervals",
    "peak_level_recall_precision",
    "stratified_auroc",
]


# Smoke tests — run via ``python -m src.eval_metrics``.
def _smoke() -> int:
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(levelname)s %(message)s")

    # ---- extract_runs: masked position should split a run ----
    # callable everywhere, binary array = [0 1 1 1 0 1 1 0 0 1 1 1 1 1]
    bin_arr = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
    mask = np.ones_like(bin_arr)
    runs = extract_runs(bin_arr, mask)
    expected = np.array([[1, 4], [5, 7], [9, 14]], dtype=np.int64)
    assert np.array_equal(runs, expected), (runs, expected)
    print(f"extract_runs (no mask split): {runs.tolist()} OK")

    # Mask out position 2 → should split the first run into [1, 2) and [3, 4)
    mask2 = np.ones_like(bin_arr)
    mask2[2] = 0
    runs2 = extract_runs(bin_arr, mask2)
    # Position 2 is masked, so the AND turns positions 1, 3 into a run [1,2)
    # then a separate run [3, 4)
    expected2 = np.array([[1, 2], [3, 4], [5, 7], [9, 14]], dtype=np.int64)
    assert np.array_equal(runs2, expected2), (runs2, expected2)
    print(f"extract_runs (mask splits run): {runs2.tolist()} OK")

    # All zeros / all masked → empty
    assert extract_runs(np.zeros(10, dtype=np.uint8), np.ones(10)).shape == (0, 2)
    assert extract_runs(np.ones(10, dtype=np.uint8), np.zeros(10)).shape == (0, 2)
    print("extract_runs (empty cases): OK")

    # ---- match_intervals: hand-construct overlapping pairs ----
    # Pred P = [10, 20), GT G = [15, 25). Overlap = 5.
    # |P| = |G| = 10. anyoverlap → yes. reciprocal50 → 5/10 = 0.5 ≥ 0.5 → yes.
    P = np.array([[10, 20]], dtype=np.int64)
    G = np.array([[15, 25]], dtype=np.int64)
    mp_a, mg_a = match_intervals(P, G, "anyoverlap")
    mp_r, mg_r = match_intervals(P, G, "reciprocal50")
    assert mp_a[0] and mg_a[0], "anyoverlap should match O=5"
    assert mp_r[0] and mg_r[0], "reciprocal50 should match O=5 ≥ 0.5 of both"
    print("match: P[10,20) ∩ G[15,25) (O=5): anyoverlap=match, reciprocal50=match")

    # Same P, but G = [18, 22) — overlap = 2. |P|=10, |G|=4. 2/10 = 0.2 < 0.5;
    # 2/4 = 0.5 ≥ 0.5. Reciprocal50 fails (needs BOTH ≥ 0.5). Anyoverlap passes.
    G2 = np.array([[18, 22]], dtype=np.int64)
    mp_a, mg_a = match_intervals(P, G2, "anyoverlap")
    mp_r, mg_r = match_intervals(P, G2, "reciprocal50")
    assert mp_a[0] and mg_a[0]
    assert not mp_r[0] and not mg_r[0], "reciprocal50 must reject lop-sided"
    print("match: P[10,20) ∩ G[18,22) (O=2): anyoverlap=match, reciprocal50=NO_match")

    # Non-overlapping
    G3 = np.array([[30, 40]], dtype=np.int64)
    mp_a, mg_a = match_intervals(P, G3, "anyoverlap")
    assert not mp_a[0] and not mg_a[0]
    print("match: P[10,20) ∩ G[30,40) (no overlap): no match in either rule")

    # Multiple GT per pred — both should be marked
    P4 = np.array([[10, 50]], dtype=np.int64)
    G4 = np.array([[5, 15], [20, 25], [45, 55]], dtype=np.int64)
    mp_a, mg_a = match_intervals(P4, G4, "anyoverlap")
    assert mp_a[0]
    assert mg_a.all(), f"all three GTs should be matched: {mg_a}"
    print("match: P[10,50) covers all 3 GTs under anyoverlap")

    # ---- per_base metrics: tiny synthetic ----
    np.random.seed(42)
    L = 1000
    probs = np.random.rand(L).astype(np.float32)
    labels = (probs > 0.7).astype(np.uint8)  # synthetically calibrated
    mask = np.ones(L, dtype=np.uint8)
    mask[100:200] = 0  # mask out 100 positions
    auroc = per_base_auroc(probs, labels, mask)
    auprc = per_base_auprc(probs, labels, mask)
    assert 0.95 <= auroc <= 1.0, f"AUROC unexpectedly low: {auroc}"
    assert 0.9 <= auprc <= 1.0, f"AUPRC unexpectedly low: {auprc}"
    print(f"per_base metrics on synthetic: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    sweep = per_base_threshold_sweep(probs, labels, mask, [0.1, 0.5, 0.9])
    assert len(sweep) == 3
    assert all("precision" in row and "recall" in row for row in sweep)
    # Recall should drop monotonically as threshold rises (on this synthetic).
    recs = [row["recall"] for row in sweep]
    assert recs[0] >= recs[1] >= recs[2], f"recall not monotonic: {recs}"
    print(f"per_base_threshold_sweep monotonic: recalls = {recs}")

    # ---- peak_level on a tiny synthetic ----
    probs2 = np.zeros(100, dtype=np.float32)
    probs2[10:20] = 0.9
    probs2[40:55] = 0.8
    probs2[80:85] = 0.4
    labels2 = np.zeros(100, dtype=np.uint8)
    labels2[12:18] = 1   # inside the first predicted peak
    labels2[60:70] = 1   # NOT predicted at threshold 0.5
    mask2_pk = np.ones(100, dtype=np.uint8)
    peak = peak_level_recall_precision(
        probs2, labels2, mask2_pk, [0.5, 0.7],
        match_rules=("anyoverlap", "reciprocal50"),
    )
    # At t=0.5: pred = [10,20), [40,55). GT = [12,18), [60,70).
    # anyoverlap: pred[10,20) overlaps GT[12,18) → matched.
    #             pred[40,55) overlaps nothing → unmatched.
    #             GT[12,18) matched; GT[60,70) unmatched.
    # → recall = 1/2 = 0.5, precision = 1/2 = 0.5
    ao_05 = peak[0.5]["anyoverlap"]
    assert ao_05["recall"] == 0.5, ao_05
    assert ao_05["precision"] == 0.5, ao_05
    print(f"peak_level @0.5 anyoverlap: recall=0.5, precision=0.5 OK")
    # reciprocal50: pred[10,20) (w=10), GT[12,18) (w=6), overlap=6.
    # 6/10 = 0.6 ≥ 0.5; 6/6 = 1.0 ≥ 0.5 → MATCH.
    # → recall = 1/2 = 0.5, precision = 1/2 = 0.5
    r50_05 = peak[0.5]["reciprocal50"]
    assert r50_05["recall"] == 0.5 and r50_05["precision"] == 0.5
    print(f"peak_level @0.5 reciprocal50: recall=0.5, precision=0.5 OK")
    diag = peak[0.5]["diagnostics"]
    assert diag["n_pred_peaks"] == 2 and diag["n_gt_peaks"] == 2
    # mean width = (10 + 15) / 2 = 12.5
    assert diag["pred_peak_mean_width"] == 12.5, diag
    print(f"peak_level @0.5 diagnostics: mean_width=12.5, n_pred=2, n_gt=2 OK")

    # ---- stratified ----
    strata = np.zeros(L, dtype=np.int8)
    strata[300:500] = 1  # pELS region
    strata[500:700] = 2  # dELS region
    s = stratified_auroc(probs, labels, mask, strata)
    assert "pELS" in s and "dELS" in s
    print(f"stratified_auroc: pELS={s['pELS']:.4f}, dELS={s['dELS']:.4f}")

    # ---- shape / length error paths ----
    try:
        per_base_auroc(probs, labels, mask[:-1])
    except ValueError as e:
        assert "length mismatch" in str(e)
        print("per_base_auroc length-mismatch error path: OK")

    print("\nEVAL_METRICS SMOKE PASSED")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
