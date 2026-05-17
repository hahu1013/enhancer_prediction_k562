"""Motif extraction from ISM saliency profiles: clustering, PWM building,
logos, and Tomtom-style matching against JASPAR.

PWMs are stored as (4, W) float with rows in ACGT order (matching the
model's one-hot channel order).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np


log = logging.getLogger(__name__)


ALPHABET_ACGT = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET_ACGT)}


def cluster_saliency_profiles(
    saliency: np.ndarray,
    n_clusters: int = 10,
    method: str = "agglomerative",
    linkage: str = "average",
) -> np.ndarray:
    """Cluster (K, W) ``saliency`` rows by cosine distance.

    Only ``method="agglomerative"`` is supported. Returns a (K,) int
    label array. ``n_clusters`` is capped at K with a warning if smaller.
    """
    if saliency.ndim != 2:
        raise ValueError(f"saliency must be 2D; got shape {saliency.shape}")
    K, W = saliency.shape
    if K == 0:
        return np.empty(0, dtype=np.int64)
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1; got {n_clusters}")
    eff_clusters = min(n_clusters, K)
    if eff_clusters < n_clusters:
        log.warning(
            "cluster_saliency_profiles: only %d rows; reducing n_clusters "
            "from %d → %d",
            K, n_clusters, eff_clusters,
        )

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        # Drop zero-norm rows before cosine clustering (NaN distance);
        # they go into a separate catchall cluster id.
        norms = np.linalg.norm(saliency, axis=1)
        nonzero = norms > 1e-12
        if nonzero.all():
            clust = AgglomerativeClustering(
                n_clusters=eff_clusters, metric="cosine", linkage=linkage,
            )
            labels = clust.fit_predict(saliency).astype(np.int64)
        else:
            sub = saliency[nonzero]
            sub_clusters = min(eff_clusters, sub.shape[0]) if sub.shape[0] else 0
            labels = np.full(K, -1, dtype=np.int64)
            if sub_clusters >= 1 and sub.shape[0] >= 1:
                clust = AgglomerativeClustering(
                    n_clusters=sub_clusters, metric="cosine", linkage=linkage,
                )
                labels[nonzero] = clust.fit_predict(sub).astype(np.int64)
            zero_cluster = (sub_clusters if sub_clusters >= 1 else 0)
            labels[~nonzero] = zero_cluster
            log.info(
                "cluster_saliency_profiles: %d zero-saliency rows grouped "
                "into catchall cluster %d (separate from the %d cosine clusters)",
                int((~nonzero).sum()), zero_cluster, sub_clusters,
            )
        return labels
    raise ValueError(f"unknown method {method!r}")


def build_pwm(
    sequences_in_cluster: Sequence[np.ndarray],
    alphabet: str = ALPHABET_ACGT,
    pseudocount: float = 0.25,
) -> dict:
    """PCM + smoothed PWM + consensus + per-position IC for a list of
    equal-length uint8 base-index sequences (0=A, 1=C, 2=G, 3=T).

    Returns a dict with keys ``pcm`` (raw counts), ``pwm`` (smoothed
    columns sum to 1), ``consensus``, ``information_content``,
    ``n_sequences``.
    """
    if len(sequences_in_cluster) == 0:
        raise ValueError("build_pwm: empty sequence list")
    if alphabet != ALPHABET_ACGT:
        raise ValueError(f"alphabet must be {ALPHABET_ACGT!r}; got {alphabet!r}")
    seqs = np.stack([np.asarray(s, dtype=np.uint8)
                      for s in sequences_in_cluster], axis=0)  # (N, W)
    N, W = seqs.shape
    pcm = np.zeros((4, W), dtype=np.int64)
    for b in range(4):
        pcm[b] = (seqs == b).sum(axis=0)

    smoothed = pcm.astype(np.float64) + pseudocount
    col_sum = smoothed.sum(axis=0)
    pwm = (smoothed / col_sum[None, :]).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log2p = np.where(pwm > 0, np.log2(pwm), 0.0)
        entropy = -(pwm * log2p).sum(axis=0)
    information_content = (2.0 - entropy).astype(np.float64)
    information_content = np.clip(information_content, 0.0, 2.0)

    consensus_idx = pwm.argmax(axis=0)
    consensus = "".join(alphabet[i] for i in consensus_idx)

    return {
        "pcm": pcm,
        "pwm": pwm,
        "consensus": consensus,
        "information_content": information_content,
        "n_sequences": int(N),
    }


def pwm_logo(pwm_dict: dict, output_path) -> None:
    """Save a sequence-logo PNG. Heights = IC; falls back to a PWM
    heatmap if ``logomaker`` is missing.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend so this works on clusters
    import matplotlib.pyplot as plt

    pwm = pwm_dict["pwm"]
    ic = pwm_dict["information_content"]
    W = pwm.shape[1]
    output_path = str(output_path)

    try:
        import logomaker
        import pandas as pd
        heights = (pwm.T * ic[:, None])
        df = pd.DataFrame(heights, columns=list(ALPHABET_ACGT))
        fig, ax = plt.subplots(figsize=(min(0.3 * W, 12), 2.4))
        logomaker.Logo(df, ax=ax, color_scheme="classic")
        ax.set_xlabel("Position")
        ax.set_ylabel("Information content (bits)")
        ax.set_title(f"Motif logo (n={pwm_dict['n_sequences']}, "
                     f"consensus={pwm_dict['consensus']})", fontsize=10)
        ax.set_ylim(0, 2.05)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        log.warning("logomaker not available; falling back to PWM heatmap "
                    "at %s", output_path)
        fig, ax = plt.subplots(figsize=(min(0.4 * W, 14), 3))
        im = ax.imshow(pwm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(4))
        ax.set_yticklabels(list(ALPHABET_ACGT))
        ax.set_xlabel("Position")
        ax.set_title(f"PWM (heatmap fallback) — n={pwm_dict['n_sequences']}, "
                     f"consensus={pwm_dict['consensus']}", fontsize=10)
        fig.colorbar(im, ax=ax, label="P(base | position)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def parse_jaspar_pfm_file(path) -> list[dict]:
    """Parse the JASPAR 2024 CORE non-redundant PFM bundle.

    Returns a list of dicts with keys ``matrix_id``, ``name``, ``pcm``,
    ``pwm`` (smoothed, pseudocount=0.25), ``information_content``, ``W``.
    """
    entries: list[dict] = []
    with open(path) as f:
        text = f.read()

    blocks = text.split("\n>")
    for i, block in enumerate(blocks):
        if i == 0:
            block = block.lstrip()
            if not block.startswith(">"):
                if i == 0 and not block.startswith(">"):
                    if ">" not in block:
                        continue
                continue
            header_line = block.splitlines()[0].lstrip(">").strip()
            body = "\n".join(block.splitlines()[1:])
        else:
            lines = block.splitlines()
            header_line = lines[0].strip()
            body = "\n".join(lines[1:])
        if not header_line:
            continue
        parts = header_line.split(None, 1)
        matrix_id = parts[0]
        name = parts[1] if len(parts) > 1 else matrix_id

        # Parse 4 rows (A, C, G, T). Lines: "A  [ 0  3  79 ... ]" or "A 0 3 79 ...".
        rows: dict[str, list[float]] = {}
        for row_line in body.splitlines():
            row_line = row_line.strip()
            if not row_line:
                continue
            if row_line[0] in "ACGT":
                base = row_line[0]
                rest = row_line[1:].replace("[", "").replace("]", "").strip()
                try:
                    counts = [float(x) for x in rest.split()]
                except ValueError:
                    continue
                rows[base] = counts
        if set(rows.keys()) != set("ACGT") or len({len(v) for v in rows.values()}) != 1:
            log.debug("parse_jaspar: skipping malformed entry %r", matrix_id)
            continue
        W = len(rows["A"])
        pcm = np.zeros((4, W), dtype=np.float64)
        for b_idx, b in enumerate("ACGT"):
            pcm[b_idx] = rows[b]
        smoothed = pcm + 0.25
        pwm = smoothed / smoothed.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2p = np.where(pwm > 0, np.log2(pwm), 0.0)
            ic = (2.0 - (-(pwm * log2p).sum(axis=0)))
        ic = np.clip(ic, 0.0, 2.0)
        entries.append({
            "matrix_id": matrix_id,
            "name": name,
            "pcm": pcm,
            "pwm": pwm,
            "information_content": ic,
            "W": int(W),
        })
    return entries


def _ic_weighted_pearson(a: np.ndarray, b: np.ndarray, ic_a: np.ndarray,
                          ic_b: np.ndarray) -> float:
    """IC-weighted mean of per-column Pearson correlations between two
    (4, W) PWMs. Weight at col p is ``0.5 * (ic_a[p] + ic_b[p])``.
    """
    W = a.shape[1]
    if W == 0:
        return float("nan")
    col_corrs = np.zeros(W, dtype=np.float64)
    col_weights = np.zeros(W, dtype=np.float64)
    for p in range(W):
        av = a[:, p]
        bv = b[:, p]
        if av.std() == 0 or bv.std() == 0:
            col_corrs[p] = 0.0
        else:
            col_corrs[p] = float(np.corrcoef(av, bv)[0, 1])
        col_weights[p] = 0.5 * (ic_a[p] + ic_b[p])
    total_w = col_weights.sum()
    if total_w <= 0:
        return float(col_corrs.mean()) if col_corrs.size else float("nan")
    return float((col_corrs * col_weights).sum() / total_w)


def tomtom_style_match(
    cluster_pwm: dict,
    cluster_ic: np.ndarray,
    jaspar_entry: dict,
    max_offset: int = 5,
) -> dict:
    """Best IC-weighted Pearson correlation across offsets in ``±max_offset``.

    Returns ``{'score', 'offset', 'overlap'}`` for the best offset.
    """
    a = cluster_pwm
    b = jaspar_entry["pwm"]
    ic_b = jaspar_entry["information_content"]
    W_a = a.shape[1]
    W_b = b.shape[1]
    min_overlap = max(1, min(W_a, W_b) - 3)

    best = {"score": -np.inf, "offset": 0, "overlap": 0}
    lo = max(-max_offset, -(W_b - min_overlap))
    hi = min(max_offset, W_a - min_overlap)
    for offset in range(lo, hi + 1):
        a_lo = max(0, offset)
        a_hi = min(W_a, offset + W_b)
        if a_hi - a_lo < min_overlap:
            continue
        b_lo = a_lo - offset
        b_hi = a_hi - offset
        sub_a = a[:, a_lo:a_hi]
        sub_b = b[:, b_lo:b_hi]
        sub_ic_a = cluster_ic[a_lo:a_hi]
        sub_ic_b = ic_b[b_lo:b_hi]
        score = _ic_weighted_pearson(sub_a, sub_b, sub_ic_a, sub_ic_b)
        if score > best["score"]:
            best = {"score": float(score), "offset": int(offset),
                    "overlap": int(a_hi - a_lo)}
    if best["score"] == -np.inf:
        best["score"] = float("nan")
    return best


def match_cluster_against_jaspar(
    cluster_pwm: np.ndarray,
    cluster_ic: np.ndarray,
    jaspar_entries: list[dict],
    top_n: int = 3,
    max_offset: int = 5,
) -> list[dict]:
    """Top-``top_n`` JASPAR matches for a cluster PWM, sorted by score desc."""
    matches: list[dict] = []
    for entry in jaspar_entries:
        m = tomtom_style_match(cluster_pwm, cluster_ic, entry,
                                max_offset=max_offset)
        matches.append({
            "matrix_id": entry["matrix_id"],
            "name": entry["name"],
            "score": m["score"],
            "offset": m["offset"],
            "overlap": m["overlap"],
        })
    matches.sort(
        key=lambda m: (m["score"] if m["score"] == m["score"] else -np.inf),
        reverse=True,
    )
    return matches[:top_n]


__all__ = [
    "cluster_saliency_profiles",
    "build_pwm",
    "pwm_logo",
    "parse_jaspar_pfm_file",
    "tomtom_style_match",
    "match_cluster_against_jaspar",
    "ALPHABET_ACGT",
]


# Smoke tests — ``python -m src.motifs``.
def _smoke() -> int:
    import tempfile, os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- cluster_saliency_profiles: 30 rows with two distinct shapes ----
    np.random.seed(0)
    K = 30; W = 11
    # 15 profiles peaked at center; 15 peaked at right edge.
    sal_a = np.zeros((15, W), dtype=np.float32); sal_a[:, 5] = 1.0
    sal_a += np.random.rand(15, W).astype(np.float32) * 0.05
    sal_b = np.zeros((15, W), dtype=np.float32); sal_b[:, 9] = 1.0
    sal_b += np.random.rand(15, W).astype(np.float32) * 0.05
    sal = np.vstack([sal_a, sal_b])
    labels = cluster_saliency_profiles(sal, n_clusters=2, method="agglomerative")
    # The 15 type-A should land in one cluster, 15 type-B in the other.
    from collections import Counter
    a_lbls = Counter(labels[:15]); b_lbls = Counter(labels[15:])
    print(f"  type-A labels: {dict(a_lbls)}; type-B labels: {dict(b_lbls)}")
    assert len(a_lbls) == 1 and len(b_lbls) == 1
    assert set(a_lbls.keys()) != set(b_lbls.keys()), "two distinct shapes should yield two clusters"
    print(f"cluster_saliency_profiles separates 2 shapes correctly")

    # n_clusters > K → reduces gracefully
    lbl = cluster_saliency_profiles(sal[:3], n_clusters=10)
    assert len(set(lbl)) <= 3
    print(f"cluster_saliency_profiles reduces n_clusters when K is small")

    # ---- build_pwm ----
    # Hand-construct: a "GATA" motif — 10 sequences with GATA at positions 0-3
    # plus random noise at positions 4-10.
    seqs = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = rng.integers(0, 4, size=11, dtype=np.uint8)
        s[0:4] = [2, 0, 3, 0]  # G, A, T, A
        seqs.append(s)
    pwm_dict = build_pwm(seqs)
    assert pwm_dict["consensus"][:4] == "GATA", pwm_dict["consensus"]
    assert pwm_dict["pcm"].shape == (4, 11) and pwm_dict["pwm"].shape == (4, 11)
    # Each PWM column sums to 1.0
    assert np.allclose(pwm_dict["pwm"].sum(axis=0), 1.0)
    # IC at the conserved positions should be high (~2.0)
    assert pwm_dict["information_content"][0] > 1.5, pwm_dict["information_content"]
    print(f"build_pwm: consensus={pwm_dict['consensus']!r}, "
          f"IC at pos 0={pwm_dict['information_content'][0]:.3f}")

    # ---- pwm_logo (test the matplotlib fallback path; logomaker may or may not be present) ----
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "logo.png")
        pwm_logo(pwm_dict, out)
        assert os.path.exists(out) and os.path.getsize(out) > 1000
        print(f"pwm_logo wrote {os.path.getsize(out):,} bytes to {out}")

    # ---- parse_jaspar_pfm_file + tomtom_style_match ----
    jaspar_text = ">MA0001.1 AGL3\nA [ 0 0 0 10 0 ]\nC [ 0 0 0 0 0 ]\nG [ 10 10 0 0 0 ]\nT [ 0 0 10 0 10 ]\n>MA0002.1 RUNX1\nA [ 5 5 5 5 5 ]\nC [ 5 5 5 5 5 ]\nG [ 5 5 5 5 5 ]\nT [ 5 5 5 5 5 ]\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "j.txt")
        with open(path, "w") as f: f.write(jaspar_text)
        entries = parse_jaspar_pfm_file(path)
        assert len(entries) == 2, f"got {len(entries)} entries"
        assert entries[0]["matrix_id"] == "MA0001.1" and entries[0]["W"] == 5
        # PWM column 0: A=C=T=0, G=10 → P(G) = (10 + 0.25) / (10 + 4*0.25)
        # = 10.25 / 11.0 ≈ 0.932
        assert entries[0]["pwm"][2, 0] > 0.9, entries[0]["pwm"][2, 0]
        print(f"parse_jaspar: {len(entries)} entries; first ID={entries[0]['matrix_id']}, W={entries[0]['W']}")

        # Build a cluster PWM identical to entry 0 → tomtom score should be high
        cluster_pwm = entries[0]["pwm"]
        cluster_ic = entries[0]["information_content"]
        m = tomtom_style_match(cluster_pwm, cluster_ic, entries[0], max_offset=2)
        assert m["score"] > 0.99, m
        print(f"tomtom self-match: score={m['score']:.4f}, offset={m['offset']}, overlap={m['overlap']}")

        # And against the flat (uniform) PWM, score should be low or NaN
        m2 = tomtom_style_match(cluster_pwm, cluster_ic, entries[1], max_offset=2)
        # Uniform PWM has std=0 per column → col_corrs all 0 → weighted average 0.
        # Strict assertion: m2 < m
        assert m2["score"] < m["score"]
        print(f"tomtom vs uniform: score={m2['score']:.4f} (< self-match)")

        # match_cluster_against_jaspar returns sorted top-N
        top = match_cluster_against_jaspar(cluster_pwm, cluster_ic, entries, top_n=2)
        assert top[0]["matrix_id"] == "MA0001.1"
        print(f"top match: {top[0]['matrix_id']} ({top[0]['name']}), score={top[0]['score']:.4f}")

    print("\nMOTIFS SMOKE PASSED")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
