"""Data pipeline for K562 enhancer prediction: genome adapter, label reader, samplers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class PackbitsGenome:
    """selene_mini.Genome adapter returning (4, L) float32 with N=0.25."""

    def __init__(self, *args, **kwargs):
        # Deferred import so smoke tests without selene_mini still load.
        from selene_mini.datasets import Genome
        self._genome = Genome(*args, **kwargs)

    def get(self, chrom, start, end, strand="+"):
        enc = self._genome.get(chrom, start, end, strand=strand).astype(np.float32)
        n_rows = enc.sum(axis=1) == 0  # rows with no 1-hot set → "N"
        enc[n_rows] = 0.25
        return enc.T  # (4, L)

    def get_chrs(self):
        return self._genome.get_chrs()

    def get_chr_lens(self):
        return self._genome.get_chr_lens()

    def uninitialize(self):
        self._genome.uninitialize()


class EnhancerLabelDataset:
    """Memmap reader for ``{labels_dir}/{chrom}.npy``, shape (L, 2) uint8.

    Channel 0 is enhancer_label, channel 1 is callable_mask. ``.get()``
    returns channel-first (2, end-start) uint8 to match selene_mini.
    """

    def __init__(self, labels_dir, chroms: Iterable[str]):
        self.labels_dir = Path(labels_dir)
        self._arrays = {}
        for chrom in chroms:
            path = self.labels_dir / f"{chrom}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing label array: {path}")
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.dtype != np.uint8:
                raise ValueError(
                    f"Unexpected label array shape/dtype at {path}: "
                    f"got shape={arr.shape}, dtype={arr.dtype}; "
                    f"expected (L, 2) uint8"
                )
            self._arrays[chrom] = arr

    def get(self, chrom: str, start: int, end: int, strand: str = "+"):
        arr = self._arrays[chrom]
        sl = arr[start:end]               # (end-start, 2)
        return np.ascontiguousarray(sl.T)  # (2, end-start)

    def uninitialize(self):
        # selene_mini calls this during pickling so the method must exist.
        pass

    def get_chrs(self):
        return list(self._arrays.keys())


def load_callable_mask_chr_weights(
    labels_dir,
    chrom_lens: list[tuple[str, int]],
) -> list[np.ndarray]:
    """DEPRECATED — per-chrom weight loader for WeightedRandomPositions.

    Replaced with uniform RandomPositions + callable_fraction_filter
    because the weighted sampler's internal float64 cumsum over the full
    genome OOMed (~107 GB peak vs 64 GB allocation).
    """
    import sys

    labels_dir = Path(labels_dir)
    weights: list[np.ndarray] = []
    for chrom, chrom_len in chrom_lens:
        path = labels_dir / f"{chrom}.npy"
        if not path.exists():
            weights.append(np.zeros(chrom_len, dtype=np.float32))
            continue
        arr = np.load(path, mmap_mode="r")
        mask = arr[:, 1].astype(np.float32, copy=True)
        if mask.shape[0] == chrom_len:
            weights.append(mask)
        elif mask.shape[0] < chrom_len:
            print(
                f"WARN [{chrom}] mask len {mask.shape[0]} < chrom_len "
                f"{chrom_len}; zero-padding tail",
                file=sys.stderr,
            )
            padded = np.zeros(chrom_len, dtype=np.float32)
            padded[: mask.shape[0]] = mask
            weights.append(padded)
        else:
            print(
                f"WARN [{chrom}] mask len {mask.shape[0]} > chrom_len "
                f"{chrom_len}; truncating",
                file=sys.stderr,
            )
            weights.append(mask[:chrom_len].copy())
    return weights


def callable_fraction_filter(target_slice, min_fraction: float = 0.5):
    """Return True (reject) if callable_mask fraction < ``min_fraction``.

    Attached at index 1 of RandomPositionsSampler.data_filters (label
    dataset). Without this filter, ~50 % of uniformly-sampled windows
    land on mostly-masked centers and carry no training signal.
    """
    # SamplerDataLoader prepends a batch dim before invoking filters, so
    # the input can be (2, L) or (1, 2, L).
    if target_slice.ndim == 3:
        target_slice = target_slice[0]
    return float(target_slice[1].mean()) < min_fraction


@dataclass
class SamplerConfig:
    """Configuration for ``build_train_sampler`` / ``build_val_sampler``."""

    genome_path: str
    labels_dir: str
    training_chroms: tuple[str, ...]
    validation_holdout: tuple[str, ...] = ("chr21",)
    test_holdout: tuple[str, ...] = ("chr22",)
    sample_length: int = 100_000
    seed: int = 436
    blacklist_chroms: tuple[str, ...] = ("chrY", "chrM")
    # 0 disables the callable_fraction filter; 0.5 keeps signal density up.
    min_callable_fraction: float = 0.5


def build_train_sampler(cfg: SamplerConfig):
    """Build a training RandomPositionsSampler (uniform + callable filter)."""
    from functools import partial

    from selene_mini.positions import RandomPositions
    from selene_mini.samplers import RandomPositionsSampler

    genome = PackbitsGenome(input_path=cfg.genome_path, storage="Packbits")
    label_ds = EnhancerLabelDataset(cfg.labels_dir, list(cfg.training_chroms))

    # Explicitly blacklist every genome chrom not in training_chroms so
    # chrX, chr8/9/10, and alt contigs are unambiguously excluded.
    train_set = set(cfg.training_chroms)
    explicit_blacklist = sorted({
        c for c, _clen in genome.get_chr_lens()
        if c not in train_set
    })
    for c in cfg.blacklist_chroms:
        if c not in explicit_blacklist:
            explicit_blacklist.append(c)

    pos = RandomPositions(
        genome,
        sample_length=cfg.sample_length,
        validation_holdout=list(cfg.validation_holdout),
        test_holdout=list(cfg.test_holdout),
        blacklist_chroms=explicit_blacklist,
        mode="train",
    )

    data_filters = None
    if cfg.min_callable_fraction > 0:
        data_filters = [
            None,
            partial(callable_fraction_filter,
                    min_fraction=cfg.min_callable_fraction),
        ]

    sampler = RandomPositionsSampler(
        datasets=[genome, label_ds],
        position_sampler=pos,
        data_filters=data_filters,
        seed=cfg.seed,
    )
    return sampler


def build_val_sampler(cfg: SamplerConfig):
    """Build a uniform validation sampler over the validation chromosome."""
    from functools import partial

    from selene_mini.positions import RandomPositions
    from selene_mini.samplers import RandomPositionsSampler

    genome = PackbitsGenome(input_path=cfg.genome_path, storage="Packbits")
    label_ds = EnhancerLabelDataset(
        cfg.labels_dir, list(cfg.validation_holdout)
    )

    pos = RandomPositions(
        genome,
        sample_length=cfg.sample_length,
        validation_holdout=list(cfg.validation_holdout),
        test_holdout=list(cfg.test_holdout),
        blacklist_chroms=list(cfg.blacklist_chroms),
        mode="validate",
    )

    data_filters = None
    if cfg.min_callable_fraction > 0:
        data_filters = [
            None,
            partial(callable_fraction_filter,
                    min_fraction=cfg.min_callable_fraction),
        ]

    sampler = RandomPositionsSampler(
        datasets=[genome, label_ds],
        position_sampler=pos,
        data_filters=data_filters,
        seed=cfg.seed + 1,
    )
    return sampler


def iter_chrom_windows(
    chrom_length: int,
    window: int = 100_000,
    stride: int = 50_000,
) -> Iterator[tuple[int, int]]:
    """Yield (start, end) tile positions covering the chromosome.

    The final window may extend past the chrom end — caller pads with
    N=0.25. Requires window > stride and (window - stride) even so the
    overlap band is symmetric.
    """
    if chrom_length <= 0:
        raise ValueError("chrom_length must be positive")
    if stride <= 0 or window <= 0:
        raise ValueError("window and stride must be positive")
    if window <= stride:
        raise ValueError("window must be > stride")
    if (window - stride) % 2 != 0:
        raise ValueError("window - stride must be even")

    if chrom_length <= stride:
        yield 0, window
        return

    last_start = ((chrom_length - stride) // stride) * stride
    for start in range(0, last_start + 1, stride):
        yield start, start + window


class DeterministicWindowDataset:
    """Single-chrom fixed enumeration of stride windows for eval.

    Out-of-bounds tail is padded N=0.25 for sequence, 0 for labels.
    """

    def __init__(
        self,
        genome: PackbitsGenome,
        label_ds: EnhancerLabelDataset,
        chrom: str,
        chrom_length: int,
        window: int = 100_000,
        stride: int = 50_000,
    ):
        self.genome = genome
        self.label_ds = label_ds
        self.chrom = chrom
        self.chrom_length = chrom_length
        self.window = window
        self.stride = stride
        self._tiles = list(iter_chrom_windows(chrom_length, window, stride))

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self._tiles[idx]
        # PackbitsGenome pads N inside the chrom but not past chrom_length;
        # fetch clamped, then pad with 0.25 N for trailing positions.
        clip_end = min(end, self.chrom_length)
        seq = np.full((4, end - start), 0.25, dtype=np.float32)
        if start < clip_end:
            chunk = self.genome.get(self.chrom, start, clip_end)  # (4, n_real)
            seq[:, : clip_end - start] = chunk
        target = np.zeros((2, end - start), dtype=np.uint8)
        if start < clip_end:
            lbl = self.label_ds.get(self.chrom, start, clip_end)  # (2, n_real)
            target[:, : clip_end - start] = lbl
        return seq, target

    @property
    def tiles(self) -> list[tuple[int, int]]:
        return list(self._tiles)


__all__ = [
    "PackbitsGenome",
    "EnhancerLabelDataset",
    "SamplerConfig",
    "build_train_sampler",
    "build_val_sampler",
    "load_callable_mask_chr_weights",
    "callable_fraction_filter",
    "iter_chrom_windows",
    "DeterministicWindowDataset",
]
