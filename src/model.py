"""K562 enhancer prediction model: Puffin-D backbone + per-base enhancer head.

The backbone is structurally identical to Dudnyk, Shi & Zhou 2024's
Puffin-D (double U-Net CNN, ~19.5M params) so ``puffin_D.pth`` loads
with ``strict=False`` populating every backbone parameter exactly. The
output head is replaced by ``Conv1d(64, 1, 1)`` returning per-base logits.

Input  : ``(B, 4, 100_000)`` one-hot DNA, float32
Output : ``(B, 1, 100_000)`` logits — apply sigmoid externally if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


# ConvBlock — copied verbatim from puffin_D.py.
# Do NOT modify; signature must stay identical for state-dict compatibility.
class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


@dataclass
class EnhancerModelConfig:
    """Hyperparameters for ``EnhancerModel``."""

    seq_len: int = 100_000
    pretrained_path: Optional[str] = None
    n_outputs: int = 1


class EnhancerModel(nn.Module):
    """Puffin-D double U-Net with a single per-base enhancer head."""

    def __init__(self, cfg: Optional[EnhancerModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = EnhancerModelConfig()
        self.cfg = cfg

        # ---- First U-Net: encoder (uplblocks + upblocks) -------------------
        # Strided downsampling: 100000 -> 25 over 7 layers (1, 4, 4, 5, 5, 5, 2).
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
                nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            ]
        )

        # ---- First U-Net: decoder (downlblocks + downblocks) ---------------
        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
                nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
            ]
        )

        # ---- Second U-Net: encoder (uplblocks2 + upblocks2) ----------------
        # Six layers (no leading 1-stride layer, since channels already 64).
        self.uplblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks2 = nn.ModuleList(
            [
                nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            ]
        )

        # ---- Second U-Net: decoder (downlblocks2 + downblocks2) ------------
        self.downlblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks2 = nn.ModuleList(
            [
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
                nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
                nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
            ]
        )

        # Output heads. final_norm matches Puffin-D's pre-output projection;
        # enhancer_head returns logits — sigmoid applied externally.
        self.final_norm = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.enhancer_head = nn.Conv1d(64, cfg.n_outputs, kernel_size=1)

        if cfg.pretrained_path is not None:
            load_puffin_pretrained(self, cfg.pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, L) one-hot → (B, 1, L) per-base enhancer logits."""
        out = x
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final_norm(out)
        logits = self.enhancer_head(out)
        return logits


def load_puffin_pretrained(
    model: EnhancerModel,
    ckpt_path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[EnhancerModel, dict]:
    """Load Puffin-D pretrained weights into an EnhancerModel in place.

    Uses ``strict=False`` so Puffin-D's original output head is dropped
    and our ``enhancer_head`` stays at fresh init. Returns ``(model, info)``
    where info reports loaded/skipped/fresh key counts for diagnostics.
    Accepts a raw state_dict or ``{"model_state_dict": ...}`` /
    ``{"model": ...}`` wrappers.
    """
    # weights_only=False: puffin_D.pth predates the PyTorch 2.6 strict default.
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())

    loaded_keys = sorted(model_keys & ckpt_keys)
    skipped_in_ckpt = sorted(ckpt_keys - model_keys)  # Puffin-D's original head
    fresh_in_model = sorted(model_keys - ckpt_keys)   # our enhancer_head

    missing, unexpected = model.load_state_dict(state, strict=False)

    # BN num_batches_tracked is treated specially by load_state_dict (silent
    # default if missing), so filter from both sides before parity asserts.
    def _is_bn_counter(key: str) -> bool:
        return key.endswith(".num_batches_tracked")

    missing_real = {k for k in missing if not _is_bn_counter(k)}
    fresh_real   = {k for k in fresh_in_model if not _is_bn_counter(k)}
    unexpected_real = {k for k in unexpected if not _is_bn_counter(k)}
    skipped_real    = {k for k in skipped_in_ckpt if not _is_bn_counter(k)}

    assert missing_real == fresh_real, (
        f"load_state_dict missing {sorted(missing_real)} differs from "
        f"fresh_in_model {sorted(fresh_real)}"
    )
    assert unexpected_real == skipped_real, (
        f"load_state_dict unexpected {sorted(unexpected_real)} differs "
        f"from skipped_in_ckpt {sorted(skipped_real)}"
    )

    bn_counters_skipped = sorted({k for k in fresh_in_model if _is_bn_counter(k)})

    info = {
        "ckpt_path": str(ckpt_path),
        "n_loaded": len(loaded_keys),
        "n_skipped_in_ckpt": len(skipped_in_ckpt),
        "n_fresh_in_model": len(fresh_in_model),
        "n_bn_counters_skipped": len(bn_counters_skipped),
        "loaded_keys": loaded_keys,
        "skipped_in_ckpt": skipped_in_ckpt,
        "fresh_in_model": fresh_in_model,
        "bn_counters_skipped": bn_counters_skipped,
        "total_model_keys": len(model_keys),
        "total_ckpt_keys": len(ckpt_keys),
    }

    print(
        f"[load_puffin_pretrained] {ckpt_path}\n"
        f"  loaded            : {info['n_loaded']:>5d} keys "
        f"(populated from checkpoint)\n"
        f"  skipped_in_ckpt   : {info['n_skipped_in_ckpt']:>5d} keys "
        f"(Puffin-D's original output head — discarded)\n"
        f"  fresh_in_model    : {info['n_fresh_in_model']:>5d} keys "
        f"(our enhancer_head — stays at fresh init)\n"
        f"  bn_counters       : {info['n_bn_counters_skipped']:>5d} keys "
        f"(benign — BatchNorm num_batches_tracked, init to 0)\n"
        f"  total_model_keys  : {info['total_model_keys']:>5d}\n"
        f"  total_ckpt_keys   : {info['total_ckpt_keys']:>5d}",
        flush=True,
    )
    if info["n_fresh_in_model"] > 0 and info["n_fresh_in_model"] <= 8:
        for k in info["fresh_in_model"]:
            print(f"    fresh: {k}", flush=True)
    if info["n_skipped_in_ckpt"] > 0 and info["n_skipped_in_ckpt"] <= 8:
        for k in info["skipped_in_ckpt"]:
            print(f"    skipped: {k}", flush=True)

    return model, info


if __name__ == "__main__":
    cfg = EnhancerModelConfig()
    model = EnhancerModel(cfg)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    x = torch.randn(1, 4, cfg.seq_len)
    with torch.no_grad():
        logits = model(x)
    print(f"logits.shape: {tuple(logits.shape)}")
    assert logits.shape == (1, cfg.n_outputs, cfg.seq_len)
    assert torch.isfinite(logits).all()
    print("EnhancerModel standalone smoke passed.")
