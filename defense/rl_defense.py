"""defense/rl_defense.py
Defense action registry for the RL agent.

Changes vs v1:
  • Added randomized_smoothing and adaptive_jpeg defenses.
  • GPU-native Gaussian and median blur (no PIL).
  • apply_defense_batch groups by action for efficient batched dispatch.
  • Backward-compatible named batch functions retained for evaluate.py:
      apply_passthrough, apply_gaussian_smoothing, apply_jpeg_compression,
      apply_median_filter, apply_bitdepth_reduction
"""

import io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Defense registry ──────────────────────────────────────────────────────
DEFENSE_NAMES: List[str] = [
    "none",                 # 0
    "gaussian_blur",        # 1 – kernel 3
    "gaussian_blur_5",      # 2 – kernel 5 (stronger)
    "jpeg_75",              # 3 – JPEG quality 75
    "jpeg_50",              # 4 – JPEG quality 50
    "median_blur",          # 5 – 3×3 median
    "bit_depth_4",          # 6 – 4-bit quantisation
    "bit_depth_5",          # 7 – 5-bit quantisation
    "randomized_smoothing", # 8 – Gaussian noise σ=0.05
    "adaptive_jpeg",        # 9 – JPEG q60 + blur 3
]

NUM_ACTIONS: int = len(DEFENSE_NAMES)


# ─────────────────────────────────────────────────────────────
# Primitive implementations (operate on GPU tensors)
# ─────────────────────────────────────────────────────────────

def _gaussian_blur(x: torch.Tensor, k: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Depthwise Gaussian blur for (C,H,W) or (B,C,H,W) tensors."""
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kernel = g1d.outer(g1d).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    single = x.dim() == 3
    x4 = x.unsqueeze(0) if single else x
    C = x4.shape[1]
    kernel = kernel.expand(C, 1, k, k)
    out = F.conv2d(x4, kernel, padding=k // 2, groups=C)
    return out.squeeze(0) if single else out


def _median_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    """GPU 3×3 median filter for (C,H,W) or (B,C,H,W) tensors."""
    single = x.dim() == 3
    x4 = x.unsqueeze(0) if single else x
    B, C, H, W = x4.shape
    patches = F.unfold(x4, kernel_size=3, padding=1).view(B, C, 9, H * W)
    med = patches.median(dim=2).values.view(B, C, H, W)
    return med.squeeze(0) if single else med


def _bit_depth_reduction(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    levels = 2 ** bits - 1
    return (x * levels).round() / levels


def _randomized_smoothing(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def _jpeg_compress_single(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """JPEG round-trip for one (C,H,W) tensor in [0,1]."""
    arr = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf)).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).to(x.device)


def _jpeg_then_blur(x: torch.Tensor, quality: int = 60) -> torch.Tensor:
    return _gaussian_blur(_jpeg_compress_single(x, quality), k=3, sigma=1.0)


# ── Single-image dispatch table ───────────────────────────────────────────
_DEFENSE_FN = {
    0: lambda x: x,
    1: lambda x: _gaussian_blur(x, k=3, sigma=1.0),
    2: lambda x: _gaussian_blur(x, k=5, sigma=1.5),
    3: lambda x: _jpeg_compress_single(x, quality=75),
    4: lambda x: _jpeg_compress_single(x, quality=50),
    5: lambda x: _median_blur_3x3(x),
    6: lambda x: _bit_depth_reduction(x, bits=4),
    7: lambda x: _bit_depth_reduction(x, bits=5),
    8: lambda x: _randomized_smoothing(x, sigma=0.05),
    9: lambda x: _jpeg_then_blur(x, quality=60),
}

# Defenses that work natively on a whole (B,C,H,W) batch tensor
_BATCH_CAPABLE = {0, 1, 2, 5, 6, 7, 8}


# ─────────────────────────────────────────────────────────────
# Batched per-image dispatch (used by RL trainer)
# ─────────────────────────────────────────────────────────────

def apply_defense_batch(x_adv: torch.Tensor, actions: List[int]) -> torch.Tensor:
    """Apply per-image defenses to a batch efficiently.

    GPU-native defenses are grouped and applied as sub-batches.
    PIL-based defenses (JPEG variants) are parallelised via ThreadPoolExecutor.

    Args:
        x_adv:   (B, C, H, W) tensor in [0,1].
        actions: list of length B with integer action indices.
    Returns:
        (B, C, H, W) defended tensor on the same device.
    """
    device = x_adv.device
    out = torch.empty_like(x_adv)

    groups = defaultdict(list)
    for i, a in enumerate(actions):
        groups[a].append(i)

    for action, idxs in groups.items():
        idx_t = torch.tensor(idxs, device=device)
        subset = x_adv[idx_t]

        if action in _BATCH_CAPABLE:
            out[idx_t] = _DEFENSE_FN[action](subset).clamp(0.0, 1.0)
        else:
            fn = _DEFENSE_FN[action]
            with ThreadPoolExecutor(max_workers=min(len(idxs), 8)) as pool:
                futures = [pool.submit(fn, subset[j]) for j in range(len(idxs))]
                results = [f.result().clamp(0.0, 1.0) for f in futures]
            out[idx_t] = torch.stack(results, dim=0)

    return out


# ─────────────────────────────────────────────────────────────
# Named batch functions – backward-compatible API for evaluate.py
# Each accepts (B, C, H, W) and returns (B, C, H, W).
# ─────────────────────────────────────────────────────────────

def apply_passthrough(x: torch.Tensor) -> torch.Tensor:
    """No-op: return images unchanged."""
    return x.clamp(0.0, 1.0)


def apply_gaussian_smoothing(x: torch.Tensor,
                              k: int = 3,
                              sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur over a full (B,C,H,W) batch."""
    return _gaussian_blur(x, k=k, sigma=sigma).clamp(0.0, 1.0)


def apply_jpeg_compression(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """JPEG compression over a full (B,C,H,W) batch (parallelised)."""
    B = x.shape[0]
    with ThreadPoolExecutor(max_workers=min(B, 8)) as pool:
        futures = [pool.submit(_jpeg_compress_single, x[i], quality)
                   for i in range(B)]
        results = [f.result().clamp(0.0, 1.0) for f in futures]
    return torch.stack(results, dim=0)


def apply_median_filter(x: torch.Tensor) -> torch.Tensor:
    """3×3 median filter over a full (B,C,H,W) batch."""
    return _median_blur_3x3(x).clamp(0.0, 1.0)


def apply_bitdepth_reduction(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Bit-depth reduction over a full (B,C,H,W) batch."""
    return _bit_depth_reduction(x, bits=bits).clamp(0.0, 1.0)