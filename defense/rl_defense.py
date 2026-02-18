import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFENSE_NAMES = ['gaussian', 'jpeg', 'median', 'bitdepth', 'passthrough']
NUM_ACTIONS = len(DEFENSE_NAMES)


# ─────────────────────────────────────────────────────────────
# Helper: build a 2-D Gaussian kernel (cached per device)
# ─────────────────────────────────────────────────────────────
_gauss_cache: dict = {}


def _get_gaussian_kernel(kernel_size: int, sigma: float,
                         device: torch.device, dtype: torch.dtype):
    key = (kernel_size, sigma, device, dtype)
    if key not in _gauss_cache:
        ax = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        _gauss_cache[key] = k.view(1, 1, kernel_size, kernel_size).expand(3, 1, -1, -1).contiguous()
    return _gauss_cache[key]


# ─────────────────────────────────────────────────────────────
# Defense 0: Gaussian Smoothing  (GPU, batched)
# ─────────────────────────────────────────────────────────────
def apply_gaussian_smoothing(x: torch.Tensor,
                             kernel_size: int = 3,
                             sigma: float = 1.0) -> torch.Tensor:
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    kernel = _get_gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    out = F.conv2d(x, kernel, padding=kernel_size // 2, groups=3).clamp(0.0, 1.0)
    return out.squeeze(0) if single else out


# ─────────────────────────────────────────────────────────────
# Defense 1: JPEG Compression  (CPU, per-image)
# ─────────────────────────────────────────────────────────────
def apply_jpeg_compression(x: torch.Tensor,
                           quality: int = 75) -> torch.Tensor:
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    device = x.device
    results = []
    for i in range(x.shape[0]):
        img_np = (x[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        results.append(torch.from_numpy(arr).permute(2, 0, 1))
    out = torch.stack(results).to(device)
    return out.squeeze(0) if single else out


# ─────────────────────────────────────────────────────────────
# Defense 2: Median Filter  (GPU, batched via unfold)
# ─────────────────────────────────────────────────────────────
def apply_median_filter(x: torch.Tensor,
                        kernel_size: int = 3) -> torch.Tensor:
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    pad = kernel_size // 2
    x_padded = F.pad(x, [pad] * 4, mode='reflect')
    # Unfold into patches → (B, C, H, W, k*k)
    patches = x_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.contiguous().view(B, C, H, W, -1)
    out = patches.median(dim=-1).values.clamp(0.0, 1.0)
    return out.squeeze(0) if single else out


# ─────────────────────────────────────────────────────────────
# Defense 3: Bit-depth Reduction  (GPU, batched)
# ─────────────────────────────────────────────────────────────
def apply_bitdepth_reduction(x: torch.Tensor,
                             bits: int = 4) -> torch.Tensor:
    num_levels = 2 ** bits - 1
    return (torch.round(x * num_levels) / num_levels).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Defense 4: Pass-through
# ─────────────────────────────────────────────────────────────
def apply_passthrough(x: torch.Tensor) -> torch.Tensor:
    return x


# ─────────────────────────────────────────────────────────────
# Dispatch helpers
# ─────────────────────────────────────────────────────────────
_DEFENSE_FNS = [
    apply_gaussian_smoothing,   # 0
    apply_jpeg_compression,     # 1
    apply_median_filter,        # 2
    apply_bitdepth_reduction,   # 3
    apply_passthrough,          # 4
]


def apply_defense(x: torch.Tensor, action: int) -> torch.Tensor:
    """Apply the defense corresponding to *action* (0-4)."""
    return _DEFENSE_FNS[action](x)


def apply_defense_batch(x_batch: torch.Tensor, actions: list) -> torch.Tensor:
    """Apply per-image defenses efficiently by grouping same-action images.

    GPU-native defenses (gaussian, bitdepth, median, passthrough) are applied
    as a single batched call.  Only JPEG requires a per-image CPU loop.
    """
    B = x_batch.shape[0]
    device = x_batch.device
    out = torch.empty_like(x_batch)

    # Group indices by action
    groups: dict[int, list[int]] = {}
    for i, a in enumerate(actions):
        groups.setdefault(a, []).append(i)

    for action, indices in groups.items():
        idx = torch.tensor(indices, device=device, dtype=torch.long)
        sub = x_batch[idx]                          # (N, C, H, W)  on GPU
        defended = _DEFENSE_FNS[action](sub)        # batched call
        out[idx] = defended.to(device)

    return out
