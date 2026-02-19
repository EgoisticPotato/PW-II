"""defense/entropy.py
Compute per-image state features for the RL defense agent.

Optimizations over v1:
  • f1 (pixel entropy) fully vectorised on GPU – no Python loop over batch.
  • f4 gradient magnitude uses a single batched backward pass (unchanged).
  • f6/f7 (global mean/std) replaced with attack-discriminative DCT-based
    high-to-low frequency energy ratio and L∞ of Laplacian response.
  • All sigmoid scaling constants exposed as module-level tunables.
  • STATE_DIM bumped to 12 (two new features added).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Public constant consumed by DQNAgent and RLDefenseTrainer
STATE_DIM = 12

# ── Sigmoid scaling constants (tune here, not buried in formulas) ──────────
_S_GRAD   = 200.0   # f4  gradient magnitude
_B_GRAD   = 0.01
_S_LAP    = 50.0    # f5  Laplacian noise
_B_LAP    = 0.05
_S_SOBEL  = 20.0    # f8  Sobel edge energy
_B_SOBEL  = 0.05
_S_VAR    = 500.0   # f10 local spatial variance
_B_VAR    = 0.005
_S_LINF   = 100.0   # f11 L∞ Laplacian
_B_LINF   = 0.05
_S_DCT    = 10.0    # f12 DCT HF/LF ratio
_B_DCT    = 0.5


def _sigmoid_scale(x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return torch.sigmoid(scale * (x - bias))


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def compute_state_features(image: torch.Tensor,
                            classifier: nn.Module,
                            device: torch.device) -> torch.Tensor:
    """Single-image wrapper – delegates to the batched implementation."""
    return compute_state_features_batch(
        image.unsqueeze(0), classifier, device
    ).squeeze(0)


def compute_state_features_batch(images: torch.Tensor,
                                  classifier: nn.Module,
                                  device: torch.device) -> torch.Tensor:
    """Compute (B, STATE_DIM=12) feature matrix.

    One forward + one backward pass for the entire batch.
    All spatial convolutions are batched on GPU.

    Features
    --------
    f01  pixel entropy (vectorised histogram)
    f02  max softmax prob (classifier confidence)
    f03  prediction entropy (normalised)
    f04  gradient magnitude (sigmoid scaled)
    f05  Laplacian noise estimate
    f06  high-frequency Sobel energy
    f07  local spatial variance
    f08  top-2 softmax margin
    f09  top-3 cumulative probability
    f10  global pixel std (kept – still useful as a normalisation signal)
    f11  L∞ of Laplacian response  ← NEW (discriminates attack strength)
    f12  DCT high/low energy ratio  ← NEW (freq-domain attack signature)
    """
    images = images.to(device).float()
    B = images.shape[0]

    # ── Grayscale (shared across spatial features) ────────────────────────
    gray_b = images.mean(dim=1)          # (B, H, W)
    gray_4d = gray_b.unsqueeze(1)        # (B, 1, H, W)

    # ── f01: pixel entropy – fully vectorised ────────────────────────────
    # Approximate per-image histogram via 256-bin soft assignment on GPU.
    gray_flat = gray_b.view(B, -1)                       # (B, N)
    bins = torch.linspace(0.0, 1.0, 257, device=device)  # 257 edges → 256 bins
    centers = (bins[:-1] + bins[1:]) / 2                 # (256,)
    # Nearest-bin assignment via clamp + floor
    idx = (gray_flat.unsqueeze(2) - centers.view(1, 1, 256)).abs().argmin(dim=2)
    hist = torch.zeros(B, 256, device=device, dtype=images.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=images.dtype))
    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-10)
    mask = hist > 0
    log_hist = torch.where(mask, torch.log2(hist.clamp(min=1e-10)),
                            torch.zeros_like(hist))
    f01 = -(hist * log_hist).sum(dim=1) / 8.0             # normalise to [0,1]

    # ── f02, f03, f08, f09: softmax features – one forward pass ──────────
    with torch.no_grad():
        logits = classifier(images)
        probs  = F.softmax(logits, dim=1)                 # (B, C)

    f02 = probs.max(dim=1).values

    pred_ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    f03 = pred_ent / math.log(probs.shape[1])

    sorted_p = probs.sort(dim=1, descending=True).values
    f08 = sorted_p[:, 0] - sorted_p[:, 1]                # top-2 margin
    f09 = sorted_p[:, :3].sum(dim=1)                     # top-3 cumulative

    # ── f04: gradient magnitude – one batched backward pass ──────────────
    with torch.enable_grad():
        imgs_g = images.detach().clone().requires_grad_(True)
        logits_g = classifier(imgs_g)
        preds_g  = logits_g.argmax(dim=1)
        loss_g   = F.cross_entropy(logits_g, preds_g, reduction='sum')
        loss_g.backward()
    raw_grad = imgs_g.grad.abs().mean(dim=(1, 2, 3))
    f04 = _sigmoid_scale(raw_grad, _S_GRAD, _B_GRAD)

    # ── Shared conv kernels ───────────────────────────────────────────────
    dtype = images.dtype

    lap_k = torch.tensor([[0., 1., 0.],
                           [1.,-4., 1.],
                           [0., 1., 0.]], device=device, dtype=dtype
                          ).view(1, 1, 3, 3)

    sx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], device=device, dtype=dtype
                       ).view(1, 1, 3, 3)

    sy = torch.tensor([[-1.,-2.,-1.],
                        [ 0., 0., 0.],
                        [ 1., 2., 1.]], device=device, dtype=dtype
                       ).view(1, 1, 3, 3)

    with torch.no_grad():
        # ── f05: Laplacian noise estimate ──
        lap    = F.conv2d(gray_4d, lap_k, padding=1)
        f05    = _sigmoid_scale(lap.abs().mean(dim=(1, 2, 3)), _S_LAP, _B_LAP)

        # ── f06: Sobel edge energy ──
        gx  = F.conv2d(gray_4d, sx, padding=1)
        gy  = F.conv2d(gray_4d, sy, padding=1)
        f06 = _sigmoid_scale(
            (gx**2 + gy**2).mean(dim=(1, 2, 3)), _S_SOBEL, _B_SOBEL
        )

        # ── f07: local spatial variance ──
        ps    = 7
        ok    = torch.ones(1, 1, ps, ps, device=device, dtype=dtype) / ps**2
        lm    = F.conv2d(gray_4d,      ok, padding=ps // 2)
        lsm   = F.conv2d(gray_4d**2,   ok, padding=ps // 2)
        lv    = (lsm - lm**2).clamp(min=0)
        f07   = _sigmoid_scale(lv.mean(dim=(1, 2, 3)), _S_VAR, _B_VAR)

        # ── f10: global pixel std ──
        flat  = images.view(B, -1)
        f10   = (flat.std(dim=1) / 0.3).clamp(0.0, 1.0)

        # ── f11: L∞ of Laplacian (attack-strength discriminator) ── NEW
        lap_linf = lap.abs().amax(dim=(1, 2, 3))          # (B,)
        f11 = _sigmoid_scale(lap_linf, _S_LINF, _B_LINF)

        # ── f12: DCT high/low frequency energy ratio ── NEW
        # Cheap approximation via difference-of-means on downsampled patches.
        # Low-freq proxy  = mean of 8×8 avg-pool blocks
        # High-freq proxy = mean of |image - low_freq_proxy|
        lf_proxy  = F.avg_pool2d(gray_4d, kernel_size=8, stride=8)
        lf_up     = F.interpolate(lf_proxy, size=gray_4d.shape[-2:],
                                  mode='nearest')
        hf_energy = (gray_4d - lf_up).abs().mean(dim=(1, 2, 3))
        lf_energy = lf_proxy.abs().mean(dim=(1, 2, 3))
        ratio     = hf_energy / (lf_energy + 1e-8)
        f12 = _sigmoid_scale(ratio, _S_DCT, _B_DCT)

    # ── Stack → (B, STATE_DIM) ────────────────────────────────────────────
    features = torch.stack(
        [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12],
        dim=1
    )
    return features.float().cpu()