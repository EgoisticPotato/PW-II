import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 10


# ─────────────────────────────────────────────────────────────
# Public API  (single-image kept for backward compat)
# ─────────────────────────────────────────────────────────────
def compute_state_features(image: torch.Tensor,
                           classifier: nn.Module,
                           device: torch.device) -> torch.Tensor:
    """Compute STATE_DIM features for one (C,H,W) image.  Delegates to batch."""
    return compute_state_features_batch(
        image.unsqueeze(0), classifier, device
    ).squeeze(0)


def compute_state_features_batch(images: torch.Tensor,
                                  classifier: nn.Module,
                                  device: torch.device) -> torch.Tensor:
    """Compute (B, STATE_DIM) feature matrix with a single forward+backward pass.

    All B images are processed together on GPU:
      • 1 batched forward  pass  → softmax features (f2, f3, f9)
      • 1 batched backward pass  → gradient magnitude (f4)
      • fully vectorised convolutions → pixel / frequency features (f1,f5-f8,f10)

    Args:
        images:     (B, C, H, W) tensor in [0, 1], on any device.
        classifier: frozen model in eval mode.
        device:     target device.
    Returns:
        (B, STATE_DIM) float32 tensor on CPU.
    """
    images = images.to(device)
    B = images.shape[0]

    # ── f1: pixel entropy (per-image histogram – cheapest loop remaining) ──
    gray_b = images.mean(dim=1)           # (B, H, W)  grayscale
    f1_list = []
    for i in range(B):
        hist = torch.histc(gray_b[i], bins=256, min=0.0, max=1.0)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        ent = -(hist * torch.log2(hist)).sum().item()
        f1_list.append(ent / 8.0)         # normalise to [0,1]
    f1 = torch.tensor(f1_list)            # (B,)

    # ── f2, f3, f9: softmax features – ONE forward pass for all B images ──
    with torch.no_grad():
        logits = classifier(images)       # (B, num_classes)
        probs  = F.softmax(logits, dim=1) # (B, num_classes)

    f2 = probs.max(dim=1).values.cpu()    # max confidence

    pred_ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    f3 = (pred_ent / math.log(probs.shape[1])).cpu()   # normalised entropy

    sorted_probs = probs.sort(dim=1, descending=True).values
    f9 = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu()  # top-2 gap

    # ── f4: gradient magnitude – ONE batched backward pass ──
    # Because each loss_i depends ONLY on images[i], reduction='sum' backprop
    # yields true per-sample gradients: images_g.grad[i] = d(loss_i)/d(images[i])
    with torch.enable_grad():
        images_g = images.detach().clone().requires_grad_(True)
        logits_g  = classifier(images_g)
        preds_g   = logits_g.argmax(dim=1)
        loss_g    = F.cross_entropy(logits_g, preds_g, reduction='sum')
        loss_g.backward()

    raw_grad = images_g.grad.abs().mean(dim=(1, 2, 3)).cpu()  # (B,)
    f4 = torch.tensor([
        1.0 / (1.0 + math.exp(-200.0 * (v.item() - 0.01)))
        for v in raw_grad
    ])

    # ── Shared grayscale batch for conv features ──────────────────────────
    gray_4d = gray_b.unsqueeze(1)         # (B, 1, H, W)

    with torch.no_grad():
        # ── f5: Laplacian noise estimate ──
        lap_k = torch.tensor([[0., 1., 0.],
                               [1.,-4., 1.],
                               [0., 1., 0.]],
                              device=device, dtype=images.dtype)
        lap_k = lap_k.view(1, 1, 3, 3)
        lap   = F.conv2d(gray_4d, lap_k, padding=1)
        raw_f5 = lap.abs().mean(dim=(1, 2, 3)).cpu()
        f5 = torch.tensor([
            1.0 / (1.0 + math.exp(-50.0 * (v.item() - 0.05)))
            for v in raw_f5
        ])

        # ── f8: high-frequency Sobel energy ──
        sx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]],
                           device=device, dtype=images.dtype).view(1, 1, 3, 3)
        sy = torch.tensor([[-1.,-2.,-1.],
                            [ 0., 0., 0.],
                            [ 1., 2., 1.]],
                           device=device, dtype=images.dtype).view(1, 1, 3, 3)
        gx = F.conv2d(gray_4d, sx, padding=1)
        gy = F.conv2d(gray_4d, sy, padding=1)
        raw_f8 = (gx ** 2 + gy ** 2).mean(dim=(1, 2, 3)).cpu()
        f8 = torch.tensor([
            1.0 / (1.0 + math.exp(-20.0 * (v.item() - 0.05)))
            for v in raw_f8
        ])

        # ── f10: local spatial variance ──
        ps = 7
        ones_k = torch.ones(1, 1, ps, ps, device=device, dtype=images.dtype) / ps**2
        lm  = F.conv2d(gray_4d,        ones_k, padding=ps // 2)
        lsm = F.conv2d(gray_4d ** 2,   ones_k, padding=ps // 2)
        lv  = (lsm - lm ** 2).clamp(min=0)
        raw_f10 = lv.mean(dim=(1, 2, 3)).cpu()
        f10 = torch.tensor([
            1.0 / (1.0 + math.exp(-500.0 * (v.item() - 0.005)))
            for v in raw_f10
        ])

    # ── f6, f7: global mean & std ──
    flat = images.view(B, -1).cpu()
    f6   = flat.mean(dim=1)
    f7   = (flat.std(dim=1) / 0.3).clamp(0.0, 1.0)

    # ── Stack → (B, STATE_DIM) ──
    features = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dim=1)
    return features.float()
