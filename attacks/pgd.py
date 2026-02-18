#!/usr/bin/env python3
"""
Madry-style PGD adversarial training script using torchattacks.PGD (15 steps)
with EfficientNetB0_GCBAM architecture (Ghost + CBAM).

Modified to load the Fetal Planes dataset from HuggingFace Hub:
    Dataset: qingyuyang/Fetal_Planes_DB
    Classes: Abdomen(0), Brain(1), Femur(2), Thorax(3), Cervix(4), Other(5)

Install dependencies:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install torchattacks datasets huggingface_hub Pillow tqdm numpy
    pip install scikit-learn tensorboard
    pip install opencv-python  (only needed if you switch back to local files)

HuggingFace login (if dataset requires authentication):
    huggingface-cli login
  OR set environment variable:
    export HF_TOKEN="hf_your_token_here"
"""

import json
import os
import shutil
import time
from datetime import datetime
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
from datasets import load_dataset

from torchattacks import PGD
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
out_dir     = "model_out_gcbam_pgd15"
config_file = "config_pgd15_gcbam.json"

epochs        = 30
batch_size    = 8
lr            = 1e-4
num_workers   = 0
log_interval_steps        = 100
summary_interval_steps    = 100
checkpoint_interval_steps = 1000

# PGD parameters (15 steps)
pgd_eps          = 8 / 255
pgd_alpha        = 2 / 255
pgd_steps        = 15
pgd_random_start = True

# Dataset split ratios (HF dataset has only one split; we divide manually)
train_ratio = 0.70
val_ratio   = 0.15
# remaining 0.15 → test (not used during training)

# Mixed precision
use_amp = True

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────
# Optional JSON config override
# ─────────────────────────────────────────────────────────────
if os.path.exists(config_file):
    try:
        with open(config_file) as f:
            cfg = json.load(f)
        batch_size       = cfg.get("training_batch_size", batch_size)
        epochs           = cfg.get("num_epochs", epochs)
        lr               = cfg.get("learning_rate", lr)
        pgd_eps          = cfg.get("epsilon", pgd_eps)
        pgd_steps        = cfg.get("k", pgd_steps)
        pgd_alpha        = cfg.get("a", pgd_alpha)
        pgd_random_start = cfg.get("random_start", pgd_random_start)
        val_ratio        = cfg.get("val_fraction", val_ratio)
        print(f"Config loaded from {config_file}")
    except Exception as e:
        print(f"Warning: could not parse {config_file}: {e}")

os.makedirs(out_dir, exist_ok=True)
if os.path.exists(config_file):
    try:
        shutil.copy(config_file, out_dir)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─────────────────────────────────────────────────────────────
# Model: EfficientNetB0 + Ghost Module + CBAM
# ─────────────────────────────────────────────────────────────
class GhostModule(nn.Module):
    def __init__(self, in_channels, ratio=2):
        super().__init__()
        self.out_channels = in_channels // ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
                      padding=1, groups=self.out_channels, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return (avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetB0_GCBAM(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.features    = efficientnet_b0(weights=weights).features
        backbone_out_ch  = 1280
        self.ghost       = GhostModule(backbone_out_ch, ratio=2)
        self.cbam        = CBAM(backbone_out_ch, reduction_ratio=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Linear(backbone_out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.ghost(x)
        x = self.cbam(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# HuggingFace Fetal Planes Dataset Wrapper
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Abdomen', 'Brain', 'Femur', 'Thorax', 'Cervix', 'Other']


class HFFetalPlanesDataset(Dataset):
    """
    Wraps a HuggingFace split of qingyuyang/Fetal_Planes_DB.
    Pipeline: PIL Image → grayscale → RGB → resize 224×224 → [0,1] tensor (C,H,W)
    This matches the original FetalPlanesDataset preprocessing exactly.
    """

    def __init__(self, hf_split, img_size=224):
        self.hf_split = hf_split
        self.img_size = img_size

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        sample = self.hf_split[idx]

        pil_img = sample['image']                              # PIL Image
        gray    = pil_img.convert('L')                        # Grayscale
        rgb     = gray.convert('RGB')                         # Back to 3-channel
        rgb     = rgb.resize((self.img_size, self.img_size),
                              Image.BILINEAR)

        img_np = np.array(rgb, dtype=np.float32) / 255.0      # H×W×3  [0,1]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # 3×H×W

        label  = int(sample['label'])
        return tensor, label


# ─────────────────────────────────────────────────────────────
# Load & Split Dataset from HuggingFace Hub
# ─────────────────────────────────────────────────────────────
def load_fetal_planes_hf(train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Loads qingyuyang/Fetal_Planes_DB from HuggingFace Hub and
    splits it into train / val / test subsets.

    Returns:
        train_dataset, val_dataset, test_dataset  (HFFetalPlanesDataset)
        num_classes (int)
    """
    print("Loading Fetal Planes dataset from HuggingFace Hub ...")
    print("  Dataset : qingyuyang/Fetal_Planes_DB")
    print(f"  Classes : {CLASS_NAMES}\n")

    hf_dataset = load_dataset("qingyuyang/Fetal_Planes_DB", split="train")
    hf_dataset = hf_dataset.shuffle(seed=seed)

    total     = len(hf_dataset)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    hf_train = hf_dataset.select(range(0, train_end))
    hf_val   = hf_dataset.select(range(train_end, val_end))
    hf_test  = hf_dataset.select(range(val_end, total))

    print(f"  Total   : {total}")
    print(f"  Train   : {len(hf_train)}")
    print(f"  Val     : {len(hf_val)}")
    print(f"  Test    : {len(hf_test)}\n")

    return (
        HFFetalPlanesDataset(hf_train),
        HFFetalPlanesDataset(hf_val),
        HFFetalPlanesDataset(hf_test),
        len(CLASS_NAMES)
    )


# ─────────────────────────────────────────────────────────────
# Load datasets and DataLoaders
# ─────────────────────────────────────────────────────────────
train_dataset, val_dataset, test_dataset, num_classes = load_fetal_planes_hf(
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    seed=seed
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

print(f"Train batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")
print(f"Test  batches : {len(test_loader)}\n")

# ─────────────────────────────────────────────────────────────
# Model, criterion, optimizer, AMP scaler
# ─────────────────────────────────────────────────────────────
model = EfficientNetB0_GCBAM(num_classes=num_classes, pretrained=False).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
print("Model placed on device.\n")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

# ─────────────────────────────────────────────────────────────
# Auto-resume from checkpoint
# ─────────────────────────────────────────────────────────────
checkpoint_files = glob.glob(os.path.join(out_dir, "checkpoint_*.pth"))

if checkpoint_files:
    latest_ckpt = max(checkpoint_files, key=os.path.getmtime)
    print(f"Resuming from checkpoint: {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)

    state_dict = ckpt["model_state_dict"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    global_step            = ckpt.get("step",  0)
    start_epoch_from_ckpt  = ckpt.get("epoch", 0)
    print(f"Resumed from step {global_step}, epoch {start_epoch_from_ckpt}\n")
else:
    print("No checkpoint found — starting from scratch.\n")
    global_step           = 0
    start_epoch_from_ckpt = 0

# ─────────────────────────────────────────────────────────────
# PGD attack (no ImageNet normalization — inputs are in [0,1])
# ─────────────────────────────────────────────────────────────
attack = PGD(model, eps=pgd_eps, alpha=pgd_alpha,
             steps=pgd_steps, random_start=pgd_random_start)

# ─────────────────────────────────────────────────────────────
# TensorBoard writer & checkpoint helper
# ─────────────────────────────────────────────────────────────
writer             = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
checkpoint_prefix  = os.path.join(out_dir, "checkpoint")
best_val_acc       = 0.0
start_time         = time.time()


def save_checkpoint(step, epoch, tag="last"):
    state = (model.module if hasattr(model, "module") else model).state_dict()
    ckpt  = {
        "step"                : step,
        "epoch"               : epoch,
        "model_state_dict"    : state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{checkpoint_prefix}_{tag}_step{step}_epoch{epoch}_{ts}.pth"
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
print(f"Starting training at {datetime.now()}")
print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
print(f"PGD — eps={pgd_eps:.4f}  alpha={pgd_alpha:.4f}  steps={pgd_steps}\n")

for epoch in range(start_epoch_from_ckpt, epochs):
    epoch_loss        = 0.0
    epoch_correct_nat = 0
    epoch_samples     = 0

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for batch_idx, (x_batch, y_batch) in enumerate(pbar):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        batch_sz = x_batch.size(0)
        epoch_samples += batch_sz

        # ── Epoch 0: clean-only training ────────────────────────
        if epoch == 0:
            with torch.no_grad():
                preds_nat = model(x_batch).argmax(dim=1)
                epoch_correct_nat += (preds_nat == y_batch).sum().item()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x_batch)
                loss    = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * batch_sz

        # ── Epochs 1+: adversarial training with PGD ────────────
        else:
            model.eval()
            try:
                x_adv = attack(x_batch, y_batch)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            except RuntimeError as e:
                print(f"Batch attack failed ({e}); falling back to per-sample...")
                adv_list = []
                for i in range(batch_sz):
                    xi = x_batch[i:i+1]
                    yi = y_batch[i:i+1]
                    adv_list.append(attack(xi, yi).detach().cpu())
                    torch.cuda.empty_cache()
                x_adv = torch.clamp(torch.cat(adv_list, dim=0).to(device), 0.0, 1.0)
            finally:
                model.train()

            # Log natural vs adversarial accuracy (no grad)
            with torch.no_grad():
                preds_nat = model(x_batch).argmax(dim=1)
                preds_adv = model(x_adv).argmax(dim=1)
                epoch_correct_nat += (preds_nat == y_batch).sum().item()

                adv_acc_batch = 100.0 * (preds_adv == y_batch).float().mean().item()
                if global_step % summary_interval_steps == 0:
                    writer.add_scalar("train/accuracy_adv_batch", adv_acc_batch, global_step)

            # Train on adversarial examples
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x_adv)
                loss    = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * batch_sz

        global_step += 1

        # ── Logging ──────────────────────────────────────────────
        if global_step % log_interval_steps == 0:
            nat_acc  = 100.0 * epoch_correct_nat / (epoch_samples + 1e-12)
            speed    = epoch_samples / (time.time() - start_time + 1e-12)
            print(f"  [Step {global_step}] loss={loss.item():.4f}  "
                  f"nat_acc={nat_acc:.2f}%  speed={speed:.1f} ex/s")

        if global_step % summary_interval_steps == 0:
            writer.add_scalar("train/loss",
                              loss.item(), global_step)
            writer.add_scalar("train/accuracy_nat",
                              100.0 * epoch_correct_nat / (epoch_samples + 1e-12),
                              global_step)

        if global_step % checkpoint_interval_steps == 0:
            save_checkpoint(global_step, epoch, tag="intermediate")

        pbar.set_postfix(
            loss    = f"{loss.item():.4f}",
            nat_acc = f"{100.0 * epoch_correct_nat / (epoch_samples + 1e-12):.2f}%"
        )

    # ── End of epoch summary ─────────────────────────────────────
    epoch_loss_avg = epoch_loss / (epoch_samples + 1e-12)
    epoch_nat_acc  = 100.0 * epoch_correct_nat / (epoch_samples + 1e-12)
    print(f"\nEpoch {epoch+1} summary: loss={epoch_loss_avg:.4f}  nat_acc={epoch_nat_acc:.2f}%")
    writer.add_scalar("epoch/loss",         epoch_loss_avg, epoch + 1)
    writer.add_scalar("epoch/accuracy_nat", epoch_nat_acc,  epoch + 1)

    # ── Validation ───────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs        = model(inputs)
            loss_v         = criterion(outputs, labels)
            val_loss      += loss_v.item() * inputs.size(0)
            _, preds       = outputs.max(1)
            val_samples   += inputs.size(0)
            val_correct   += preds.eq(labels).sum().item()

    val_loss_avg = val_loss / (val_samples + 1e-12)
    val_acc      = 100.0 * val_correct / (val_samples + 1e-12)
    print(f"Validation: loss={val_loss_avg:.4f}  acc={val_acc:.2f}%")
    writer.add_scalar("val/loss",     val_loss_avg, epoch + 1)
    writer.add_scalar("val/accuracy", val_acc,      epoch + 1)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(global_step, epoch, tag="bestval")

    # ── Save per-epoch checkpoint ─────────────────────────────────
    save_checkpoint(global_step, epoch, tag=f"epoch{epoch+1}")

# ─────────────────────────────────────────────────────────────
# Final save
# ─────────────────────────────────────────────────────────────
print("\nTraining complete.")
save_checkpoint(global_step, epochs - 1, tag="final")

final_state = (model.module if hasattr(model, "module") else model).state_dict()
torch.save(final_state, os.path.join(out_dir, "model_weights.pt"))
print(f"Final weights saved to {out_dir}/model_weights.pt")

# ─────────────────────────────────────────────────────────────
# Test set evaluation (overall + per-class)
# ─────────────────────────────────────────────────────────────
print("\nEvaluating on held-out test set ...")
model.eval()
test_correct, test_total         = 0, 0
class_correct = [0] * num_classes
class_total   = [0] * num_classes

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs        = model(inputs)
        _, predicted   = outputs.max(1)
        test_total    += labels.size(0)
        test_correct  += predicted.eq(labels).sum().item()
        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += predicted[mask].eq(labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

print(f"\nOverall Test Accuracy : {100. * test_correct / test_total:.2f}%")
print("Per-class Accuracy:")
for c in range(num_classes):
    if class_total[c] > 0:
        acc = 100. * class_correct[c] / class_total[c]
        print(f"  {CLASS_NAMES[c]:10s}: {acc:.2f}%  ({class_correct[c]}/{class_total[c]})")

writer.close()