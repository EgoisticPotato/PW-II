# Install dependencies:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install tqdm opencv-python numpy datasets huggingface_hub Pillow

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────
# Ghost Module
# ─────────────────────────────────────────────────────────────
class GhostModule(nn.Module):
    def __init__(self, in_channels, ratio=2):
        super(GhostModule, self).__init__()
        self.out_channels = in_channels // ratio

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
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


# ─────────────────────────────────────────────────────────────
# CBAM (Convolutional Block Attention Module)
# ─────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
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
        out = avg_out + max_out
        return out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ─────────────────────────────────────────────────────────────
# EfficientNet-B0 + Ghost Module + CBAM
# ─────────────────────────────────────────────────────────────
class EfficientNetB0_GCBAM(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super(EfficientNetB0_GCBAM, self).__init__()

        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = efficientnet_b0(weights=None)

        self.features = self.backbone.features
        backbone_out_channels = 1280

        self.ghost = GhostModule(backbone_out_channels, ratio=2)
        ghost_out_channels = backbone_out_channels

        self.cbam = CBAM(ghost_out_channels, reduction_ratio=16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(ghost_out_channels, 256),
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
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────
# HuggingFace Fetal Planes Dataset Wrapper
#
# Dataset: qingyuyang/Fetal_Planes_DB
#   - 6 classes: Abdomen(0), Brain(1), Femur(2), Thorax(3), Cervix(4), Other(5)
#   - Contains 'image' (PIL Image) and 'label' (int) columns
#   - Only a 'train' split is available; we manually split train/val/test
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['Abdomen', 'Brain', 'Femur', 'Thorax', 'Cervix', 'Other']


class HFFetalPlanesDataset(Dataset):
    """
    Wraps the HuggingFace qingyuyang/Fetal_Planes_DB dataset.
    Converts PIL images → grayscale → RGB → resized tensor.
    """

    def __init__(self, hf_split, img_size=224):
        """
        Args:
            hf_split : a HuggingFace Dataset split (e.g. dataset['train'])
            img_size  : target image size (square)
        """
        self.hf_split = hf_split
        self.img_size = img_size

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        sample = self.hf_split[idx]

        # PIL Image from HuggingFace
        pil_img = sample['image']

        # Convert to grayscale then back to RGB (matches original pipeline)
        gray = pil_img.convert('L')                    # grayscale PIL
        rgb  = gray.convert('RGB')                     # H×W×3 PIL

        # Resize
        rgb = rgb.resize((self.img_size, self.img_size), Image.BILINEAR)

        # To numpy → normalize → tensor
        img_np = np.array(rgb, dtype=np.float32) / 255.0   # H×W×3
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # 3×H×W

        label = int(sample['label'])
        return tensor, label


# ─────────────────────────────────────────────────────────────
# Load & Split Dataset from HuggingFace Hub
# ─────────────────────────────────────────────────────────────
def load_fetal_planes_hf(train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Loads qingyuyang/Fetal_Planes_DB from HuggingFace Hub.
    Since only a 'train' split exists, we manually split into train/val/test.

    Returns:
        train_dataset, val_dataset, test_dataset  (HFFetalPlanesDataset instances)
    """
    print("Loading Fetal Planes dataset from HuggingFace Hub...")
    print("Dataset: qingyuyang/Fetal_Planes_DB")
    print("Classes: Abdomen, Brain, Femur, Thorax, Cervix, Other\n")

    hf_dataset = load_dataset("qingyuyang/Fetal_Planes_DB", split="train")

    # Shuffle deterministically
    hf_dataset = hf_dataset.shuffle(seed=seed)

    total = len(hf_dataset)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    hf_train = hf_dataset.select(range(0, train_end))
    hf_val   = hf_dataset.select(range(train_end, val_end))
    hf_test  = hf_dataset.select(range(val_end, total))

    print(f"Total samples : {total}")
    print(f"Train samples : {len(hf_train)}")
    print(f"Val   samples : {len(hf_val)}")
    print(f"Test  samples : {len(hf_test)}\n")

    train_dataset = HFFetalPlanesDataset(hf_train)
    val_dataset   = HFFetalPlanesDataset(hf_val)
    test_dataset  = HFFetalPlanesDataset(hf_test)

    return train_dataset, val_dataset, test_dataset


# ─────────────────────────────────────────────────────────────
# Training Function (with checkpoint resume)
# ─────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, num_epochs=15, device='cuda',
                checkpoint_dir='checkpoints', resume=True):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from latest checkpoint if available
    if resume:
        checkpoints = [f for f in os.listdir(checkpoint_dir)
                       if f.startswith('epoch_') and f.endswith('.pt')]
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            print(f"Resuming from checkpoint: {ckpt_path}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        # ── Training phase ──────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{train_loss / len(train_loader):.4f}",
                'acc' : f"{100. * train_correct / train_total:.2f}%"
            })

        # ── Validation phase ─────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]  '):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # ── Metrics ──────────────────────────────────────────────
        train_acc  = 100. * train_correct / train_total
        val_acc    = 100. * val_correct   / val_total
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch + 1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%  |  '
              f'Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.2f}%')

        scheduler.step(val_loss)

        # ── Save checkpoint ───────────────────────────────────────
        snapshot_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history'        : history
        }, snapshot_path)
        print(f"Checkpoint saved: {snapshot_path}")

    # Save final weights
    torch.save(model.state_dict(), 'model_weights.pt')
    print("\nFinal model weights saved to model_weights.pt")
    return history


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # ── Load dataset from HuggingFace ────────────────────────────
    # Set HF_TOKEN env variable if the dataset requires authentication:
    #   export HF_TOKEN="hf_your_token_here"
    # or login once via CLI:
    #   huggingface-cli login
    train_dataset, val_dataset, test_dataset = load_fetal_planes_hf(
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42
    )

    # ── DataLoaders ──────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────
    model = EfficientNetB0_GCBAM(num_classes=6, pretrained=False)
    model = model.to(device)

    print("Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}\n")

    # ── Train ────────────────────────────────────────────────────
    print("Starting training...")
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=15,
        device=device,
        checkpoint_dir='checkpoints',
        resume=True
    )

    # ── Evaluate on test set ──────────────────────────────────────
    print("\nEvaluating on test set...")
    model.eval()
    test_correct, test_total = 0, 0

    # Per-class stats
    class_correct = [0] * 6
    class_total   = [0] * 6

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total   += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for c in range(6):
                mask = labels == c
                class_correct[c] += predicted[mask].eq(labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'\nOverall Test Accuracy: {test_acc:.2f}%\n')
    print("Per-class Accuracy:")
    for c in range(6):
        if class_total[c] > 0:
            acc = 100. * class_correct[c] / class_total[c]
            print(f"  {CLASS_NAMES[c]:10s}: {acc:.2f}%  ({class_correct[c]}/{class_total[c]})")