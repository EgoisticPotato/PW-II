"""Train the base EfficientNetB0_GCBAM classifier on the Fetal Planes dataset.

Run this FIRST before RL defense training.

Usage:
    python train_classifier.py [--epochs 15] [--batch_size 64] [--lr 0.001]
                               [--checkpoint_dir checkpoints] [--resume]
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_classifier import (
    CLASS_NAMES,
    EfficientNetB0_GCBAM,
    load_fetal_planes_hf,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train base classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use ImageNet pretrained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Dataset ──────────────────────────────────────────────────
    train_dataset, val_dataset, test_dataset = load_fetal_planes_hf(
        train_ratio=0.70, val_ratio=0.15, seed=42,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # ── Model ────────────────────────────────────────────────────
    model = EfficientNetB0_GCBAM(num_classes=6, pretrained=args.pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Train ────────────────────────────────────────────────────
    print("Starting training ...")
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )

    # ── Test evaluation ──────────────────────────────────────────
    print("\nEvaluating on test set ...")
    model.eval()
    test_correct, test_total = 0, 0
    class_correct = [0] * 6
    class_total = [0] * 6

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            for c in range(6):
                mask = labels == c
                class_correct[c] += predicted[mask].eq(labels[mask]).sum().item()
                class_total[c] += mask.sum().item()

    print(f"\nOverall Test Accuracy: {100. * test_correct / test_total:.2f}%")
    for c in range(6):
        if class_total[c] > 0:
            acc = 100. * class_correct[c] / class_total[c]
            print(f"  {CLASS_NAMES[c]:10s}: {acc:.2f}%  ({class_correct[c]}/{class_total[c]})")


if __name__ == "__main__":
    main()
