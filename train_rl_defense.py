"""Orchestrator: train the RL-based adversarial defense agent.

Prerequisite: run  python train_classifier.py  first to produce model_weights.pt.

Usage:
    python train_rl_defense.py [--classifier_weights model_weights.pt]
                               [--epochs 30] [--lr 1e-3] [--out_dir rl_defense_out]
"""

import argparse

import torch
from torch.utils.data import DataLoader

from models.base_classifier import load_fetal_planes_hf
from rl.train_rl import RLDefenseTrainer


def main():
    parser = argparse.ArgumentParser(description="Train RL defense agent")
    parser.add_argument("--classifier_weights", type=str, default="model_weights.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="DataLoader batch size (not DQN minibatch)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_decay_steps", type=int, default=10_000)
    parser.add_argument("--buffer_capacity", type=int, default=50_000)
    parser.add_argument("--agent_batch_size", type=int, default=64)
    parser.add_argument("--target_update_freq", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="rl_defense_out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset (same split as classifier training) ───────────────
    train_dataset, val_dataset, _ = load_fetal_planes_hf(
        train_ratio=0.70, val_ratio=0.15, seed=42,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ── Trainer ───────────────────────────────────────────────────
    trainer = RLDefenseTrainer(
        classifier_weights_path=args.classifier_weights,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay_steps=args.epsilon_decay_steps,
        buffer_capacity=args.buffer_capacity,
        agent_batch_size=args.agent_batch_size,
        target_update_freq=args.target_update_freq,
    )

    trainer.train()
    print(f"\nRL defense training complete.  Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
