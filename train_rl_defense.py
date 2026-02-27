"""train_rl_defense.py
Orchestrator: train the RL-based adversarial defense agent.

IMPORTANT – run  python train_classifier.py  first to produce the base
classifier weights, then (optionally) adversarially fine-tune it via
train_adversarial.py to produce model_out_gcbam_pgd15/model_weights.pt.

The RL agent should be trained on top of the ADVERSARIALLY TRAINED
classifier, not the plain classifier, because the frozen model must already
have partial robustness for the RL signal to be meaningful.

Usage
-----
python train_rl_defense.py \\
    --classifier_weights model_out_gcbam_pgd15/model_weights.pt \\
    --epochs 30 \\
    --batch_size 32 \\
    --out_dir rl_defense_out

All arguments have sensible defaults that match the recommendations in the
optimization analysis.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from models.base_classifier import load_fetal_planes_hf
from rl.train_rl import RLDefenseTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train RL defense agent")

    # ── Paths ───────────────────────────────────────────────────────────
    p.add_argument(
        "--classifier_weights", type=str,
        # Default: adversarially trained model (critical for non-zero RL signal)
        default="model_out_gcbam_pgd15/model_weights.pt",
        help="Path to frozen classifier weights.  Use adversarially trained "
             "weights (PGD-AT) for best results.",
    )
    p.add_argument("--out_dir", type=str, default="rl_defense_out")

    # ── Data ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int, default=32,
                   help="DataLoader batch size (not DQN minibatch)")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--train_ratio",  type=float, default=0.70)
    p.add_argument("--val_ratio",    type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)

    # ── Training schedule ────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--warmup_batches", type=int, default=200,
                   help="Number of batches to use only FGSM (agent warm-up)")

    # ── DQN hyperparameters ──────────────────────────────────────────────
    p.add_argument("--lr",                  type=float, default=5e-4)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--epsilon_decay_steps", type=int,   default=15_000)
    p.add_argument("--buffer_capacity",     type=int,   default=100_000)
    p.add_argument("--agent_batch_size",    type=int,   default=128)
    p.add_argument("--target_update_freq",  type=int,   default=300)

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\nLoading dataset …")
    train_dataset, val_dataset, _ = load_fetal_planes_hf(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    print(f"\nInitialising RLDefenseTrainer …")
    print(f"  Classifier weights : {args.classifier_weights}")
    print(f"  Epochs             : {args.epochs}")
    print(f"  LR                 : {args.lr}")
    print(f"  Epsilon decay steps: {args.epsilon_decay_steps:,}")
    print(f"  Buffer capacity    : {args.buffer_capacity:,}")
    print(f"  Agent batch size   : {args.agent_batch_size}")
    print(f"  Target update freq : {args.target_update_freq}")
    print(f"  Warmup batches     : {args.warmup_batches}")

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
        warmup_batches=args.warmup_batches,
    )

    trainer.train()
    print(f"\nRL defense training complete.  Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()  