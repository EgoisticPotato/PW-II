import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchattacks import FGSM, PGD
from tqdm import tqdm

from defense.entropy import STATE_DIM, compute_state_features, compute_state_features_batch
from defense.rl_defense import (
    DEFENSE_NAMES,
    NUM_ACTIONS,
    apply_defense_batch,
)
from models.base_classifier import CLASS_NAMES, EfficientNetB0_GCBAM
from rl.dqn_agent import DQNAgent


# torchattacks ships BIM under the name BIM
try:
    from torchattacks import BIM
except ImportError:
    from torchattacks import BIM as BIM


# ─────────────────────────────────────────────────────────────
# RL Defense Trainer
# ─────────────────────────────────────────────────────────────
class RLDefenseTrainer:
    """Train a DQN agent that selects per-image adversarial defenses.

    Feedback loop (per batch):
        1. Randomly pick an attack (FGSM / BIM / PGD).
        2. Generate adversarial images.
        3. Compute state features from attacked images only.
        4. Agent selects defense action (epsilon-greedy).
        5. Apply selected defense per-image.
        6. Classify defended image with the *frozen* classifier.
        7. Reward: +1 correct, -1 wrong.
        8. Store transition and train DQN via experience replay.
    """

    def __init__(
        self,
        classifier_weights_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        out_dir: str = "rl_defense_out",
        num_epochs: int = 30,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_decay_steps: int = 10_000,
        buffer_capacity: int = 50_000,
        agent_batch_size: int = 64,
        target_update_freq: int = 500,
    ):
        self.device = device
        self.out_dir = out_dir
        self.num_epochs = num_epochs
        os.makedirs(out_dir, exist_ok=True)

        # ── Load & freeze classifier ──────────────────────────────
        self.classifier = EfficientNetB0_GCBAM(num_classes=6, pretrained=False).to(device)
        ckpt = torch.load(classifier_weights_path, map_location=device)
        # Support both full-checkpoint and plain state-dict formats
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.classifier.load_state_dict(ckpt["model_state"])
        else:
            self.classifier.load_state_dict(ckpt)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        # ── Attacks (bound to frozen classifier) ──────────────────
        # Fewer steps during RL training (7 vs 15) — the agent needs
        # diverse perturbations, not maximally strong attacks.
        self.attacks = {
            "fgsm": FGSM(self.classifier, eps=8 / 255),
            "bim":  BIM(self.classifier, eps=8 / 255, alpha=2 / 255, steps=7),
            "pgd":  PGD(self.classifier, eps=8 / 255, alpha=2 / 255, steps=7,
                        random_start=True),
        }
        self.attack_names = list(self.attacks.keys())

        # ── DQN agent ─────────────────────────────────────────────
        self.agent = DQNAgent(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            lr=lr,
            gamma=gamma,
            epsilon_decay_steps=epsilon_decay_steps,
            buffer_capacity=buffer_capacity,
            batch_size=agent_batch_size,
            target_update_freq=target_update_freq,
            device=str(device),
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # AMP for mixed-precision (RTX 3060 has fp16 Tensor Cores)
        self.use_amp = (device.type == "cuda")

        self.writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        self.global_step = 0
        self.best_val_acc = 0.0

    # ─────────────────────────────────── training ──────────────
    def train(self):
        print(f"\n{'='*60}")
        print("  RL Defense Training")
        print(f"  Epochs: {self.num_epochs}  |  Device: {self.device}")
        print(f"  Output: {self.out_dir}")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            metrics = self._train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # ── Checkpoint ────────────────────────────────────────
            self.agent.save(os.path.join(self.out_dir, f"rl_agent_epoch{epoch}.pth"))
            if val_metrics["overall_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["overall_acc"]
                self.agent.save(os.path.join(self.out_dir, "rl_agent_best.pth"))

            print(
                f"Epoch {epoch+1}/{self.num_epochs}  |  "
                f"Train Acc: {metrics['accuracy']:.2f}%  "
                f"Reward: {metrics['mean_reward']:.3f}  "
                f"Loss: {metrics['mean_loss']:.4f}  "
                f"Eps: {self.agent.epsilon:.3f}  |  "
                f"Val Acc: {val_metrics['overall_acc']:.2f}%"
            )

        # Final save
        self.agent.save(os.path.join(self.out_dir, "rl_agent_final.pth"))
        self.writer.close()
        print(f"\nTraining complete.  Best val accuracy: {self.best_val_acc:.2f}%")

    # ─────────────────────────────── single epoch ──────────────
    def _train_one_epoch(self, epoch: int) -> dict:
        total_reward, total_correct, total_samples = 0.0, 0, 0
        total_loss, loss_count = 0.0, 0
        action_counts = [0] * NUM_ACTIONS
        per_attack_correct = {n: 0 for n in self.attack_names}
        per_attack_total   = {n: 0 for n in self.attack_names}

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} [RL Train]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            B = images.size(0)

            # 1. Random attack
            atk_name = random.choice(self.attack_names)
            attack = self.attacks[atk_name]
            x_adv = attack(images, labels).clamp(0.0, 1.0)

            # 2. State features – single batched forward+backward pass
            states_np = compute_state_features_batch(
                x_adv, self.classifier, self.device
            ).numpy()  # (B, STATE_DIM)

            # 3. Agent selects actions
            actions = self.agent.select_action_batch(states_np)

            # 4. Apply defenses
            defended = apply_defense_batch(x_adv, actions)

            # 5. Classify
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.classifier(defended.to(self.device))
                preds = logits.argmax(dim=1)

            # 6. Rewards
            correct_mask = preds.eq(labels)
            rewards = [1.0 if c else -1.0 for c in correct_mask.cpu().tolist()]

            # 7. Store transitions & update
            next_states = np.zeros_like(states_np)
            for i in range(B):
                self.agent.store_transition(
                    states_np[i], actions[i], rewards[i],
                    next_states[i], True,
                )
            loss_val = self.agent.update()

            # ── Bookkeeping ───────────────────────────────────────
            total_reward += sum(rewards)
            total_correct += int(correct_mask.sum().item())
            total_samples += B
            if loss_val > 0:
                total_loss += loss_val
                loss_count += 1
            for a in actions:
                action_counts[a] += 1
            per_attack_correct[atk_name] += int(correct_mask.sum().item())
            per_attack_total[atk_name] += B

            self.global_step += 1

            # ── TensorBoard (per-step) ────────────────────────────
            step = self.global_step
            self.writer.add_scalar("train/reward_batch",
                                   np.mean(rewards), step)
            self.writer.add_scalar("train/accuracy_batch",
                                   correct_mask.float().mean().item() * 100, step)
            self.writer.add_scalar("train/epsilon", self.agent.epsilon, step)
            if loss_val > 0:
                self.writer.add_scalar("train/dqn_loss", loss_val, step)

            pbar.set_postfix({
                "acc": f"{100.*total_correct/total_samples:.1f}%",
                "rwd": f"{total_reward/total_samples:.2f}",
                "eps": f"{self.agent.epsilon:.2f}",
            })

        # ── Epoch-level TensorBoard ───────────────────────────────
        acc = 100.0 * total_correct / max(total_samples, 1)
        mean_rwd = total_reward / max(total_samples, 1)
        mean_loss = total_loss / max(loss_count, 1)

        self.writer.add_scalar("train/accuracy_epoch", acc, epoch)
        self.writer.add_scalar("train/reward_epoch", mean_rwd, epoch)
        self.writer.add_scalar("train/loss_epoch", mean_loss, epoch)

        # Defense selection frequency
        for idx, name in enumerate(DEFENSE_NAMES):
            freq = action_counts[idx] / max(total_samples, 1)
            self.writer.add_scalar(f"train/defense_freq_{name}", freq, epoch)

        # Per-attack accuracy
        for atk in self.attack_names:
            if per_attack_total[atk] > 0:
                a = 100.0 * per_attack_correct[atk] / per_attack_total[atk]
                self.writer.add_scalar(f"train/attack_{atk}_acc", a, epoch)

        return {"accuracy": acc, "mean_reward": mean_rwd, "mean_loss": mean_loss}

    # ──────────────────────────────── validation ───────────────
    def validate(self, epoch: int) -> dict:
        """Greedy evaluation on the validation set, per-attack.

        NOTE: no @torch.no_grad() here because torchattacks and
        compute_state_features_batch both need gradients internally.
        """
        per_attack_correct = {n: 0 for n in self.attack_names}
        per_attack_total   = {n: 0 for n in self.attack_names}
        action_counts = {n: [0]*NUM_ACTIONS for n in self.attack_names}

        for atk_name in self.attack_names:
            attack = self.attacks[atk_name]
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                B = images.size(0)

                # attacks need grad enabled (they handle it internally)
                x_adv = attack(images, labels).clamp(0.0, 1.0)

                # state features need one forward+backward pass
                states_np = compute_state_features_batch(
                    x_adv, self.classifier, self.device
                ).numpy()

                actions = self.agent.select_action_batch(states_np,
                                                         evaluate=True)
                defended = apply_defense_batch(x_adv, actions)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.classifier(defended.to(self.device))
                    preds = logits.argmax(dim=1)
                correct = preds.eq(labels).sum().item()

                per_attack_correct[atk_name] += int(correct)
                per_attack_total[atk_name] += B
                for a in actions:
                    action_counts[atk_name][a] += 1

        overall_correct = sum(per_attack_correct.values())
        overall_total   = sum(per_attack_total.values())
        overall_acc = 100.0 * overall_correct / max(overall_total, 1)

        for atk in self.attack_names:
            if per_attack_total[atk] > 0:
                a = 100.0 * per_attack_correct[atk] / per_attack_total[atk]
                self.writer.add_scalar(f"val/accuracy_{atk}", a, epoch)
                # Defense freq
                for idx, dname in enumerate(DEFENSE_NAMES):
                    f = action_counts[atk][idx] / max(per_attack_total[atk], 1)
                    self.writer.add_scalar(
                        f"val/defense_freq_{atk}_{dname}", f, epoch)

        self.writer.add_scalar("val/accuracy_overall", overall_acc, epoch)
        return {"overall_acc": overall_acc,
                "per_attack": {
                    k: 100.0 * per_attack_correct[k] / max(per_attack_total[k], 1)
                    for k in self.attack_names
                }}
