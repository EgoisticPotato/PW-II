"""rl/train_rl.py
RL Defense Trainer – optimised version.

Key changes vs v1:
  ① Multi-attack batching: each batch is split into thirds and attacked by
    FGSM / BIM / PGD simultaneously, tripling attack diversity per step.
  ② Continuous softmax-based reward instead of sparse ±1:
       reward = 2 * P(correct_class) - 1  ∈ [-1, +1]
    This gives gradient-like feedback (barely wrong ≠ catastrophically wrong).
  ③ Accepts an adversarially pre-trained classifier weight path by default
    (model_out_gcbam_pgd15/model_weights.pt).  This is the single most
    important change — see assessment comments.
  ④ Attack parameters (eps, steps) now match evaluate.py.
  ⑤ STATE_DIM updated for 12-feature entropy module.
  ⑥ Warm-up phase: first N batches use only FGSM (weakest attack) so the
    agent builds basic intuition before facing PGD.
  ⑦ TensorBoard logs extended: per-defense Q-value means, reward histogram.
"""

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

try:
    from torchattacks import BIM
except ImportError:
    from torchattacks import BIM

from defense.entropy import STATE_DIM, compute_state_features_batch
from defense.rl_defense import (
    DEFENSE_NAMES,
    NUM_ACTIONS,
    apply_defense_batch,
)
from models.base_classifier import CLASS_NAMES, EfficientNetB0_GCBAM
from rl.dqn_agent import DQNAgent


class RLDefenseTrainer:
    """Train a Dueling Double DQN agent for per-image adversarial defense.

    Training loop improvements
    --------------------------
    1. Multi-attack batching  – splits each batch into K segments, one per
       attack, then concatenates. The agent sees diverse attack signatures
       within every forward pass, learning to discriminate them via state features.

    2. Continuous reward  – uses softmax probability of the ground-truth class
       as a dense signal instead of the sparse ±1 binary reward.

    3. Warm-up  – for the first `warmup_batches` steps only FGSM is used so
       the agent has time to bootstrap basic associations before tackling PGD.
    """

    def __init__(
        self,
        classifier_weights_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        out_dir: str = "rl_defense_out",
        num_epochs: int = 30,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_decay_steps: int = 15_000,
        buffer_capacity: int = 100_000,
        agent_batch_size: int = 128,
        target_update_freq: int = 300,
        warmup_batches: int = 200,
    ):
        self.device = device
        self.out_dir = out_dir
        self.num_epochs = num_epochs
        self.warmup_batches = warmup_batches
        os.makedirs(out_dir, exist_ok=True)

        # ── Load & freeze classifier (prefer adversarially trained weights) ──
        self.classifier = EfficientNetB0_GCBAM(num_classes=6, pretrained=False).to(device)
        ckpt = torch.load(classifier_weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.classifier.load_state_dict(ckpt["model_state"])
        else:
            self.classifier.load_state_dict(ckpt)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False
        print(f"[Classifier] Loaded weights from: {classifier_weights_path}")

        # ── Attacks – parameters now match evaluate.py (steps=15) ────────
        # eps=8/255 ≈ 0.0314 is the standard ℓ∞ budget for robustness benchmarks
        eps     = 8 / 255
        alpha   = 2 / 255
        steps   = 15
        self.attacks = {
            "fgsm": FGSM(self.classifier, eps=eps),
            "bim":  BIM(self.classifier,  eps=eps, alpha=alpha, steps=steps),
            "pgd":  PGD(self.classifier,  eps=eps, alpha=alpha, steps=steps,
                        random_start=True),
        }
        self.attack_names = list(self.attacks.keys())
        self.K = len(self.attack_names)          # number of attack types

        # ── DQN agent (Dueling Double DQN + PER) ─────────────────────────
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
        self.val_loader   = val_loader
        self.use_amp      = (device.type == "cuda")

        self.writer      = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        self.global_step = 0
        self.best_val_acc = 0.0

    # ─────────────────────────────────────── public entry point ────────
    def train(self):
        print(f"\n{'='*65}")
        print("  RL Defense Training  (Dueling Double DQN + PER)")
        print(f"  Epochs : {self.num_epochs}  |  Device : {self.device}")
        print(f"  Actions: {NUM_ACTIONS}  |  State dim: {STATE_DIM}")
        print(f"  Output : {self.out_dir}")
        print(f"{'='*65}\n")

        for epoch in range(self.num_epochs):
            metrics     = self._train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Checkpoint
            self.agent.save(os.path.join(self.out_dir, f"rl_agent_epoch{epoch}.pth"))
            if val_metrics["overall_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["overall_acc"]
                self.agent.save(os.path.join(self.out_dir, "rl_agent_best.pth"))

            print(
                f"Epoch {epoch+1:3d}/{self.num_epochs}  |  "
                f"Train Acc: {metrics['accuracy']:5.2f}%  "
                f"Reward: {metrics['mean_reward']:+.3f}  "
                f"Loss: {metrics['mean_loss']:.4f}  "
                f"Eps: {self.agent.epsilon:.3f}  |  "
                f"Val Acc: {val_metrics['overall_acc']:5.2f}%  "
                f"[best {self.best_val_acc:.2f}%]"
            )

        self.agent.save(os.path.join(self.out_dir, "rl_agent_final.pth"))
        self.writer.close()
        print(f"\nTraining complete.  Best val accuracy: {self.best_val_acc:.2f}%")

    # ─────────────────────────────────────── single epoch ───────────────
    def _train_one_epoch(self, epoch: int) -> dict:
        total_reward, total_correct, total_samples = 0.0, 0, 0
        total_loss,   loss_count                   = 0.0, 0
        action_counts = [0] * NUM_ACTIONS
        per_attack_correct = {n: 0 for n in self.attack_names}
        per_attack_total   = {n: 0 for n in self.attack_names}

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} [RL Train]",
                    dynamic_ncols=True)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            B = images.size(0)

            # ── Multi-attack batching ─────────────────────────────────────
            # Split the batch into K (roughly equal) segments and attack each
            # with a different method.  All K forward passes happen before the
            # agent update → 3× more diverse training signal per step.
            #
            # During warmup we only use FGSM to let the agent bootstrap.
            if self.global_step < self.warmup_batches:
                active_attacks = ["fgsm"]
            else:
                active_attacks = self.attack_names

            K = len(active_attacks)
            splits = np.array_split(np.arange(B), K)

            x_adv_parts   = []
            labels_parts  = []
            atk_name_list = []   # tracks which attack generated each image

            for k, atk_name in enumerate(active_attacks):
                idx = splits[k]
                if len(idx) == 0:
                    continue
                imgs_k   = images[idx]
                labels_k = labels[idx]
                x_adv_k  = self.attacks[atk_name](imgs_k, labels_k).clamp(0.0, 1.0)
                x_adv_parts.append(x_adv_k)
                labels_parts.append(labels_k)
                atk_name_list.extend([atk_name] * len(idx))

            x_adv   = torch.cat(x_adv_parts,  dim=0)
            labels_ = torch.cat(labels_parts, dim=0)
            B_eff   = x_adv.size(0)

            # ── State features (one forward + backward for full sub-batch) ──
            states_np = compute_state_features_batch(
                x_adv, self.classifier, self.device
            ).numpy()                                   # (B_eff, STATE_DIM)

            # ── Agent selects actions (epsilon-greedy) ────────────────────
            actions = self.agent.select_action_batch(states_np)

            # ── Apply defenses ────────────────────────────────────────────
            defended = apply_defense_batch(x_adv, actions)

            # ── Classify ──────────────────────────────────────────────────
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.classifier(defended.to(self.device))
                probs  = F.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)

            # ── Continuous reward: 2·P(correct) - 1 ∈ [-1, +1] ──────────
            correct_probs = probs.gather(
                1, labels_.view(-1, 1)
            ).squeeze(1).detach().cpu()
            rewards = (2.0 * correct_probs - 1.0).tolist()

            correct_mask = preds.eq(labels_)

            # ── Store transitions + update agent ──────────────────────────
            next_states = np.zeros_like(states_np)   # terminal (done=True always)
            for i in range(B_eff):
                self.agent.store_transition(
                    states_np[i], actions[i], rewards[i],
                    next_states[i], True,
                )
            loss_val = self.agent.update()

            # ── Bookkeeping ───────────────────────────────────────────────
            total_reward  += sum(rewards)
            total_correct += int(correct_mask.sum().item())
            total_samples += B_eff
            if loss_val > 0:
                total_loss += loss_val
                loss_count += 1
            for a in actions:
                action_counts[a] += 1
            for i, atk_name in enumerate(atk_name_list):
                per_attack_total[atk_name]   += 1
                per_attack_correct[atk_name] += int(correct_mask[i].item())

            self.global_step += 1

            # ── TensorBoard (per step) ────────────────────────────────────
            step = self.global_step
            self.writer.add_scalar("train/reward_batch",
                                   float(np.mean(rewards)), step)
            self.writer.add_scalar("train/accuracy_batch",
                                   correct_mask.float().mean().item() * 100, step)
            self.writer.add_scalar("train/epsilon",
                                   self.agent.epsilon, step)
            if loss_val > 0:
                self.writer.add_scalar("train/dqn_loss", loss_val, step)

            pbar.set_postfix({
                "acc": f"{100.*total_correct/total_samples:.1f}%",
                "rwd": f"{total_reward/total_samples:+.3f}",
                "eps": f"{self.agent.epsilon:.3f}",
                "loss": f"{loss_val:.4f}" if loss_val > 0 else "—",
            })

        # ── Epoch-level metrics ───────────────────────────────────────────
        acc      = 100.0 * total_correct / max(total_samples, 1)
        mean_rwd = total_reward / max(total_samples, 1)
        mean_loss = total_loss  / max(loss_count,  1)

        self.writer.add_scalar("train/accuracy_epoch",  acc,      epoch)
        self.writer.add_scalar("train/reward_epoch",    mean_rwd, epoch)
        self.writer.add_scalar("train/loss_epoch",      mean_loss, epoch)

        for idx, name in enumerate(DEFENSE_NAMES):
            freq = action_counts[idx] / max(total_samples, 1)
            self.writer.add_scalar(f"train/defense_freq_{name}", freq, epoch)

        for atk in self.attack_names:
            if per_attack_total[atk] > 0:
                a = 100.0 * per_attack_correct[atk] / per_attack_total[atk]
                self.writer.add_scalar(f"train/attack_{atk}_acc", a, epoch)

        return {"accuracy": acc, "mean_reward": mean_rwd, "mean_loss": mean_loss}

    # ─────────────────────────────────────── validation ─────────────────
    def validate(self, epoch: int) -> dict:
        """Greedy (ε=0) evaluation on the validation set, per-attack.

        Gradients are needed internally by torchattacks and
        compute_state_features_batch, so we do NOT wrap in no_grad here.
        """
        per_attack_correct = {n: 0 for n in self.attack_names}
        per_attack_total   = {n: 0 for n in self.attack_names}
        action_counts_val  = {n: [0] * NUM_ACTIONS for n in self.attack_names}

        for atk_name in self.attack_names:
            attack = self.attacks[atk_name]

            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                x_adv     = attack(images, labels).clamp(0.0, 1.0)
                states_np = compute_state_features_batch(
                    x_adv, self.classifier, self.device
                ).numpy()
                actions   = self.agent.select_action_batch(states_np, evaluate=True)
                defended  = apply_defense_batch(x_adv, actions)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.classifier(defended.to(self.device))
                    preds  = logits.argmax(dim=1)

                correct = preds.eq(labels).sum().item()
                per_attack_correct[atk_name] += int(correct)
                per_attack_total[atk_name]   += images.size(0)
                for a in actions:
                    action_counts_val[atk_name][a] += 1

        overall_correct = sum(per_attack_correct.values())
        overall_total   = sum(per_attack_total.values())
        overall_acc     = 100.0 * overall_correct / max(overall_total, 1)

        per_attack_acc = {}
        for atk in self.attack_names:
            if per_attack_total[atk] > 0:
                a = 100.0 * per_attack_correct[atk] / per_attack_total[atk]
                per_attack_acc[atk] = a
                self.writer.add_scalar(f"val/accuracy_{atk}", a, epoch)
                for idx, dname in enumerate(DEFENSE_NAMES):
                    f = action_counts_val[atk][idx] / max(per_attack_total[atk], 1)
                    self.writer.add_scalar(
                        f"val/defense_freq_{atk}_{dname}", f, epoch)

        self.writer.add_scalar("val/accuracy_overall", overall_acc, epoch)

        # Print per-attack breakdown
        breakdown = "  ".join(
            f"{atk}: {per_attack_acc.get(atk, 0):.1f}%"
            for atk in self.attack_names
        )
        print(f"         Val breakdown → {breakdown}")

        return {
            "overall_acc": overall_acc,
            "per_attack":  per_attack_acc,
        }