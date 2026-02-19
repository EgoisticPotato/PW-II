"""Full evaluation: static defenses vs RL dynamic defense.

Produces:
    - Accuracy heatmap  (condition x defense)
    - Per-attack grouped bar charts
    - Per-class accuracy charts
    - Confusion matrices
    - RL defense selection frequency
    - Perturbation analysis (L2 / Linf)
    - CBAM attention map comparisons
    - Reward / epsilon / loss curves (from TensorBoard logs)
    - McNemar statistical significance tests
    - summary_results.json

Usage:
    python evaluate.py --classifier_weights model_weights.pt \
                       --rl_agent_path rl_defense_out/rl_agent_final.pth \
                       --out_dir eval_results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchattacks import FGSM, PGD
from tqdm import tqdm

try:
    from torchattacks import BIM
except ImportError:
    from torchattacks import BIM as BIM

from defense.entropy import STATE_DIM, compute_state_features, compute_state_features_batch
from defense.rl_defense import (
    DEFENSE_NAMES,
    NUM_ACTIONS,
    apply_defense_batch,
    apply_gaussian_smoothing,
    apply_jpeg_compression,
    apply_median_filter,
    apply_bitdepth_reduction,
    apply_passthrough,
)
from models.base_classifier import CLASS_NAMES, EfficientNetB0_GCBAM, load_fetal_planes_hf
from rl.dqn_agent import DQNAgent


NUM_CLASSES = 6
CONDITIONS = ["clean", "fgsm", "bim", "pgd"]
STATIC_DEFENSES = ["none", "gaussian", "jpeg", "median", "bitdepth"]
ALL_DEFENSES = STATIC_DEFENSES + ["rl"]

_STATIC_FNS = {
    "none":     apply_passthrough,
    "gaussian": apply_gaussian_smoothing,
    "jpeg":     apply_jpeg_compression,
    "median":   apply_median_filter,
    "bitdepth": apply_bitdepth_reduction,
}


# ═════════════════════════════════════════════════════════════
# Evaluator
# ═════════════════════════════════════════════════════════════
class FullEvaluator:
    def __init__(self, classifier_weights_path: str, rl_agent_path: str,
                 test_loader: DataLoader, device: torch.device,
                 out_dir: str = "eval_results",
                 tb_log_dir: str | None = None):
        self.device = device
        self.out_dir = out_dir
        self.tb_log_dir = tb_log_dir
        os.makedirs(out_dir, exist_ok=True)

        # ── Classifier ────────────────────────────────────────────
        self.classifier = EfficientNetB0_GCBAM(num_classes=NUM_CLASSES,
                                                pretrained=False).to(device)
        ckpt = torch.load(classifier_weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.classifier.load_state_dict(ckpt["model_state"])
        else:
            self.classifier.load_state_dict(ckpt)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        # ── RL agent ──────────────────────────────────────────────
        self.agent = DQNAgent(state_dim=STATE_DIM, num_actions=NUM_ACTIONS,
                              device=str(device))
        self.agent.load(rl_agent_path)
        # online_net is the correct attribute in the updated DQNAgent
        self.agent.online_net.eval()

        # ── Attacks ───────────────────────────────────────────────
        self.attacks = {
            "fgsm": FGSM(self.classifier, eps=8/255),
            "bim":  BIM(self.classifier, eps=8/255, alpha=2/255, steps=15),
            "pgd":  PGD(self.classifier, eps=8/255, alpha=2/255, steps=15,
                        random_start=True),
        }

        self.test_loader = test_loader
        self.writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # ─────────────────────── main entry ──────────────────────────
    def evaluate_all(self):
        print(f"\n{'='*60}")
        print("  Full Evaluation: Static vs Dynamic (RL) Defense")
        print(f"{'='*60}\n")

        results = {}
        rl_actions = {}
        perturbation_data = {}
        attention_samples = {}

        for cond in CONDITIONS:
            print(f"\n── Condition: {cond} ──")
            rl_actions[cond] = []
            if cond != "clean":
                perturbation_data[cond] = {"l2": [], "linf": []}

            for defense in ALL_DEFENSES:
                all_preds, all_labels, all_correct = [], [], []

                for images, labels in tqdm(
                    self.test_loader,
                    desc=f"  {cond:5s} + {defense:9s}",
                    leave=False,
                ):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    B = images.size(0)

                    # ── Attack (or clean passthrough) ──
                    if cond == "clean":
                        x_input = images
                    else:
                        x_input = self.attacks[cond](images, labels).clamp(0, 1)

                    # ── Perturbation stats (once per attack condition) ──
                    if cond != "clean" and defense == "none":
                        diff = (x_input - images).view(B, -1)
                        for i in range(B):
                            perturbation_data[cond]["l2"].append(
                                diff[i].norm(2).item())
                            perturbation_data[cond]["linf"].append(
                                diff[i].abs().max().item())

                    # ── Defense ──
                    if defense == "rl":
                        states_np = compute_state_features_batch(
                            x_input, self.classifier, self.device
                        ).numpy()
                        actions = self.agent.select_action_batch(
                            states_np, evaluate=True)
                        rl_actions[cond].extend(actions)
                        defended = apply_defense_batch(x_input, actions)
                    else:
                        fn = _STATIC_FNS[defense]
                        defended = fn(x_input)

                    # ── Classify ──
                    with torch.no_grad():
                        logits = self.classifier(defended.to(self.device))
                        preds = logits.argmax(dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    all_correct.extend(preds.eq(labels).cpu().tolist())

                    # ── Attention samples (for PGD visualisation) ──
                    if (cond == "pgd" and defense in ("none", "rl")
                            and len(attention_samples.get((cond, defense), [])) < 5):
                        key = (cond, defense)
                        attention_samples.setdefault(key, [])
                        for i in range(min(2, B)):
                            attention_samples[key].append({
                                "clean":    images[i].cpu(),
                                "attacked": x_input[i].cpu(),
                                "defended": defended[i].cpu(),
                            })

                results[(cond, defense)] = {
                    "preds":   all_preds,
                    "labels":  all_labels,
                    "correct": all_correct,
                }

        # ── Metrics & plots ───────────────────────────────────────
        acc_table = self._accuracy_table(results)
        per_class = self._per_class_accuracy(results)
        conf_mats = self._confusion_matrices(results)
        rl_freq   = self._defense_frequency(rl_actions)
        sig_tests = self._statistical_tests(results)

        self._plot_accuracy_heatmap(acc_table)
        self._plot_per_attack_bars(acc_table)
        self._plot_per_class_accuracy(per_class)
        self._plot_confusion_matrices(conf_mats)
        self._plot_defense_frequency(rl_freq)
        self._plot_perturbation_analysis(perturbation_data)
        self._plot_attention_maps(attention_samples)
        self._plot_training_curves()

        summary = {
            "accuracy_table": {
                f"{c}+{d}": float(acc_table[i][j])
                for i, c in enumerate(CONDITIONS)
                for j, d in enumerate(ALL_DEFENSES)
            },
            "perturbation_stats": {
                atk: {
                    "l2_mean":   float(np.mean(v["l2"])),
                    "l2_std":    float(np.std(v["l2"])),
                    "linf_mean": float(np.mean(v["linf"])),
                    "linf_std":  float(np.std(v["linf"])),
                }
                for atk, v in perturbation_data.items()
            },
            "statistical_tests":    sig_tests,
            "rl_defense_frequency": rl_freq,
        }
        with open(os.path.join(self.out_dir, "summary_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

        self.writer.close()
        print(f"\n{'='*60}")
        print(f"  All results saved to {self.out_dir}/")
        print(f"{'='*60}")
        self._print_accuracy_table(acc_table)

    # ═══════════════════ metric computation ═══════════════════════
    def _accuracy_table(self, results) -> np.ndarray:
        table = np.zeros((len(CONDITIONS), len(ALL_DEFENSES)))
        for i, c in enumerate(CONDITIONS):
            for j, d in enumerate(ALL_DEFENSES):
                corr = results[(c, d)]["correct"]
                table[i, j] = 100.0 * sum(corr) / max(len(corr), 1)
        return table

    def _per_class_accuracy(self, results) -> dict:
        out = {}
        for (c, d), r in results.items():
            pca = []
            for cls in range(NUM_CLASSES):
                mask = [l == cls for l in r["labels"]]
                total = sum(mask)
                if total > 0:
                    correct = sum(co for co, m in zip(r["correct"], mask) if m)
                    pca.append(100.0 * correct / total)
                else:
                    pca.append(0.0)
            out[f"{c}+{d}"] = pca
        return out

    def _confusion_matrices(self, results) -> dict:
        out = {}
        for (c, d), r in results.items():
            out[f"{c}+{d}"] = confusion_matrix(
                r["labels"], r["preds"], labels=list(range(NUM_CLASSES))
            ).tolist()
        return out

    def _defense_frequency(self, rl_actions) -> dict:
        out = {}
        for cond, acts in rl_actions.items():
            total = max(len(acts), 1)
            out[cond] = {
                DEFENSE_NAMES[a]: acts.count(a) / total
                for a in range(NUM_ACTIONS)
            }
        return out

    def _statistical_tests(self, results) -> dict:
        out = {}
        for cond in CONDITIONS:
            if cond == "clean":
                continue
            rl_corr = results[(cond, "rl")]["correct"]
            for sd in STATIC_DEFENSES:
                st_corr = results[(cond, sd)]["correct"]
                b = sum(s and not r for s, r in zip(st_corr, rl_corr))
                c = sum(r and not s for s, r in zip(st_corr, rl_corr))
                if b + c > 0:
                    chi2  = (b - c) ** 2 / (b + c)
                    p_val = 1.0 - scipy_stats.chi2.cdf(chi2, df=1)
                else:
                    chi2, p_val = 0.0, 1.0
                out[f"{cond}_rl_vs_{sd}"] = {
                    "mcnemar_chi2":    round(chi2, 4),
                    "p_value":         round(p_val, 6),
                    "significant_005": bool(p_val < 0.05),
                }
        return out

    # ═══════════════════ plotting ══════════════════════════════════
    def _plot_accuracy_heatmap(self, table):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(table, cmap="YlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(ALL_DEFENSES)))
        ax.set_xticklabels(ALL_DEFENSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels(CONDITIONS)
        for i in range(len(CONDITIONS)):
            for j in range(len(ALL_DEFENSES)):
                ax.text(j, i, f"{table[i,j]:.1f}%", ha="center", va="center",
                        fontsize=9,
                        color="white" if table[i, j] < 50 else "black")
        ax.set_title("Accuracy: Condition × Defense")
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "accuracy_heatmap.png"), dpi=150)
        plt.close(fig)

    def _plot_per_attack_bars(self, table):
        x = np.arange(len(CONDITIONS))
        width = 0.12
        fig, ax = plt.subplots(figsize=(12, 6))
        for j, d in enumerate(ALL_DEFENSES):
            offset = (j - len(ALL_DEFENSES) / 2 + 0.5) * width
            ax.bar(x + offset, table[:, j], width, label=d)
        ax.set_xlabel("Input Condition")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-Attack Accuracy by Defense Strategy")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "per_attack_accuracy.png"), dpi=150)
        plt.close(fig)

    def _plot_per_class_accuracy(self, per_class):
        for cond in ["fgsm", "bim", "pgd"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(NUM_CLASSES)
            width = 0.12
            for j, d in enumerate(ALL_DEFENSES):
                key = f"{cond}+{d}"
                vals = per_class.get(key, [0] * NUM_CLASSES)
                offset = (j - len(ALL_DEFENSES) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=d)
            ax.set_xlabel("Class")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Per-Class Accuracy under {cond.upper()} Attack")
            ax.set_xticks(x)
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 105)
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir,
                        f"per_class_accuracy_{cond}.png"), dpi=150)
            plt.close(fig)

    def _plot_confusion_matrices(self, conf_mats):
        key_combos = [(c, d) for c in ["pgd", "fgsm"]
                      for d in ["none", "gaussian", "rl"]]
        ncols, nrows = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        for idx, (c, d) in enumerate(key_combos):
            ax = axes[idx // ncols, idx % ncols]
            cm = np.array(conf_mats[f"{c}+{d}"])
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{c}+{d}", fontsize=9)
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(CLASS_NAMES, fontsize=6)
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=6)
        fig.suptitle("Confusion Matrices (selected)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "confusion_matrices.png"), dpi=150)
        plt.close(fig)

    def _plot_defense_frequency(self, freq):
        attack_conds = [c for c in CONDITIONS if c != "clean"]
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(attack_conds))
        width = 0.08
        for j, dname in enumerate(DEFENSE_NAMES):
            vals = [freq.get(c, {}).get(dname, 0) * 100 for c in attack_conds]
            offset = (j - len(DEFENSE_NAMES) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=dname)
        ax.set_xlabel("Attack Condition")
        ax.set_ylabel("Selection Frequency (%)")
        ax.set_title("RL Agent Defense Selection Frequency by Attack")
        ax.set_xticks(x)
        ax.set_xticklabels(attack_conds)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "defense_frequency.png"), dpi=150)
        plt.close(fig)

    def _plot_perturbation_analysis(self, pdata):
        if not pdata:
            return
        attacks = list(pdata.keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, title in [
            (axes[0], "l2",   "L2 Perturbation Norm"),
            (axes[1], "linf", "L∞ Perturbation Norm"),
        ]:
            data = [pdata[a][metric] for a in attacks]
            ax.boxplot(data, tick_labels=attacks)
            ax.set_title(title)
            ax.set_ylabel("Norm")
        fig.suptitle("Perturbation Analysis per Attack")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "perturbation_analysis.png"), dpi=150)
        plt.close(fig)

    def _plot_attention_maps(self, samples):
        if not samples:
            return
        key_none = ("pgd", "none")
        key_rl   = ("pgd", "rl")
        items_none = samples.get(key_none, [])[:3]
        items_rl   = samples.get(key_rl,   [])[:3]
        n = min(len(items_none), len(items_rl), 3)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        col_titles = ["Clean", "PGD Attacked", "Attn (attacked)",
                      "RL Defended", "Attn (defended)"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=9)

        for i in range(n):
            clean_img = items_none[i]["clean"]
            atk_img   = items_none[i]["attacked"]
            def_img   = items_rl[i]["defended"]
            attn_atk  = self._get_attention_map(atk_img)
            attn_def  = self._get_attention_map(def_img)

            for j, img in enumerate([clean_img, atk_img, attn_atk,
                                      def_img, attn_def]):
                ax = axes[i, j]
                if img.dim() == 3 and img.shape[0] <= 3:
                    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                else:
                    ax.imshow(img.squeeze().numpy(), cmap="jet")
                ax.axis("off")

        fig.suptitle("Attention Map Comparison (PGD)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "attention_maps.png"), dpi=150)
        plt.close(fig)

    def _get_attention_map(self, image: torch.Tensor) -> torch.Tensor:
        """Extract CBAM spatial attention map for a single (C,H,W) image."""
        x = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.classifier.features(x)
            feat = self.classifier.ghost(feat)
            feat = feat * self.classifier.cbam.channel_attention(feat)
            attn = self.classifier.cbam.spatial_attention(feat)  # (1,1,H,W)
        attn = F.interpolate(attn, size=(224, 224), mode="bilinear",
                             align_corners=False)
        return attn.squeeze().cpu()

    def _plot_training_curves(self):
        log_dir = self.tb_log_dir
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(self.out_dir) or ".", "rl_defense_out", "tb")
        if not os.path.isdir(log_dir):
            print(f"  [skip] TensorBoard logs not found at {log_dir}")
            return

        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError:
            print("  [skip] install tensorboard to extract training curves")
            return

        ea = EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])

        for tag, fname, ylabel in [
            ("train/reward_epoch",    "reward_curve.png",   "Mean Reward"),
            ("train/loss_epoch",      "dqn_loss.png",       "DQN Loss"),
            ("train/accuracy_epoch",  "train_accuracy.png", "Accuracy (%)"),
            ("val/accuracy_overall",  "val_accuracy.png",   "Val Accuracy (%)"),
        ]:
            if tag not in tags:
                continue
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals  = [e.value for e in events]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, vals, marker="o", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(tag.replace("/", " / "))
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, fname), dpi=150)
            plt.close(fig)

        if "train/epsilon" in tags:
            events = ea.Scalars("train/epsilon")
            steps = [e.step for e in events]
            vals  = [e.value for e in events]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, vals)
            ax.set_xlabel("Step")
            ax.set_ylabel("Epsilon")
            ax.set_title("Epsilon Decay (cosine)")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "epsilon_decay.png"), dpi=150)
            plt.close(fig)

    # ═══════════════════ pretty-print ════════════════════════════
    def _print_accuracy_table(self, table):
        header = f"{'':10s}" + "".join(f"{d:>10s}" for d in ALL_DEFENSES)
        print(header)
        print("-" * len(header))
        for i, c in enumerate(CONDITIONS):
            row = f"{c:10s}" + "".join(f"{table[i,j]:9.1f}%" for j in range(len(ALL_DEFENSES)))
            print(row)
        print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--classifier_weights", type=str,
                        default="model_weights.pt")
    parser.add_argument("--rl_agent_path", type=str,
                        default="rl_defense_out/rl_agent_final.pth")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument("--out_dir",      type=str, default="eval_results")
    parser.add_argument("--tb_log_dir",   type=str, default=None,
                        help="Path to RL training TensorBoard logs "
                             "(default: rl_defense_out/tb)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_dataset = load_fetal_planes_hf(
        train_ratio=0.70, val_ratio=0.15, seed=42)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = FullEvaluator(
        classifier_weights_path=args.classifier_weights,
        rl_agent_path=args.rl_agent_path,
        test_loader=test_loader,
        device=device,
        out_dir=args.out_dir,
        tb_log_dir=args.tb_log_dir,
    )
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()