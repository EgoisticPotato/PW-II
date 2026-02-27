"""rl/dqn_agent.py
DQN agent for per-image adversarial defense selection.

Improvements over v1:
  • Dueling DQN architecture  (V(s) + A(s,a) decomposition)
  • Prioritized Experience Replay (PER) with importance-sampling correction
  • Double DQN target computation (decouples action selection from evaluation)
  • Cosine-annealed epsilon decay (faster initial exploration, slower tail)
  • Gradient clipping for stable training
  • Separate optimiser param groups (supports per-layer LR in future)
"""

import math
import os
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Prioritized Replay Buffer
# ─────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """Proportional prioritized replay (Schaul et al., 2016).

    Priorities stored as raw TD-error magnitudes + small ε for stability.
    Sampling is O(N) using numpy; suitable for buffers up to ~100 k.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self._max_priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        N = len(self.buffer)
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs, replace=False
                                   if N >= batch_size else True)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors: np.ndarray, eps: float = 1e-6):
        for i, td in zip(indices, td_errors):
            p = abs(float(td)) + eps
            self.priorities[i] = p
            if p > self._max_priority:
                self._max_priority = p

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────
# Dueling DQN Network
# ─────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """Dueling DQN with shared feature extractor and separate V / A heads.

    Architecture:
        input → [FC(256) → LayerNorm → SiLU] × 2
              → value stream  :  FC(128) → FC(1)
              → advantage stream: FC(128) → FC(num_actions)
        Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
    """

    def __init__(self, state_dim: int, num_actions: int,
                 hidden: int = 256, hidden2: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, 1),
        )
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V = self.value_head(feat)                      # (B, 1)
        A = self.advantage_head(feat)                  # (B, num_actions)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# ─────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────
class DQNAgent:
    """Double Dueling DQN with PER and cosine epsilon schedule.

    Key differences from v1:
      • Uses DuelingDQN instead of a plain MLP.
      • Uses PrioritizedReplayBuffer; TD errors fed back after each update.
      • Double DQN: online net selects action, target net evaluates it.
      • Epsilon follows a cosine annealing schedule.
      • Gradient clipping (max_norm=10) for training stability.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 15_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        target_update_freq: int = 300,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        device: str = "cuda",
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end

        self._step = 0   # global update counter

        # Networks
        self.online_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=lr, eps=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epsilon_decay_steps, eta_min=lr * 0.01
        )

        self.buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)

    # ── Epsilon (cosine schedule) ─────────────────────────────
    @property
    def epsilon(self) -> float:
        t = min(self._step, self.epsilon_decay_steps)
        cos_val = math.cos(math.pi * t / self.epsilon_decay_steps)
        return self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (1 + cos_val)

    # ── Beta for PER IS correction (anneal from start → end) ─
    @property
    def _per_beta(self) -> float:
        t = min(self._step, self.epsilon_decay_steps)
        frac = t / self.epsilon_decay_steps
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    # ── Action selection ──────────────────────────────────────
    def select_action(self, state: np.ndarray,
                      evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(s)
        return int(q.argmax(dim=1).item())

    def select_action_batch(self, states: np.ndarray,
                            evaluate: bool = False) -> List[int]:
        """Vectorised action selection for a batch of states."""
        B = states.shape[0]
        if not evaluate and self.epsilon > 0:
            rand_mask = np.random.random(B) < self.epsilon
            rand_actions = np.random.randint(0, self.num_actions, B)
        else:
            rand_mask = np.zeros(B, dtype=bool)
            rand_actions = np.zeros(B, dtype=int)

        s_t = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(s_t)               # (B, A)
        greedy = q_vals.argmax(dim=1).cpu().numpy()

        actions = np.where(rand_mask, rand_actions, greedy)
        return actions.tolist()

    # ── Replay buffer interface ───────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────
    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        (states, actions, rewards, next_states, dones,
         indices, is_weights) = self.buffer.sample(self.batch_size, self._per_beta)

        s  = torch.from_numpy(states).float().to(self.device)
        a  = torch.from_numpy(actions).long().to(self.device)
        r  = torch.from_numpy(rewards).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        d  = torch.from_numpy(dones).float().to(self.device)
        w  = torch.from_numpy(is_weights).float().to(self.device)

        # ── Double DQN target ─────────────────────────────────
        with torch.no_grad():
            # online net selects best action in next state
            best_next_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
            # target net evaluates that action
            next_q = self.target_net(ns).gather(1, best_next_actions).squeeze(1)
            targets = r + self.gamma * next_q * (1.0 - d)

        # ── Current Q values ──────────────────────────────────
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ── Weighted Huber (Smooth L1) loss ───────────────────
        td_errors = (targets - current_q).detach().cpu().numpy()
        losses = F.smooth_l1_loss(current_q, targets, reduction='none')
        loss = (losses * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors)

        # Hard update target network
        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "step":               self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._step = ckpt.get("step", 0)
        self.target_net.eval()