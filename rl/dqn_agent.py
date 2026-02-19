import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────
class DQNetwork(nn.Module):
    """Simple MLP mapping state -> Q-values for each action."""

    def __init__(self, state_dim: int = 10, num_actions: int = 5,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, state_dim).  Returns: Q-values (B, num_actions)."""
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────
class DQNAgent:
    """DQN agent with target network, epsilon-greedy, and experience replay."""

    def __init__(self,
                 state_dim: int = 10,
                 num_actions: int = 5,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 10_000,
                 buffer_capacity: int = 50_000,
                 batch_size: int = 64,
                 target_update_freq: int = 500,
                 device: str = 'cuda'):

        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.policy_net = DQNetwork(state_dim, num_actions).to(self.device)
        self.target_net = DQNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.steps_done = 0

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray,
                      evaluate: bool = False) -> int:
        """Epsilon-greedy action selection for a single state."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            s = torch.from_numpy(np.asarray(state, dtype=np.float32))
            s = s.unsqueeze(0).to(self.device)
            q_values = self.policy_net(s)
            return int(q_values.argmax(dim=1).item())

    def select_action_batch(self, states: np.ndarray,
                            evaluate: bool = False) -> list:
        """Epsilon-greedy for a batch of states.  Returns list of ints."""
        B = states.shape[0]
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
            greedy = self.policy_net(s).argmax(dim=1).cpu().tolist()

        if evaluate:
            return greedy

        actions = []
        for i in range(B):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.num_actions))
            else:
                actions.append(greedy[i])
        return actions

    # ------------------------------------------------------------------ #
    # Replay storage
    # ------------------------------------------------------------------ #
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #
    # Learning step
    # ------------------------------------------------------------------ #
    def update(self) -> float:
        """One gradient step from a replay-buffer sample.

        Returns:
            Loss value (float), or 0.0 if not enough data yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        s  = torch.from_numpy(states).to(self.device)
        a  = torch.from_numpy(actions).to(self.device).unsqueeze(1)   # (B, 1)
        r  = torch.from_numpy(rewards).to(self.device).unsqueeze(1)   # (B, 1)
        ns = torch.from_numpy(next_states).to(self.device)
        d  = torch.from_numpy(dones.astype(np.float32)).to(self.device).unsqueeze(1)

        # Current Q-values for chosen actions
        q_values = self.policy_net(s).gather(1, a)  # (B, 1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]  # (B, 1)
            target = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Periodic hard update of target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self._update_epsilon()

        return loss.item()

    # ------------------------------------------------------------------ #
    # Epsilon schedule
    # ------------------------------------------------------------------ #
    def _update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.steps_done / self.epsilon_decay_steps,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save({
            'policy_net':  self.policy_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'steps_done':  self.steps_done,
            'epsilon':     self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps_done = ckpt['steps_done']
        self.epsilon    = ckpt['epsilon']
