import random
from collections import deque, namedtuple

import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """Fixed-size circular experience-replay buffer for DQN training.

    Transitions are stored as lightweight numpy arrays to keep GPU memory
    free.  The buffer automatically evicts the oldest entries once full.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    # ------------------------------------------------------------------ #
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store one (s, a, r, s', done) transition."""
        self.buffer.append(Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int):
        """Sample a random minibatch.

        Returns:
            states      : np.ndarray  (batch_size, state_dim)
            actions     : np.ndarray  (batch_size,)  int64
            rewards     : np.ndarray  (batch_size,)  float32
            next_states : np.ndarray  (batch_size, state_dim)
            dones       : np.ndarray  (batch_size,)  bool
        """
        batch = random.sample(self.buffer, batch_size)
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.bool_)
        return states, actions, rewards, next_states, dones

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.buffer)
