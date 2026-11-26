"""Atari-specific replay buffers with frame stacking and image preprocessing.

Key differences from buffer.py:
1. Observations are images (num_stack, height, width) instead of flat vectors
2. Actions are discrete integers instead of continuous vectors
3. Frame stacking wrapper for environment
4. Image preprocessing (grayscale, resize)
"""

from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np
import jax.numpy as jnp
import jax
import gymnasium as gym
from gymnasium import spaces
import cv2


# =============================================================================
# Preprocessing and Wrappers
# =============================================================================

def preprocess_frame(frame: np.ndarray, target_size: int = 84) -> np.ndarray:
    """Convert RGB Atari frame to grayscale and resize.

    Args:
        frame: RGB image (210, 160, 3) uint8
        target_size: Target image size (default 84)

    Returns:
        Grayscale image (target_size, target_size) float32 normalized to [0, 1]
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


class FrameStackWrapper(gym.Wrapper):
    """Stack the last k frames for Atari observations."""

    def __init__(self, env: gym.Env, num_stack: int = 4, target_size: int = 84):
        super().__init__(env)
        self.num_stack = num_stack
        self.target_size = target_size
        self.frames = deque(maxlen=num_stack)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(num_stack, target_size, target_size),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = preprocess_frame(obs, target_size=self.target_size)

        for _ in range(self.num_stack):
            self.frames.append(obs)

        return np.stack(self.frames, axis=0).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = preprocess_frame(obs, target_size=self.target_size)
        self.frames.append(obs)

        stacked = np.stack(self.frames, axis=0).astype(np.float32)
        return stacked, reward, terminated, truncated, info


def wrap_env_atari(env: gym.Env, num_stack: int = 4, target_size: int = 84) -> gym.Env:
    """Apply standard Atari wrappers."""
    return FrameStackWrapper(env, num_stack=num_stack, target_size=target_size)


# =============================================================================
# Uniform Replay Buffer
# =============================================================================

class UniformReplayBufferAtari:
    """Simple Atari replay buffer with uniform sampling."""

    def __init__(self, obs_shape: Tuple[int, int, int], capacity: int):
        """Initialize buffer.

        Args:
            obs_shape: (num_stack, height, width) e.g., (4, 84, 84)
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self._ptr = 0
        self._size = 0
        self._obs_shape = obs_shape

        # Storage - actions are integers for discrete
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        mask: float
    ):
        """Insert single transition."""
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.next_observations[self._ptr] = next_obs
        self.rewards[self._ptr] = reward
        self.masks[self._ptr] = mask

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Uniform sampling."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": jnp.array(self.observations[indices]),
            "actions": jnp.array(self.actions[indices]),
            "next_observations": jnp.array(self.next_observations[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "masks": jnp.array(self.masks[indices]),
        }

    def load_from_minari(self, dataset_id: str, download: bool = True, sample_percentage: float = 1.0):
        """Load offline dataset from Minari."""
        import minari
        dataset = minari.load_dataset(dataset_id, download=download)

        transitions = []
        for episode in dataset.iterate_episodes():
            # Preprocess all frames
            episode_frames = [preprocess_frame(obs) for obs in episode.observations]

            for i in range(len(episode.actions)):
                # Need i+5 frames: obs uses [i:i+4], next_obs uses [i+1:i+5]
                if i + 5 > len(episode_frames):
                    break

                obs_stack = np.stack(episode_frames[i:i+4], axis=0)
                next_obs_stack = np.stack(episode_frames[i+1:i+5], axis=0)

                terminated = episode.terminations[i] if hasattr(episode, 'terminations') else False
                truncated = episode.truncations[i] if hasattr(episode, 'truncations') else False
                mask = 0.0 if (terminated or truncated) else 1.0

                transitions.append((
                    obs_stack,
                    int(episode.actions[i]),
                    next_obs_stack,
                    float(episode.rewards[i]),
                    mask
                ))

        # Random sampling
        if sample_percentage < 1.0:
            num_samples = int(len(transitions) * sample_percentage)
            indices = np.random.choice(len(transitions), size=num_samples, replace=False)
            transitions = [transitions[i] for i in indices]

        for obs, action, next_obs, reward, mask in transitions:
            self.insert(obs, action, next_obs, reward, mask)

    def load_mixed_datasets(self, dataset_specs, download: bool = True):
        """Load and mix multiple datasets."""
        for dataset_id, percentage in dataset_specs:
            print(f"Loading {dataset_id} with {percentage*100:.1f}% of transitions...")
            self.load_from_minari(dataset_id, download=download, sample_percentage=percentage)


# =============================================================================
# Priority Replay Buffer
# =============================================================================

class PriorityReplayBufferAtari:
    """Priority replay buffer for Atari."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        capacity: int,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self._ptr = 0
        self._size = 0
        self.max_priority = 1.0
        self._obs_shape = obs_shape

        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        mask: float
    ):
        """Insert with max priority."""
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.next_observations[self._ptr] = next_obs
        self.rewards[self._ptr] = reward
        self.masks[self._ptr] = mask
        self.priorities[self._ptr] = self.max_priority

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def insert_with_priority(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        mask: float,
        priority: float
    ):
        """Insert with specified priority."""
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.next_observations[self._ptr] = next_obs
        self.rewards[self._ptr] = reward
        self.masks[self._ptr] = mask
        self.priorities[self._ptr] = priority

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        self.max_priority = self.priorities[:self._size].max()

    def load_from_buffer(self, source_buffer: UniformReplayBufferAtari, priority: float = 1.0):
        """Load from another buffer with specified priority."""
        for i in range(source_buffer.size):
            self.insert_with_priority(
                source_buffer.observations[i],
                source_buffer.actions[i],
                source_buffer.next_observations[i],
                source_buffer.rewards[i],
                source_buffer.masks[i],
                priority
            )

    def _get_beta(self, step: Optional[int]) -> float:
        """Anneal beta from beta_start to 1.0."""
        if step is None:
            return 1.0
        frac = min(step / self.beta_frames, 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def sample(
        self,
        batch_size: int,
        step: Optional[int] = None,
        utd_ratio: Optional[int] = None
    ) -> Tuple[Dict[str, jnp.ndarray], np.ndarray, jnp.ndarray]:
        """Sample with priority weighting."""
        priorities = self.priorities[:self._size]
        probs = priorities / priorities.sum()
        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=True)

        beta = self._get_beta(step)
        weights = (self._size * probs[indices]) ** (-beta)

        if utd_ratio is not None:
            weights = weights / weights.sum() * utd_ratio
        else:
            weights = weights / weights.max()

        weights = jnp.array(weights, dtype=jnp.float32)

        batch = {
            "observations": jnp.array(self.observations[indices]),
            "actions": jnp.array(self.actions[indices]),
            "next_observations": jnp.array(self.next_observations[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "masks": jnp.array(self.masks[indices]),
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities."""
        if isinstance(priorities, jax.Array):
            priorities = np.asarray(priorities)
        priorities = np.abs(priorities).flatten() + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = self.priorities[:self._size].max()

    def compute_entropy(self) -> float:
        """Compute entropy of priority distribution."""
        priorities = self.priorities[:self._size]
        probs = priorities / priorities.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def load_from_minari(self, dataset_id: str, download: bool = True, sample_percentage: float = 1.0):
        """Load offline dataset from Minari."""
        import minari
        dataset = minari.load_dataset(dataset_id, download=download)

        transitions = []
        for episode in dataset.iterate_episodes():
            episode_frames = [preprocess_frame(obs) for obs in episode.observations]

            for i in range(len(episode.actions)):
                if i + 5 > len(episode_frames):
                    break

                obs_stack = np.stack(episode_frames[i:i+4], axis=0)
                next_obs_stack = np.stack(episode_frames[i+1:i+5], axis=0)

                terminated = episode.terminations[i] if hasattr(episode, 'terminations') else False
                truncated = episode.truncations[i] if hasattr(episode, 'truncations') else False
                mask = 0.0 if (terminated or truncated) else 1.0

                transitions.append((
                    obs_stack,
                    int(episode.actions[i]),
                    next_obs_stack,
                    float(episode.rewards[i]),
                    mask
                ))

        if sample_percentage < 1.0:
            num_samples = int(len(transitions) * sample_percentage)
            indices = np.random.choice(len(transitions), size=num_samples, replace=False)
            transitions = [transitions[i] for i in indices]

        for obs, action, next_obs, reward, mask in transitions:
            self.insert(obs, action, next_obs, reward, mask)

    def load_mixed_datasets(self, dataset_specs, download: bool = True):
        """Load and mix multiple datasets."""
        for dataset_id, percentage in dataset_specs:
            print(f"Loading {dataset_id} with {percentage*100:.1f}% of transitions...")
            self.load_from_minari(dataset_id, download=download, sample_percentage=percentage)
