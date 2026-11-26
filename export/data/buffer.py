"""Priority replay buffer with NumPy storage and JAX-compatible sampling."""

from typing import Dict, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jax
import gymnasium as gym
from gymnasium import spaces


def _get_flat_obs_dim(observation_space: spaces.Space) -> int:
    """Compute flattened observation dimension for any space type."""
    if isinstance(observation_space, spaces.Dict):
        return sum(_get_flat_obs_dim(v) for v in observation_space.spaces.values())
    elif isinstance(observation_space, spaces.Box):
        return int(np.prod(observation_space.shape))
    elif isinstance(observation_space, spaces.Tuple):
        return sum(_get_flat_obs_dim(s) for s in observation_space.spaces)
    else:
        raise NotImplementedError(f"Unsupported observation space: {type(observation_space)}")


def _flatten_obs(obs, observation_space: spaces.Space) -> np.ndarray:
    """Flatten observation to 1D array, handling dict/tuple/array."""
    if isinstance(observation_space, spaces.Dict):
        # Use sorted keys for consistent ordering
        parts = [_flatten_obs(obs[k], observation_space.spaces[k])
                 for k in sorted(observation_space.spaces.keys())]
        return np.concatenate(parts).astype(np.float32)
    elif isinstance(observation_space, spaces.Box):
        return np.asarray(obs).flatten().astype(np.float32)
    elif isinstance(observation_space, spaces.Tuple):
        parts = [_flatten_obs(o, s) for o, s in zip(obs, observation_space.spaces)]
        return np.concatenate(parts).astype(np.float32)
    else:
        raise NotImplementedError(f"Unsupported observation space: {type(observation_space)}")


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flatten observations to 1D, handling Dict/Tuple/Box spaces."""

    def __init__(self, env):
        super().__init__(env)
        self._orig_observation_space = env.observation_space
        flat_dim = _get_flat_obs_dim(env.observation_space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        return _flatten_obs(obs, self._orig_observation_space)


class NormalizeActionWrapper(gym.ActionWrapper):
    """Normalize action space to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.orig_low = env.action_space.low
        self.orig_high = env.action_space.high
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )

    def action(self, action):
        # Map [-1, 1] to [low, high]
        action = np.clip(action, -1.0, 1.0)
        return (action + 1.0) / 2.0 * (self.orig_high - self.orig_low) + self.orig_low


def wrap_env(env: gym.Env) -> gym.Env:
    """Apply standard wrappers for flat obs and normalized actions."""
    env = FlattenObsWrapper(env)
    if isinstance(env.action_space, spaces.Box):
        env = NormalizeActionWrapper(env)
    return env


class UniformReplayBuffer:
    """Simple replay buffer with uniform sampling (no priority tracking)."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self._ptr = 0
        self._size = 0
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        # Storage
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
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
        action: np.ndarray,
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
        """Uniform sampling - returns batch dict only."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": jnp.array(self.observations[indices]),
            "actions": jnp.array(self.actions[indices]),
            "next_observations": jnp.array(self.next_observations[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "masks": jnp.array(self.masks[indices]),
        }

    def load_from_minari(self, dataset_id: str, download: bool = True, sample_percentage: float = 1.0):
        """Load offline dataset from Minari with optional random sampling.

        Args:
            dataset_id: Minari dataset identifier
            download: Whether to download if not present
            sample_percentage: Percentage (0.0-1.0) of transitions to randomly sample
        """
        import minari
        dataset = minari.load_dataset(dataset_id, download=download)

        # Collect all transitions first if sampling
        transitions = []
        for episode in dataset.iterate_episodes():
            obs = self._flatten_obs(episode.observations)
            for i in range(len(episode.actions)):
                terminated = episode.terminations[i] if hasattr(episode, 'terminations') else False
                truncated = episode.truncations[i] if hasattr(episode, 'truncations') else False
                mask = 0.0 if (terminated or truncated) else 1.0
                transitions.append((
                    obs[i],
                    np.array(episode.actions[i], dtype=np.float32),
                    obs[i + 1],
                    float(episode.rewards[i]),
                    mask
                ))

        # Randomly sample if percentage < 1.0
        if sample_percentage < 1.0:
            num_samples = int(len(transitions) * sample_percentage)
            indices = np.random.choice(len(transitions), size=num_samples, replace=False)
            transitions = [transitions[i] for i in indices]

        # Insert sampled transitions
        for obs, action, next_obs, reward, mask in transitions:
            self.insert(obs, action, next_obs, reward, mask)

    def load_mixed_datasets(self, dataset_specs, download: bool = True):
        """Load and mix multiple datasets with specified percentages.

        Args:
            dataset_specs: List of tuples (dataset_id, percentage)
                          e.g., [('dataset1', 0.5), ('dataset2', 1.0)]
            download: Whether to download if not present
        """
        for dataset_id, percentage in dataset_specs:
            print(f"Loading {dataset_id} with {percentage*100:.1f}% of transitions...")
            self.load_from_minari(dataset_id, download=download, sample_percentage=percentage)

    def _flatten_obs(self, obs) -> np.ndarray:
        """Flatten observations (handles dict/tuple/array)."""
        if isinstance(obs, np.ndarray):
            return obs.reshape(obs.shape[0], -1).astype(np.float32)
        elif isinstance(obs, dict):
            parts = [obs[k].reshape(obs[k].shape[0], -1) for k in sorted(obs.keys())]
            return np.concatenate(parts, axis=1).astype(np.float32)
        elif isinstance(obs, (list, tuple)):
            parts = [np.array(o).reshape(np.array(o).shape[0], -1) for o in obs]
            return np.concatenate(parts, axis=1).astype(np.float32)
        else:
            raise NotImplementedError(f"Unknown obs type: {type(obs)}")


class PriorityReplayBuffer:
    """
    NumPy-based replay buffer with priority sampling.

    Priorities are used directly for sampling (no internal alpha scaling).
    Apply alpha to priority values externally when calling update_priorities.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
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

        # Storage
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)  # 1 - done
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        mask: float
    ):
        """Insert single transition with max priority."""
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
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        mask: float,
        priority: float
    ):
        """Insert single transition with specified priority."""
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.next_observations[self._ptr] = next_obs
        self.rewards[self._ptr] = reward
        self.masks[self._ptr] = mask
        self.priorities[self._ptr] = priority

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        # Update max_priority to actual max of all priorities in buffer
        self.max_priority = self.priorities[:self._size].max()

    def load_from_buffer(self, source_buffer: "UniformReplayBuffer", priority: float = 1.0):
        """Load all data from another buffer with specified priority."""
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
        """
        Sample batch with priority weighting.

        Args:
            batch_size: Number of samples to draw
            step: Current training step (for beta annealing)
            utd_ratio: If provided, normalize weights to sum to utd_ratio instead of max=1.0

        Returns:
            batch: dict with observations, actions, next_observations, rewards, masks
            indices: NumPy array for priority updates
            weights: JAX array of IS weights
        """
        priorities = self.priorities[:self._size]
        probs = priorities / priorities.sum()
        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=True)

        beta = self._get_beta(step)
        weights = (self._size * probs[indices]) ** (-beta)

        if utd_ratio is not None:
            # Normalize weights to sum to utd_ratio
            weights = weights / weights.sum() * utd_ratio
        else:
            # Normalize weights to max=1.0
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
        """Update priorities for sampled transitions."""
        if isinstance(priorities, jax.Array):
            priorities = np.asarray(priorities)
        priorities = np.abs(priorities).flatten() + self.epsilon
        self.priorities[indices] = priorities
        # Set max_priority to actual max of all priorities in buffer
        self.max_priority = self.priorities[:self._size].max()

    def compute_entropy(self) -> float:
        """Compute the entropy of the priority distribution.

        Returns:
            Entropy in nats (natural log units)
        """
        priorities = self.priorities[:self._size]
        probs = priorities / priorities.sum()
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def load_from_minari(self, dataset_id: str, download: bool = True, sample_percentage: float = 1.0):
        """Load offline dataset from Minari with optional random sampling.

        Args:
            dataset_id: Minari dataset identifier
            download: Whether to download if not present
            sample_percentage: Percentage (0.0-1.0) of transitions to randomly sample
        """
        import minari
        dataset = minari.load_dataset(dataset_id, download=download)

        # Collect all transitions first if sampling
        transitions = []
        for episode in dataset.iterate_episodes():
            obs = self._flatten_obs(episode.observations)
            for i in range(len(episode.actions)):
                terminated = episode.terminations[i] if hasattr(episode, 'terminations') else False
                truncated = episode.truncations[i] if hasattr(episode, 'truncations') else False
                mask = 0.0 if (terminated or truncated) else 1.0
                transitions.append((
                    obs[i],
                    np.array(episode.actions[i], dtype=np.float32),
                    obs[i + 1],
                    float(episode.rewards[i]),
                    mask
                ))

        # Randomly sample if percentage < 1.0
        if sample_percentage < 1.0:
            num_samples = int(len(transitions) * sample_percentage)
            indices = np.random.choice(len(transitions), size=num_samples, replace=False)
            transitions = [transitions[i] for i in indices]

        # Insert sampled transitions
        for obs, action, next_obs, reward, mask in transitions:
            self.insert(obs, action, next_obs, reward, mask)

    def load_mixed_datasets(self, dataset_specs, download: bool = True):
        """Load and mix multiple datasets with specified percentages.

        Args:
            dataset_specs: List of tuples (dataset_id, percentage)
                          e.g., [('dataset1', 0.5), ('dataset2', 1.0)]
            download: Whether to download if not present
        """
        for dataset_id, percentage in dataset_specs:
            print(f"Loading {dataset_id} with {percentage*100:.1f}% of transitions...")
            self.load_from_minari(dataset_id, download=download, sample_percentage=percentage)

    def _flatten_obs(self, obs) -> np.ndarray:
        """Flatten observations (handles dict/tuple/array)."""
        if isinstance(obs, np.ndarray):
            return obs.reshape(obs.shape[0], -1).astype(np.float32)
        elif isinstance(obs, dict):
            parts = [obs[k].reshape(obs[k].shape[0], -1) for k in sorted(obs.keys())]
            return np.concatenate(parts, axis=1).astype(np.float32)
        elif isinstance(obs, (list, tuple)):
            parts = [np.array(o).reshape(np.array(o).shape[0], -1) for o in obs]
            return np.concatenate(parts, axis=1).astype(np.float32)
        else:
            raise NotImplementedError(f"Unknown obs type: {type(obs)}")
