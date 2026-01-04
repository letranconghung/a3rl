import minari
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class FlattenAndNormalizeWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that:
    1. Flattens all observations (handles Dict, Tuple, and nested spaces)
    2. Renormalizes action space to (-1, 1)
    """
    def __init__(self, env):
        super().__init__(env)

        # Flatten observation space
        self.observation_space = self._flatten_observation_space(env.observation_space)

        # Renormalize action space to (-1, 1)
        if isinstance(env.action_space, spaces.Box):
            self.original_action_low = env.action_space.low
            self.original_action_high = env.action_space.high
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=env.action_space.shape,
                dtype=np.float32
            )
        else:
            # For discrete or other action spaces, keep as is
            self.action_space = env.action_space
            self.original_action_low = None
            self.original_action_high = None

    def _flatten_observation_space(self, space):
        """Flatten any observation space to a Box space."""
        if isinstance(space, spaces.Box):
            # Already a box, just flatten it
            flat_dim = int(np.prod(space.shape))
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(flat_dim,),
                dtype=np.float32
            )
        elif isinstance(space, spaces.Dict):
            # Flatten dictionary spaces
            total_dim = 0
            for key in sorted(space.spaces.keys()):
                subspace = space.spaces[key]
                if isinstance(subspace, spaces.Box):
                    total_dim += int(np.prod(subspace.shape))
                else:
                    raise NotImplementedError(f"Dict contains non-Box space: {type(subspace)}")
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        elif isinstance(space, spaces.Tuple):
            # Flatten tuple spaces
            total_dim = 0
            for subspace in space.spaces:
                if isinstance(subspace, spaces.Box):
                    total_dim += int(np.prod(subspace.shape))
                else:
                    raise NotImplementedError(f"Tuple contains non-Box space: {type(subspace)}")
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(space)}")

    def _flatten_observation(self, obs):
        """Flatten observation to 1D array."""
        if isinstance(self.env.observation_space, spaces.Box):
            return obs.flatten().astype(np.float32)
        elif isinstance(self.env.observation_space, spaces.Dict):
            # Concatenate dictionary values in sorted key order
            flat_parts = []
            for key in sorted(obs.keys()):
                flat_parts.append(obs[key].flatten())
            return np.concatenate(flat_parts).astype(np.float32)
        elif isinstance(self.env.observation_space, spaces.Tuple):
            # Concatenate tuple elements
            flat_parts = []
            for item in obs:
                flat_parts.append(item.flatten())
            return np.concatenate(flat_parts).astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported observation type")

    def _denormalize_action(self, action):
        """Convert action from (-1, 1) to original action space range."""
        if self.original_action_low is None:
            return action

        # Map from (-1, 1) to (low, high)
        # action is in [-1, 1], we want to map to [low, high]
        action = np.clip(action, -1.0, 1.0)
        original_action = (action + 1.0) / 2.0 * (self.original_action_high - self.original_action_low) + self.original_action_low
        return original_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_observation(obs), info

    def step(self, action):
        # Denormalize action before passing to environment
        original_action = self._denormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(original_action)
        return self._flatten_observation(obs), reward, terminated, truncated, info

class ReplayBuffer:
    """
    Replay buffer with optional prioritized experience replay (PER) support.

    When alpha=0 (default), sampling is uniform (standard replay buffer).
    When alpha>0, sampling is weighted by priorities with importance sampling correction.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        capacity: Maximum buffer size
        device: Device for tensor outputs
        alpha: Priority exponent (0=uniform, 1=full priority). Default: 0
        beta_start: Initial importance sampling exponent. Default: 0.4
        beta_frames: Steps to anneal beta from beta_start to 1.0. Default: 100000
        epsilon: Small constant added to priorities to avoid zero. Default: 1e-6
    """

    def __init__(self, obs_dim, action_dim, capacity, device='cuda',
                 alpha=0.0, beta_start=0.4, beta_frames=100000, epsilon=1e-6):
        self.device = device
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._insert_index = 0
        self.size = 0

        # Priority replay parameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.max_priority = 1.0  # Track max priority for new samples

        # Data storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Priority storage (initialized to 1.0 for uniform initial sampling)
        self.priorities = np.ones(capacity, dtype=np.float32)
    
    def _flatten_obs_batch(self, observations):
        """
        Flatten a batch of observations to 2D array (batch_size, obs_dim).
        Handles Box, Dict, and Tuple observation types analogous to FlattenAndNormalizeWrapper.
        """
        if isinstance(observations, np.ndarray):
            # Box observation: just reshape to (batch, flat_dim)
            batch_size = observations.shape[0]
            return observations.reshape(batch_size, -1).astype(np.float32)
        elif isinstance(observations, dict):
            # Dict observation: concatenate values in sorted key order
            flat_parts = []
            for key in sorted(observations.keys()):
                arr = np.array(observations[key], dtype=np.float32)
                batch_size = arr.shape[0]
                flat_parts.append(arr.reshape(batch_size, -1))
            return np.concatenate(flat_parts, axis=1).astype(np.float32)
        elif isinstance(observations, (list, tuple)):
            # Tuple observation: concatenate elements
            flat_parts = []
            for item in observations:
                arr = np.array(item, dtype=np.float32)
                batch_size = arr.shape[0]
                flat_parts.append(arr.reshape(batch_size, -1))
            return np.concatenate(flat_parts, axis=1).astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported observation type: {type(observations)}")

    def load_from_minari(self, dataset_id, download):
        """
        Append minari dataset to replay buffer.
        Handles Box, Dict, and Tuple observations by flattening them.
        """
        minari_dataset = minari.load_dataset(dataset_id, download)
        for episode in minari_dataset.iterate_episodes():
            # Flatten observations (handles dict/tuple/box)
            all_obs = self._flatten_obs_batch(episode.observations)
            obs = all_obs[:-1]
            next_obs = all_obs[1:]

            self.add_episode({
                'obs': obs,
                'actions': np.array(episode.actions, dtype=np.float32),
                'next_obs': next_obs,
                'rewards': np.array(episode.rewards, dtype=np.float32),
                'dones': np.logical_or(episode.terminations, episode.truncations).astype(np.float32)
            })
    
    def add(self, obs, action, next_obs, reward, done):
        """
        Add a singular transition.
        New samples receive max_priority to ensure they are sampled at least once.
        """
        self.obs[self._insert_index] = obs
        self.actions[self._insert_index] = action
        self.next_obs[self._insert_index] = next_obs
        self.rewards[self._insert_index] = reward
        self.dones[self._insert_index] = done
        self.priorities[self._insert_index] = self.max_priority  # New samples get max priority
        self._insert_index = (self._insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, episode):
        """
            Add episode of dictionary: obs, next_obs, rewards, actions, dones, which are all np arrays.
        """
        episode_length = episode['obs'].shape[0]
        assert episode['obs'].shape == (episode_length, self.obs_dim)
        assert episode['next_obs'].shape == (episode_length, self.obs_dim)
        assert episode['rewards'].shape == (episode_length,)
        assert episode['actions'].shape == (episode_length, self.action_dim)
        assert episode['dones'].shape == (episode_length,)
        
        for idx in range(episode_length):
            self.add(episode['obs'][idx], episode['actions'][idx], episode['next_obs'][idx], episode['rewards'][idx], episode['dones'][idx])
        
    def _get_beta(self, step):
        """
        Anneal beta from beta_start to 1.0 over beta_frames steps.

        Args:
            step: Current training step (None returns 1.0)

        Returns:
            Current beta value for importance sampling correction
        """
        if step is None:
            return 1.0
        fraction = min(step / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def sample(self, batch_size, device='cuda', step=None):
        """
        Sample a batch from the replay buffer.

        When alpha=0: Uniform random sampling (backward compatible).
        When alpha>0: Priority-weighted sampling with importance sampling weights.

        Args:
            batch_size: Number of samples to draw
            device: Device for tensor outputs
            step: Current training step for beta annealing (only used if alpha>0)

        Returns:
            If alpha=0: (batch_dict, indices, None)
            If alpha>0: (batch_dict, indices, importance_weights)

            batch_dict contains: obs, actions, next_obs, rewards, dones
            indices: numpy array of sampled indices (for priority updates)
            importance_weights: tensor of IS weights or None
        """
        if self.alpha == 0:
            # Uniform sampling (original behavior)
            indices = np.random.randint(0, self.size, size=batch_size)
            weights = None
        else:
            # Priority-weighted sampling
            priorities = self.priorities[:self.size] ** self.alpha
            probs = priorities / priorities.sum()

            # Sample with replacement
            indices = np.random.choice(self.size, size=batch_size, p=probs, replace=True)

            # Compute importance sampling weights
            beta = self._get_beta(step)
            # w_i = (N * P(i))^(-beta) / max(w)
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / weights.max()  # Normalize by max for stability
            weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        batch = {
            'obs': torch.as_tensor(self.obs[indices], device=device),
            'actions': torch.as_tensor(self.actions[indices], device=device),
            'next_obs': torch.as_tensor(self.next_obs[indices], device=device),
            'rewards': torch.as_tensor(self.rewards[indices], device=device),
            'dones': torch.as_tensor(self.dones[indices], device=device)
        }

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.

        Args:
            indices: numpy array of indices to update
            priorities: numpy array or tensor of new priority values (e.g., density weights)
        """
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        # Ensure priorities are positive
        priorities = np.abs(priorities).flatten() + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {
            'obs': self.obs,
            'next_obs': self.next_obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'priorities': self.priorities,
            'max_priority': self.max_priority,
            'size': self.size,
            '_insert_index': self._insert_index
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.obs = state_dict['obs']
        self.next_obs = state_dict['next_obs']
        self.actions = state_dict['actions']
        self.rewards = state_dict['rewards']
        self.dones = state_dict['dones']
        self.size = state_dict['size']
        self._insert_index = state_dict['_insert_index']
        self.priorities = state_dict['priorities']
        self.max_priority = state_dict.get('max_priority', 1.0)