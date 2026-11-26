"""SAC agent with discrete actions and CNN encoder for Atari environments.

Key differences from continuous SAC (sac.py):
1. CNN encoder instead of MLP for image observations
2. Categorical policy instead of Tanh-Normal for discrete actions
3. Q-networks output Q-values for ALL actions (batch, num_actions) instead of single Q(s,a)
4. No entropy backup in target Q (discrete SAC typically omits this)
5. Actions are integers, not continuous vectors
"""

from functools import partial
from typing import Optional, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import optax

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

default_init = nn.initializers.xavier_uniform


def subsample_ensemble(key: jax.Array, params: dict,
                       num_sample: Optional[int], num_qs: int) -> dict:
    """Randomly select num_sample Q-networks from ensemble for target computation."""
    if num_sample is None or num_sample == num_qs:
        return params
    indices = jax.random.choice(key, num_qs, shape=(num_sample,), replace=False)
    return jax.tree_util.tree_map(lambda p: p[indices], params)


# =============================================================================
# Network Architectures
# =============================================================================

class CNNEncoder(nn.Module):
    """CNN encoder for visual observations (shared between actor/critic)."""
    features: Sequence[int] = (32, 64, 64)
    kernels: Sequence[int] = (8, 4, 3)
    strides: Sequence[int] = (4, 2, 1)
    latent_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input: (batch, num_stack, height, width) - NCHW format
        # Flax Conv expects NHWC, so transpose
        x = jnp.transpose(x, (0, 2, 3, 1))

        for features, kernel, stride in zip(self.features, self.kernels, self.strides):
            x = nn.Conv(features, (kernel, kernel), (stride, stride), padding="VALID")(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        return x


class CategoricalActor(nn.Module):
    """Categorical actor for discrete actions."""
    num_actions: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Returns logits for categorical distribution."""
        x = features
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=default_init())(x)
            x = nn.relu(x)
        logits = nn.Dense(self.num_actions, kernel_init=default_init())(x)
        return logits


class DiscreteCritic(nn.Module):
    """Q-network for discrete actions - outputs Q(s,a) for ALL actions."""
    num_actions: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Returns Q-values for all actions: (batch, num_actions)."""
        x = features
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        q_values = nn.Dense(self.num_actions, kernel_init=default_init())(x)
        return q_values


class DiscreteCriticEnsemble(nn.Module):
    """Ensemble of discrete Q-networks using vmap."""
    critic_cls: type
    num: int = 10

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.critic_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


class Temperature(nn.Module):
    """Learnable temperature parameter."""
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            lambda _: jnp.array(jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)


# =============================================================================
# SAC Atari Learner
# =============================================================================

class SACAtariLearner(struct.PyTreeNode):
    """SAC agent with discrete actions for Atari."""

    rng: jax.Array
    encoder: TrainState
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState

    tau: float = struct.field(pytree_node=False, default=0.005)
    discount: float = struct.field(pytree_node=False, default=0.99)
    target_entropy: float = struct.field(pytree_node=False, default=-1.0)
    num_qs: int = struct.field(pytree_node=False, default=10)
    num_min_qs: int = struct.field(pytree_node=False, default=2)

    @classmethod
    def create(
        cls,
        seed: int,
        obs_shape: Tuple[int, int, int],
        num_actions: int,
        cnn_features: Sequence[int] = (32, 64, 64),
        cnn_kernels: Sequence[int] = (8, 4, 3),
        cnn_strides: Sequence[int] = (4, 2, 1),
        encoder_dim: int = 512,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 10,
        num_min_qs: int = 2,
        encoder_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
    ):
        """Create SACAtariLearner."""
        rng = jax.random.PRNGKey(seed)
        rng, encoder_key, actor_key, critic_key, temp_key = jax.random.split(rng, 5)

        dummy_obs = jnp.zeros((1, *obs_shape))

        # Encoder
        encoder_def = CNNEncoder(
            features=cnn_features,
            kernels=cnn_kernels,
            strides=cnn_strides,
            latent_dim=encoder_dim
        )
        encoder_params = encoder_def.init(encoder_key, dummy_obs)["params"]
        encoder = TrainState.create(
            apply_fn=encoder_def.apply,
            params=encoder_params,
            tx=optax.adam(encoder_lr)
        )

        # Actor
        dummy_features = jnp.zeros((1, encoder_dim))
        actor_def = CategoricalActor(num_actions=num_actions, hidden_dims=hidden_dims)
        actor_params = actor_def.init(actor_key, dummy_features)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(actor_lr)
        )

        # Critic ensemble
        critic_cls = partial(DiscreteCritic, num_actions=num_actions, hidden_dims=hidden_dims)
        critic_def = DiscreteCriticEnsemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, dummy_features)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(critic_lr)
        )

        # Target critic (use num_min_qs for subsampled ensemble, like sac.py)
        target_critic_def = DiscreteCriticEnsemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # Temperature
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(temp_lr)
        )

        # Target entropy: -0.98 * log(1/num_actions) for discrete
        target_entropy = -0.98 * jnp.log(1.0 / num_actions)

        return cls(
            rng=rng,
            encoder=encoder,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            tau=tau,
            discount=discount,
            target_entropy=target_entropy,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
        )

    def update_critic(
        self,
        batch: Dict[str, jnp.ndarray],
        weights: Optional[jnp.ndarray] = None
    ) -> Tuple["SACAtariLearner", Dict[str, jnp.ndarray]]:
        """Update critic networks."""
        rng, key1, key2 = jax.random.split(self.rng, 3)

        # Encode next observations (stop gradient for target computation)
        next_features = self.encoder.apply_fn(
            {"params": self.encoder.params},
            batch["next_observations"]
        )

        # Get policy for next states
        next_logits = self.actor.apply_fn(
            {"params": self.actor.params},
            next_features
        )
        next_probs = jax.nn.softmax(next_logits)
        next_log_probs = jax.nn.log_softmax(next_logits)

        # Subsample target ensemble (like sac.py)
        target_params = subsample_ensemble(key1, self.target_critic.params,
                                           self.num_min_qs, self.num_qs)

        # Get target Q-values for all actions
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            next_features
        )  # (num_min_qs, batch, num_actions)
        next_q = next_qs.min(axis=0)  # (batch, num_actions)

        # Expected Q under policy (V = sum_a pi(a|s) * Q(s,a))
        temperature = self.temp.apply_fn({"params": self.temp.params})
        next_v = (next_probs * (next_q - temperature * next_log_probs)).sum(axis=-1)

        # Target Q-value
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_v

        def critic_loss_fn(encoder_params, critic_params):
            features = self.encoder.apply_fn(
                {"params": encoder_params},
                batch["observations"]
            )
            qs = self.critic.apply_fn(
                {"params": critic_params},
                features
            )  # (num_qs, batch, num_actions)

            # Get Q-values for taken actions
            batch_size = batch["observations"].shape[0]
            actions = batch["actions"].astype(jnp.int32)
            q_taken = qs[:, jnp.arange(batch_size), actions]  # (num_qs, batch)

            td_error = (q_taken - target_q[None, :]) ** 2

            if weights is None:
                loss = td_error.mean()
            else:
                loss = (weights[None, :] * td_error).mean()

            return loss, {"critic_loss": loss, "q_mean": qs.mean(), "qs": qs}

        grads, info = jax.grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.critic.params
        )

        encoder_grads, critic_grads = grads
        encoder = self.encoder.apply_gradients(grads=encoder_grads)
        critic = self.critic.apply_gradients(grads=critic_grads)

        # Soft update target
        target_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_params)

        return self.replace(
            encoder=encoder,
            critic=critic,
            target_critic=target_critic,
            rng=rng
        ), info

    def update_actor(
        self,
        batch: Dict[str, jnp.ndarray],
        weights: Optional[jnp.ndarray] = None
    ) -> Tuple["SACAtariLearner", Dict[str, jnp.ndarray]]:
        """Update actor network."""
        rng, key = jax.random.split(self.rng)

        def actor_loss_fn(encoder_params, actor_params):
            features = self.encoder.apply_fn(
                {"params": encoder_params},
                batch["observations"]
            )
            logits = self.actor.apply_fn(
                {"params": actor_params},
                features
            )

            probs = jax.nn.softmax(logits)
            log_probs = jax.nn.log_softmax(logits)

            # Get Q-values for all actions
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                features
            )  # (num_qs, batch, num_actions)
            q_mean = qs.mean(axis=0)  # (batch, num_actions)

            # Actor loss: E_a~pi [alpha * log pi(a|s) - Q(s,a)]
            temperature = self.temp.apply_fn({"params": self.temp.params})
            per_action_loss = probs * (temperature * log_probs - q_mean)
            per_sample_loss = per_action_loss.sum(axis=-1)  # (batch,)

            # Entropy for logging
            entropy = -(probs * log_probs).sum(axis=-1)

            # Compute V = Σ_a π(a|s) * [Q(s,a) - α * log π(a|s)]
            # V for each Q in ensemble: (num_qs, batch)
            v = (probs[None, :, :] * (qs - temperature * log_probs[None, :, :])).sum(axis=-1)

            if weights is None:
                loss = per_sample_loss.mean()
                entropy_mean = entropy.mean()
            else:
                loss = (weights * per_sample_loss).mean()
                entropy_mean = (weights * entropy).mean()

            return loss, {"actor_loss": loss, "entropy": entropy_mean, "v": v}

        grads, info = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.actor.params
        )

        encoder_grads, actor_grads = grads
        encoder = self.encoder.apply_gradients(grads=encoder_grads)
        actor = self.actor.apply_gradients(grads=actor_grads)

        return self.replace(encoder=encoder, actor=actor, rng=rng), info

    def update_temperature(
        self,
        entropy: jnp.ndarray
    ) -> Tuple["SACAtariLearner", Dict[str, jnp.ndarray]]:
        """Update temperature parameter."""
        def temp_loss_fn(params):
            temperature = self.temp.apply_fn({"params": params})
            loss = temperature * (entropy - self.target_entropy)
            return loss, {"temperature": temperature, "temp_loss": loss}

        grads, info = jax.grad(temp_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), info

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        utd_ratio: int,
        weights: Optional[jnp.ndarray] = None,
        offline_only_batch: Optional[Dict[str, jnp.ndarray]] = None,
        online_only_batch: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple["SACAtariLearner", Dict[str, jnp.ndarray]]:
        """Full update with UTD ratio."""
        new_agent = self
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0] // utd_ratio

        for i in range(utd_ratio):
            mini_batch = jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * batch_size, batch_size),
                batch
            )
            mini_weights = None
            if weights is not None:
                mini_weights = jax.lax.dynamic_slice_in_dim(weights, i * batch_size, batch_size)

            new_agent, critic_info = new_agent.update_critic(mini_batch, mini_weights)

        # Actor and temperature update on last mini-batch
        new_agent, actor_info = new_agent.update_actor(mini_batch, mini_weights)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        info = {**critic_info, **actor_info, **temp_info}

        rng = new_agent.rng
        temperature = new_agent.temp.apply_fn({"params": new_agent.temp.params})

        # Compute Q-values and V for priority computation if batches provided
        if offline_only_batch is not None:
            rng, key = jax.random.split(rng)
            offline_features = new_agent.encoder.apply_fn(
                {"params": new_agent.encoder.params},
                offline_only_batch["observations"]
            )
            offline_qs = new_agent.critic.apply_fn(
                {"params": new_agent.critic.params},
                offline_features
            )  # (num_qs, batch, num_actions)

            # Compute V = Σ_a π(a|s) * [Q(s,a) - α * log π(a|s)]
            offline_logits = new_agent.actor.apply_fn(
                {"params": new_agent.actor.params},
                offline_features
            )
            offline_probs = jax.nn.softmax(offline_logits)
            offline_log_probs = jax.nn.log_softmax(offline_logits)
            # V for each Q in ensemble: (num_qs, batch)
            offline_v = (offline_probs[None, :, :] * (offline_qs - temperature * offline_log_probs[None, :, :])).sum(axis=-1)

            info["offline_qs"] = offline_qs
            info["offline_v"] = offline_v

        if online_only_batch is not None:
            rng, key = jax.random.split(rng)
            online_features = new_agent.encoder.apply_fn(
                {"params": new_agent.encoder.params},
                online_only_batch["observations"]
            )
            online_qs = new_agent.critic.apply_fn(
                {"params": new_agent.critic.params},
                online_features
            )  # (num_qs, batch, num_actions)

            # Compute V
            online_logits = new_agent.actor.apply_fn(
                {"params": new_agent.actor.params},
                online_features
            )
            online_probs = jax.nn.softmax(online_logits)
            online_log_probs = jax.nn.log_softmax(online_logits)
            online_v = (online_probs[None, :, :] * (online_qs - temperature * online_log_probs[None, :, :])).sum(axis=-1)

            info["online_qs"] = online_qs
            info["online_v"] = online_v

        new_agent = new_agent.replace(rng=rng)
        return new_agent, info

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, "SACAtariLearner"]:
        """Sample actions for environment interaction."""
        rng, key = jax.random.split(self.rng)

        features = self.encoder.apply_fn(
            {"params": self.encoder.params},
            observations
        )
        logits = self.actor.apply_fn(
            {"params": self.actor.params},
            features
        )

        dist = tfd.Categorical(logits=logits)
        actions = dist.sample(seed=key)

        return actions, self.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Deterministic actions for evaluation (argmax)."""
        features = self.encoder.apply_fn(
            {"params": self.encoder.params},
            observations
        )
        logits = self.actor.apply_fn(
            {"params": self.actor.params},
            features
        )
        return jnp.argmax(logits, axis=-1)


# =============================================================================
# Density Network for Atari
# =============================================================================

class DensityMLPHead(nn.Module):
    """Density head that takes encoded features and action."""
    hidden_dims: Sequence[int] = (256, 256)
    num_actions: int = 18

    @nn.compact
    def __call__(self, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute density ratio for state-action pairs."""
        # One-hot encode discrete actions
        action_one_hot = jax.nn.one_hot(actions.astype(jnp.int32), self.num_actions)
        x = jnp.concatenate([features, action_one_hot], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        x = nn.Dense(1)(x)
        return nn.softplus(x).squeeze(-1)


class DensityNetworkAtari(struct.PyTreeNode):
    """Density ratio estimator for Atari (uses CNN encoder)."""

    encoder_state: TrainState
    head_state: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        obs_shape: Tuple[int, int, int],
        num_actions: int,
        cnn_features: Sequence[int] = (32, 64, 64),
        cnn_kernels: Sequence[int] = (8, 4, 3),
        cnn_strides: Sequence[int] = (4, 2, 1),
        encoder_dim: int = 512,
        hidden_dims: Sequence[int] = (256, 256),
        encoder_lr: float = 3e-4,
        head_lr: float = 3e-4,
    ):
        """Create DensityNetworkAtari."""
        rng = jax.random.PRNGKey(seed)
        rng, encoder_key, head_key = jax.random.split(rng, 3)

        dummy_obs = jnp.zeros((1, *obs_shape))

        # Encoder
        encoder_def = CNNEncoder(
            features=cnn_features,
            kernels=cnn_kernels,
            strides=cnn_strides,
            latent_dim=encoder_dim
        )
        encoder_params = encoder_def.init(encoder_key, dummy_obs)["params"]
        encoder_state = TrainState.create(
            apply_fn=encoder_def.apply,
            params=encoder_params,
            tx=optax.adam(encoder_lr)
        )

        # Density head
        dummy_features = jnp.zeros((1, encoder_dim))
        dummy_actions = jnp.zeros((1,), dtype=jnp.int32)

        head_def = DensityMLPHead(hidden_dims=hidden_dims, num_actions=num_actions)
        head_params = head_def.init(head_key, dummy_features, dummy_actions)["params"]
        head_state = TrainState.create(
            apply_fn=head_def.apply,
            params=head_params,
            tx=optax.adam(head_lr)
        )

        return cls(encoder_state=encoder_state, head_state=head_state)

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass: returns density weights."""
        features = self.encoder_state.apply_fn(
            {"params": self.encoder_state.params},
            observations
        )
        return self.head_state.apply_fn(
            {"params": self.head_state.params},
            features,
            actions
        )

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        utd_ratio: int
    ) -> Tuple["DensityNetworkAtari", Dict[str, jnp.ndarray]]:
        """Update density network (matches density.py pattern)."""
        new_density = self
        minibatch_size = jax.tree_util.tree_leaves(batch)[0].shape[0] // utd_ratio

        for i in range(utd_ratio):
            mini_batch = jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * minibatch_size, minibatch_size),
                batch
            )

            # De-interleave: even indices are offline, odd indices are online
            offline_obs = mini_batch["observations"][0::2]
            offline_actions = mini_batch["actions"][0::2]
            online_obs = mini_batch["observations"][1::2]
            online_actions = mini_batch["actions"][1::2]

            new_density, info = new_density._update_step(
                offline_obs, offline_actions, online_obs, online_actions
            )

        return new_density, info

    def _update_step(
        self,
        offline_obs: jnp.ndarray,
        offline_actions: jnp.ndarray,
        online_obs: jnp.ndarray,
        online_actions: jnp.ndarray,
    ) -> Tuple["DensityNetworkAtari", Dict[str, jnp.ndarray]]:
        """Single density update step."""
        def loss_fn(encoder_params, head_params):
            offline_features = self.encoder_state.apply_fn(
                {"params": encoder_params},
                offline_obs
            )
            offline_w = self.head_state.apply_fn(
                {"params": head_params},
                offline_features,
                offline_actions
            )

            online_features = self.encoder_state.apply_fn(
                {"params": encoder_params},
                online_obs
            )
            online_w = self.head_state.apply_fn(
                {"params": head_params},
                online_features,
                online_actions
            )

            # f-divergence loss
            offline_f_star_f_prime = -jnp.log(2.0 / (offline_w + 1) + 1e-10)
            online_f_prime = jnp.log(2 * online_w / (online_w + 1) + 1e-10)

            loss = offline_f_star_f_prime.mean() - online_f_prime.mean()

            return loss, {
                "density_loss": loss,
                "offline_weight": offline_w.mean(),
                "online_weight": online_w.mean(),
                "offline_weights": offline_w,
            }

        grads, info = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder_state.params,
            self.head_state.params
        )

        encoder_grads, head_grads = grads
        encoder_state = self.encoder_state.apply_gradients(grads=encoder_grads)
        head_state = self.head_state.apply_gradients(grads=head_grads)

        return self.replace(encoder_state=encoder_state, head_state=head_state), info
