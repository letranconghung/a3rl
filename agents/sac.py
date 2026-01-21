"""SAC agent with JAX/Flax, supporting weighted losses for priority replay."""

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
tfb = tfp.bijectors

default_init = nn.initializers.xavier_uniform


class MLP(nn.Module):
    """MLP with optional layer norm."""
    hidden_dims: Sequence[int] = (256, 256)
    activate_final: bool = False
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)
        return x


class Critic(nn.Module):
    """Critic network: concat(obs, action) -> MLP -> scalar."""
    hidden_dims: Sequence[int] = (256, 256)
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True,
                use_layer_norm=self.use_layer_norm)(x)
        return nn.Dense(1, kernel_init=default_init())(x).squeeze(-1)


class CriticEnsemble(nn.Module):
    """Vectorized ensemble via nn.vmap."""
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


class TanhNormalActor(nn.Module):
    """Actor network outputting tanh-squashed normal distribution."""
    hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> tfd.Distribution:
        x = MLP(self.hidden_dims, activate_final=True, use_layer_norm=False)(observations)
        mean = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_std = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return tfd.TransformedDistribution(base_dist, tfb.Tanh())


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


def subsample_ensemble(key: jax.Array, params: dict,
                       num_sample: Optional[int], num_qs: int) -> dict:
    """Randomly select num_sample Q-networks from ensemble for target computation."""
    if num_sample is None or num_sample == num_qs:
        return params
    indices = jax.random.choice(key, num_qs, shape=(num_sample,), replace=False)
    return jax.tree_util.tree_map(lambda p: p[indices], params)


class SACLearner(struct.PyTreeNode):
    """SAC Agent supporting weighted losses for priority replay."""

    rng: jax.Array
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState

    tau: float = struct.field(pytree_node=False, default=0.005)
    discount: float = struct.field(pytree_node=False, default=0.99)
    target_entropy: float = struct.field(pytree_node=False, default=-1.0)
    num_qs: int = struct.field(pytree_node=False, default=10)
    num_min_qs: int = struct.field(pytree_node=False, default=2)
    backup_entropy: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(
        cls,
        seed: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 10,
        num_min_qs: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        use_layer_norm: bool = True,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):
        """Factory method to create SACLearner."""
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        # Actor
        actor_def = TanhNormalActor(hidden_dims=tuple(hidden_dims), action_dim=action_dim)
        actor_params = actor_def.init(actor_key, dummy_obs)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(actor_lr)
        )

        # Critic ensemble
        critic_cls = partial(Critic, hidden_dims=tuple(hidden_dims),
                             use_layer_norm=use_layer_norm)
        critic_def = CriticEnsemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, dummy_obs, dummy_action)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(critic_lr)
        )

        # Target critic (no optimizer needed) - use num_min_qs for subsampled ensemble
        target_critic_def = CriticEnsemble(critic_cls, num=num_min_qs or num_qs)
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

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            tau=tau,
            discount=discount,
            target_entropy=-action_dim * 0.5,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    def update_critic(
        self,
        batch: Dict[str, jnp.ndarray],
        weights: Optional[jnp.ndarray] = None
    ) -> Tuple["SACLearner", Dict[str, jnp.ndarray]]:
        """Update critic with optional importance sampling weights."""
        rng, key1, key2 = jax.random.split(self.rng, 3)

        # Compute target Q
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])
        next_actions = dist.sample(seed=key1)
        next_log_probs = dist.log_prob(next_actions)

        # Subsample ensemble for target
        target_params = subsample_ensemble(key2, self.target_critic.params,
                                           self.num_min_qs, self.num_qs)
        next_qs = self.target_critic.apply_fn({"params": target_params},
                                              batch["next_observations"], next_actions)
        next_q = next_qs.min(axis=0)

        temperature = self.temp.apply_fn({"params": self.temp.params})
        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            target_q = target_q - self.discount * batch["masks"] * temperature * next_log_probs

        def critic_loss_fn(params):
            qs = self.critic.apply_fn({"params": params},
                                      batch["observations"], batch["actions"])
            td_error = (qs - target_q) ** 2  # (num_qs, batch)

            if weights is None:
                loss = td_error.mean()
            else:
                # weights: (batch,), td_error: (num_qs, batch)
                # Apply weights to each Q-network's loss, then average across Q-networks
                loss = (weights[None, :] * td_error).mean()

            return loss, {"critic_loss": loss, "q_mean": qs.mean(), "qs": qs}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        # Soft update target
        target_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_actor(
        self,
        batch: Dict[str, jnp.ndarray],
        weights: Optional[jnp.ndarray] = None
    ) -> Tuple["SACLearner", Dict[str, jnp.ndarray]]:
        """Update actor with optional importance sampling weights."""
        rng, key = jax.random.split(self.rng)

        def actor_loss_fn(params):
            dist = self.actor.apply_fn({"params": params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)

            qs = self.critic.apply_fn({"params": self.critic.params},
                                      batch["observations"], actions)
            q_mean = qs.mean(axis=0)  # (batch,)

            temperature = self.temp.apply_fn({"params": self.temp.params})
            per_sample_loss = temperature * log_probs - q_mean

            if weights is None:
                loss = per_sample_loss.mean()
                entropy = -log_probs.mean()
            else:
                loss = (weights * per_sample_loss).mean()
                entropy = -(weights * log_probs).mean()

            return loss, {"actor_loss": loss, "entropy": entropy, "qs_policy": qs}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), info

    def update_temperature(
        self,
        entropy: jnp.ndarray
    ) -> Tuple["SACLearner", Dict[str, jnp.ndarray]]:
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
        offline_only_batch: Optional[Dict[str, jnp.ndarray]] = None
    ) -> Tuple["SACLearner", Dict[str, jnp.ndarray]]:
        """Full update with UTD ratio. Returns Q-values from last minibatch for priority computation.

        Args:
            batch: Main training batch
            utd_ratio: Number of gradient updates per environment step
            weights: Optional importance sampling weights for main batch
            offline_only_batch: Optional offline-only batch for computing normalization statistics
                               (no backpropagation on these samples)
        """
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

        # Actor and temperature update on last mini-batch only
        new_agent, actor_info = new_agent.update_actor(mini_batch, mini_weights)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        info = {**critic_info, **actor_info, **temp_info}

        # If offline_only_batch is provided, compute Q-values for normalization
        if offline_only_batch is not None:
            rng, key = jax.random.split(new_agent.rng)

            # Compute Q-values for offline batch actions (no gradient)
            offline_qs = new_agent.critic.apply_fn(
                {"params": new_agent.critic.params},
                offline_only_batch["observations"],
                offline_only_batch["actions"]
            )

            # Compute Q-values for policy actions on offline batch (no gradient)
            offline_dist = new_agent.actor.apply_fn(
                {"params": new_agent.actor.params},
                offline_only_batch["observations"]
            )
            offline_policy_actions = offline_dist.sample(seed=key)
            offline_qs_policy = new_agent.critic.apply_fn(
                {"params": new_agent.critic.params},
                offline_only_batch["observations"],
                offline_policy_actions
            )

            # Store Q-values in info for computing advantage with same beta
            info["offline_qs"] = offline_qs
            info["offline_qs_policy"] = offline_qs_policy

            new_agent = new_agent.replace(rng=rng)

        return new_agent, info

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, "SACLearner"]:
        """Sample actions for environment interaction."""
        rng, key = jax.random.split(self.rng)
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        actions = dist.sample(seed=key)
        return actions, self.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Deterministic actions for evaluation (mean of base distribution, tanh-squashed)."""
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        # For TanhNormal: get mean of base distribution and apply tanh
        return jnp.tanh(dist.distribution.mean())
