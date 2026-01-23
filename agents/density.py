"""Density network for offline/online ratio estimation in JAX/Flax."""

from functools import partial
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import optax


class DensityMLP(nn.Module):
    """MLP with Softplus output for density ratio estimation."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return nn.softplus(x)  # Positive output


class DensityNetwork(struct.PyTreeNode):
    """Density ratio estimator using f-divergence."""

    state: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 3e-4
    ):
        rng = jax.random.PRNGKey(seed)

        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        model = DensityMLP(hidden_dims=tuple(hidden_dims))
        params = model.init(rng, dummy_obs, dummy_action)["params"]

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(lr)
        )

        return cls(state=state)

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass: returns density weights."""
        return self.state.apply_fn({"params": self.state.params}, observations, actions)

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        utd_ratio: int
    ) -> Tuple["DensityNetwork", Dict[str, jnp.ndarray]]:
        """
        Update using f-divergence density ratio estimation.

        Matches SAC UTD ratio pattern: accepts interleaved offline/online batch,
        splits into minibatches, and de-interleaves within each UTD step.

        Loss: E_offline[-log(2/(w+1))] - E_online[log(2w/(w+1))]

        Args:
            batch: Combined batch with interleaved offline/online samples.
                   Shape: [batch_size * utd_ratio * 2, ...] where samples alternate
                   offline, online, offline, online, ...
            utd_ratio: Number of gradient updates to perform

        Returns:
            Updated density network and info dict
        """
        new_density = self
        # Total samples = batch_size * utd_ratio * 2 (half offline, half online, interleaved)
        # Each minibatch = batch_size * 2, then de-interleave to get batch_size each
        minibatch_size = jax.tree_util.tree_leaves(batch)[0].shape[0] // utd_ratio

        for i in range(utd_ratio):
            # Slice minibatch (contains interleaved offline/online)
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
    ) -> Tuple["DensityNetwork", Dict[str, jnp.ndarray]]:
        """Single density update step."""
        def loss_fn(params):
            offline_w = self.state.apply_fn({"params": params}, offline_obs, offline_actions)
            online_w = self.state.apply_fn({"params": params}, online_obs, online_actions)

            offline_f_star_f_prime = -jnp.log(2.0 / (offline_w + 1) + 1e-10)
            online_f_prime = jnp.log(2 * online_w / (online_w + 1) + 1e-10)

            loss = offline_f_star_f_prime.mean() - online_f_prime.mean()

            return loss, {
                "density_loss": loss,
                "offline_weight": offline_w.mean(),
                "online_weight": online_w.mean(),
                "offline_weights": offline_w,  # For denom computation
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        return self.replace(state=new_state), info
