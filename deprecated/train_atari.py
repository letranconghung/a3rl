"""Training script for SAC-discrete with Atari environments.

Key differences from train.py:
1. Uses CNN encoder instead of MLP
2. Discrete actions (integers) instead of continuous
3. Frame stacking and image preprocessing
4. Different advantage computation for discrete Q-values
"""

import argparse
import numpy as np
import jax.numpy as jnp
import wandb
import minari
from tqdm import trange
from pathlib import Path
from datetime import datetime

from agents.sac_atari import SACAtariLearner, DensityNetworkAtari
from data.buffer_atari import UniformReplayBufferAtari, PriorityReplayBufferAtari, wrap_env_atari


# =============================================================================
# Utility Functions
# =============================================================================

def parse_dataset_spec(spec_str):
    """Parse dataset specification string into (dataset_id, percentage) tuples."""
    if ',' in spec_str or ':' in spec_str:
        specs = []
        for part in spec_str.split(','):
            if ':' in part:
                dataset_id, pct = part.rsplit(':', 1)
                specs.append((dataset_id.strip(), float(pct)))
            else:
                specs.append((part.strip(), 1.0))
        return specs
    else:
        return [(spec_str, 1.0)]


def combine_batches(offline_batch, online_batch):
    """Interleave offline and online batches for balanced minibatches."""
    combined = {}
    for k in offline_batch:
        off = offline_batch[k]
        on = online_batch[k]
        # Interleave: [off[0], on[0], off[1], on[1], ...]
        interleaved = jnp.stack([off, on], axis=1).reshape(-1, *off.shape[1:])
        combined[k] = interleaved
    return combined


def compute_a3rl_priorities_separate(
    offline_qs_from_batch: jnp.ndarray,
    offline_v_from_batch: jnp.ndarray,
    online_qs_from_batch: jnp.ndarray,
    online_v_from_batch: jnp.ndarray,
    priority_offline_qs: jnp.ndarray,
    priority_offline_v: jnp.ndarray,
    priority_online_qs: jnp.ndarray,
    priority_online_v: jnp.ndarray,
    priority_offline_actions: jnp.ndarray,
    priority_online_actions: jnp.ndarray,
    offline_density: jnp.ndarray,
    advantage_lambda: float,
    advantage_beta: float,
    priority_alpha: float,
):
    """Compute A3RL priorities for discrete actions.

    For discrete SAC:
    - Q-values: (num_qs, batch, num_actions)
    - V-values: (num_qs, batch) computed as Σ_a π(a|s) * [Q(s,a) - α*log π(a|s)]
    - Advantage: A(s,a) = Q(s,a) - V(s)
    """
    # Extract Q-values for taken actions from normalization batches
    # offline_qs_from_batch: (num_qs, batch, num_actions)
    # We need Q(s, a_taken) for advantage, but for normalization we use mean advantage
    # Simpler: use mean Q - V as the advantage statistic

    # Compute advantage statistics for offline batch (for normalization)
    # A = mean_a[Q(s,a)] - V(s) gives a sense of how "good" the state is
    offline_mean_q = offline_qs_from_batch.mean(axis=2)  # (num_qs, batch)
    offline_advantages = offline_mean_q - offline_v_from_batch  # (num_qs, batch)
    offline_adv_mean = offline_advantages.mean(axis=0)  # (batch,)
    offline_adv_std = offline_advantages.std(axis=0)
    offline_advantage_stats = offline_adv_mean + advantage_beta * offline_adv_std

    # Online batch
    online_mean_q = online_qs_from_batch.mean(axis=2)
    online_advantages = online_mean_q - online_v_from_batch
    online_adv_mean = online_advantages.mean(axis=0)
    online_adv_std = online_advantages.std(axis=0)
    online_advantage_stats = online_adv_mean + advantage_beta * online_adv_std

    # Normalization statistics
    offline_advantage_mean = offline_advantage_stats.mean()
    offline_advantage_std = offline_advantage_stats.std()
    online_advantage_mean = online_advantage_stats.mean()
    online_advantage_std = online_advantage_stats.std()

    # Priority buffer samples - compute A(s, a_taken) = Q(s, a_taken) - V(s)
    batch_size_offline = priority_offline_actions.shape[0]
    batch_size_online = priority_online_actions.shape[0]

    # Get Q(s, a_taken) for priority samples
    priority_offline_q_taken = priority_offline_qs[:, jnp.arange(batch_size_offline), priority_offline_actions.astype(jnp.int32)]  # (num_qs, batch)
    priority_offline_advantages = priority_offline_q_taken - priority_offline_v  # (num_qs, batch)
    priority_offline_adv_mean = priority_offline_advantages.mean(axis=0)  # (batch,)
    priority_offline_adv_std = priority_offline_advantages.std(axis=0)
    priority_offline_advantage = priority_offline_adv_mean + advantage_beta * priority_offline_adv_std

    priority_online_q_taken = priority_online_qs[:, jnp.arange(batch_size_online), priority_online_actions.astype(jnp.int32)]
    priority_online_advantages = priority_online_q_taken - priority_online_v
    priority_online_adv_mean = priority_online_advantages.mean(axis=0)
    priority_online_adv_std = priority_online_advantages.std(axis=0)
    priority_online_advantage = priority_online_adv_mean + advantage_beta * priority_online_adv_std

    # Normalize with z-score
    priority_offline_advantage_zscore = (priority_offline_advantage - offline_advantage_mean) / jnp.maximum(offline_advantage_std, 1e-8)
    priority_online_advantage_zscore = (priority_online_advantage - online_advantage_mean) / jnp.maximum(online_advantage_std, 1e-8)

    # Clip z-scores
    priority_offline_advantage_zscore = jnp.clip(priority_offline_advantage_zscore, -3.0, 3.0)
    priority_online_advantage_zscore = jnp.clip(priority_online_advantage_zscore, -3.0, 3.0)

    # Compute denom from offline density
    density_raised = offline_density ** priority_alpha
    denom = density_raised.sum()

    # Offline priority: (density^alpha / denom) * exp(lambda * adv_zscore)
    offline_advantage_scaled = advantage_lambda * priority_offline_advantage_zscore
    offline_density_term = density_raised / jnp.maximum(denom, 1e-8)
    offline_advantage_term = jnp.exp(offline_advantage_scaled)
    offline_priorities = offline_density_term * offline_advantage_term

    # Online priority: exp(lambda * adv_zscore)
    online_advantage_scaled = advantage_lambda * priority_online_advantage_zscore
    online_advantage_term = jnp.exp(online_advantage_scaled)
    online_priorities = online_advantage_term

    # Clean up NaN/inf
    offline_priorities = jnp.nan_to_num(offline_priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    offline_priorities = jnp.clip(offline_priorities, 1e-5, 1e5)
    online_priorities = jnp.nan_to_num(online_priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    online_priorities = jnp.clip(online_priorities, 1e-5, 1e5)

    metrics = {
        "offline_advantage_stats_mean": offline_advantage_stats.mean(),
        "offline_advantage_stats_std": offline_advantage_stats.std(),
        "online_advantage_stats_mean": online_advantage_stats.mean(),
        "online_advantage_stats_std": online_advantage_stats.std(),
        "priority_offline_advantage_mean": priority_offline_advantage.mean(),
        "priority_online_advantage_mean": priority_online_advantage.mean(),
        "priority_offline_advantage_zscore_mean": priority_offline_advantage_zscore.mean(),
        "priority_online_advantage_zscore_mean": priority_online_advantage_zscore.mean(),
        "denom": denom,
        "offline_priority_mean": offline_priorities.mean(),
        "online_priority_mean": online_priorities.mean(),
    }

    return offline_priorities, online_priorities, metrics


def evaluate(agent, env, num_episodes=10):
    """Evaluate agent deterministically."""
    returns = []
    lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0

        while not done:
            action = agent.eval_actions(obs[None].astype(np.float32))
            action = int(np.asarray(action[0]))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        "eval/return_mean": np.mean(returns),
        "eval/return_std": np.std(returns),
        "eval/length_mean": np.mean(lengths),
    }


# =============================================================================
# Training Functions
# =============================================================================

def train_rlpd(args, agent, offline_buffer, online_buffer, env, eval_env):
    """RLPD training: uniform 50/50 sampling."""
    obs, _ = env.reset()

    for step in trange(args.max_env_steps, desc="Training (RLPD)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = int(np.asarray(action[0]))

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0
        online_buffer.insert(obs, action, next_obs, reward, mask)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            if "episode" in info:
                wandb.log({
                    "train/episode_return": info["episode"]["r"],
                    "train/episode_length": info["episode"]["l"],
                }, step=step)

        # Sample 50/50
        half_batch = (args.batch_size * args.utd_ratio) // 2
        offline_batch = offline_buffer.sample(half_batch)
        online_batch = online_buffer.sample(half_batch)
        batch = combine_batches(offline_batch, online_batch)

        # Update
        agent, info = agent.update(batch, args.utd_ratio, weights=None)

        # Logging
        if step % args.log_interval == 0:
            wandb.log({
                "train/critic_loss": float(info["critic_loss"]),
                "train/actor_loss": float(info["actor_loss"]),
                "train/temperature": float(info["temperature"]),
                "train/entropy": float(info["entropy"]),
                "train/q_mean": float(info["q_mean"]),
            }, step=step)

        # Evaluation
        if step % args.eval_interval == 0:
            eval_info = evaluate(agent, eval_env)
            wandb.log(eval_info, step=step)
            print(f"\nStep {step}: eval_return={eval_info['eval/return_mean']:.2f}")

    return agent


def train_a3rl(args, agent, density_net, offline_buffer, online_buffer,
               priority_offline_buffer, priority_online_buffer, env, eval_env):
    """A3RL training with priority replay."""
    obs, _ = env.reset()

    warmup_steps = int(0.25 * args.max_env_steps)
    warmup_priority = float(np.exp(args.advantage_lambda * 3.0))

    print(f"A3RL with separate priority buffers")
    print(f"First {warmup_steps} steps: warmup with uniform sampling")
    print(f"Remaining {args.max_env_steps - warmup_steps} steps: priority sampling")

    for step in trange(args.max_env_steps, desc="Training (A3RL)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = int(np.asarray(action[0]))

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to buffers
        online_buffer.insert(obs, action, next_obs, reward, mask)
        if step < warmup_steps:
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, warmup_priority
            )
        else:
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, priority_online_buffer.max_priority
            )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            if "episode" in info:
                wandb.log({
                    "train/episode_return": info["episode"]["r"],
                    "train/episode_length": info["episode"]["l"],
                }, step=step)

        # Density update (uniform 50/50)
        half_total = (args.batch_size * args.utd_ratio) // 2
        density_offline_batch = offline_buffer.sample(half_total)
        density_online_batch = online_buffer.sample(half_total)
        density_batch = combine_batches(density_offline_batch, density_online_batch)
        density_net, density_info = density_net.update(density_batch, args.utd_ratio)

        # SAC update
        is_warmup = step < warmup_steps

        if is_warmup:
            # Uniform sampling during warmup
            half_batch = (args.batch_size * args.utd_ratio) // 2
            sac_offline_batch = offline_buffer.sample(half_batch)
            sac_online_batch = online_buffer.sample(half_batch)
            sac_batch = combine_batches(sac_offline_batch, sac_online_batch)

            agent, sac_info = agent.update(
                sac_batch, args.utd_ratio,
                weights=None,
                offline_only_batch=None,
                online_only_batch=None
            )
        else:
            # Priority sampling
            half_batch = (args.batch_size * args.utd_ratio) // 2
            sac_offline_batch, offline_indices, _ = priority_offline_buffer.sample(
                half_batch, step=step, utd_ratio=None
            )
            sac_online_batch, online_indices, _ = priority_online_buffer.sample(
                half_batch, step=step, utd_ratio=None
            )
            sac_batch = combine_batches(sac_offline_batch, sac_online_batch)

            # Normalization batches
            norm_batch_size = (args.batch_size * args.utd_ratio) // 2
            norm_offline_batch = offline_buffer.sample(norm_batch_size)
            norm_online_batch = online_buffer.sample(norm_batch_size)

            agent, sac_info = agent.update(
                sac_batch, args.utd_ratio,
                weights=None,
                offline_only_batch=norm_offline_batch,
                online_only_batch=norm_online_batch
            )

        # Priority update (after warmup)
        if not is_warmup:
            last_batch_size = args.batch_size
            last_offline_indices = offline_indices[-last_batch_size//2:]
            last_online_indices = online_indices[-last_batch_size//2:]

            # Get observations/actions for density
            last_obs = sac_batch["observations"][-last_batch_size:]
            last_actions = sac_batch["actions"][-last_batch_size:]

            # Forward density (only need offline density for priority weighting)
            last_density = density_net(last_obs, last_actions)
            offline_density = last_density[0::2]

            # Split Q-values and V-values (interleaved: off, on, off, on, ...)
            priority_offline_qs = sac_info["qs"][:, -last_batch_size::2]
            priority_online_qs = sac_info["qs"][:, -last_batch_size+1::2]
            priority_offline_v = sac_info["v"][:, -last_batch_size::2]
            priority_online_v = sac_info["v"][:, -last_batch_size+1::2]

            # Get actions for priority samples (interleaved)
            priority_offline_actions = last_actions[0::2]
            priority_online_actions = last_actions[1::2]

            # Get normalization batch Q-values and V-values
            offline_qs_from_batch = sac_info["offline_qs"]
            offline_v_from_batch = sac_info["offline_v"]
            online_qs_from_batch = sac_info["online_qs"]
            online_v_from_batch = sac_info["online_v"]

            # Compute priorities using V for advantage: A(s,a) = Q(s,a) - V(s)
            offline_priorities, online_priorities, priority_metrics = compute_a3rl_priorities_separate(
                offline_qs_from_batch,
                offline_v_from_batch,
                online_qs_from_batch,
                online_v_from_batch,
                priority_offline_qs,
                priority_offline_v,
                priority_online_qs,
                priority_online_v,
                priority_offline_actions,
                priority_online_actions,
                offline_density,
                args.advantage_lambda,
                args.advantage_beta,
                args.priority_alpha,
            )

            # Update priorities
            priority_offline_buffer.update_priorities(last_offline_indices, np.asarray(offline_priorities))
            priority_online_buffer.update_priorities(last_online_indices, np.asarray(online_priorities))

        # Logging
        if step % args.log_interval == 0:
            offline_entropy = priority_offline_buffer.compute_entropy()
            online_entropy = priority_online_buffer.compute_entropy()
            log_dict = {
                "train/critic_loss": float(sac_info["critic_loss"]),
                "train/actor_loss": float(sac_info["actor_loss"]),
                "train/temperature": float(sac_info["temperature"]),
                "train/entropy": float(sac_info["entropy"]),
                "train/q_mean": float(sac_info["q_mean"]),
                "train/density_loss": float(density_info["density_loss"]),
                "train/offline_weight": float(density_info["offline_weight"]),
                "train/online_weight": float(density_info["online_weight"]),
                "train/offline_max_priority": priority_offline_buffer.max_priority,
                "train/online_max_priority": priority_online_buffer.max_priority,
                "train/offline_priority_entropy": float(offline_entropy),
                "train/online_priority_entropy": float(online_entropy),
                "train/is_warmup": float(is_warmup),
            }
            if not is_warmup:
                for k, v in priority_metrics.items():
                    log_dict[f"priority/{k}"] = float(v)
            wandb.log(log_dict, step=step)

        # Evaluation
        if step % args.eval_interval == 0:
            eval_info = evaluate(agent, eval_env)
            wandb.log(eval_info, step=step)
            print(f"\nStep {step}: eval_return={eval_info['eval/return_mean']:.2f}")

    return agent


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="a3rl", choices=["rlpd", "a3rl"])

    # Core
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--max_env_steps", type=int, default=250_001)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--utd_ratio", type=int, default=10)
    parser.add_argument("--collect_steps", type=int, default=5000)

    # Atari-specific
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--cnn_features", type=int, nargs="+", default=[32, 64, 64])
    parser.add_argument("--cnn_kernels", type=int, nargs="+", default=[8, 4, 3])
    parser.add_argument("--cnn_strides", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--encoder_dim", type=int, default=512)

    # SAC hyperparameters
    parser.add_argument("--encoder_lr", type=float, default=3e-4)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--density_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--num_min_qs", type=int, default=2)

    # Priority replay (A3RL)
    parser.add_argument("--priority_alpha", type=float, default=0.2)
    parser.add_argument("--priority_beta_start", type=float, default=0.4)
    parser.add_argument("--priority_beta_frames", type=int, default=None)
    parser.add_argument("--advantage_lambda", type=float, default=1.0)
    parser.add_argument("--advantage_beta", type=float, default=-0.2)

    # Logging
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--wandb_entity", type=str, default="hung-active-rlpd")
    parser.add_argument("--wandb_project", type=str, default="a3rl_main")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_version", type=str, default="v00")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()

    # Defaults
    if args.priority_beta_frames is None:
        args.priority_beta_frames = args.max_env_steps

    # Seed
    np.random.seed(args.seed)

    # Parse dataset
    dataset_specs = parse_dataset_spec(args.dataset_id)

    # Create dataset name
    if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
        dataset_name = dataset_specs[0][0]
    else:
        dataset_name = "_".join([f"{ds}p{int(pct*100)}" for ds, pct in dataset_specs])

    # Parameter string
    if args.algorithm == "rlpd":
        param_str = "rlpd"
    else:
        param_str = f"a3rl_a{args.priority_alpha}_l{args.advantage_lambda}_b{args.advantage_beta}"

    if args.run_id is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        args.run_id = f"{args.wandb_version}_{dataset_name.replace('/', '_')}_seed{args.seed}_{param_str}_{timestamp}"

    if args.wandb_group is None:
        args.wandb_group = f"{args.wandb_version}_{dataset_name.replace('/', '_')}_{param_str}"

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Environment
    first_dataset_id = dataset_specs[0][0]
    dataset = minari.load_dataset(first_dataset_id, download=True)
    env = wrap_env_atari(dataset.recover_environment(), num_stack=args.frame_stack)
    eval_env = wrap_env_atari(dataset.recover_environment(), num_stack=args.frame_stack)

    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1000)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")

    # Agent
    agent = SACAtariLearner.create(
        seed=args.seed,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cnn_features=tuple(args.cnn_features),
        cnn_kernels=tuple(args.cnn_kernels),
        cnn_strides=tuple(args.cnn_strides),
        encoder_dim=args.encoder_dim,
        num_qs=args.num_qs,
        num_min_qs=args.num_min_qs,
        encoder_lr=args.encoder_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temp_lr=args.alpha_lr,
        discount=args.discount,
        tau=args.tau,
    )

    # Warmup priority
    warmup_priority = float(np.exp(args.advantage_lambda * 3.0)) if args.algorithm == "a3rl" else 1.0

    # Setup buffers
    if args.algorithm == "rlpd":
        offline_buffer = UniformReplayBufferAtari(obs_shape=obs_shape, capacity=50000)
        if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
            offline_buffer.load_from_minari(dataset_specs[0][0], download=True)
        else:
            offline_buffer.load_mixed_datasets(dataset_specs, download=True)
        print(f"Loaded offline dataset: {offline_buffer.size} transitions")

        online_buffer = UniformReplayBufferAtari(obs_shape=obs_shape, capacity=50000)
        density_net = None
        priority_offline_buffer = None
        priority_online_buffer = None

    else:  # a3rl
        offline_buffer = UniformReplayBufferAtari(obs_shape=obs_shape, capacity=50000)
        if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
            offline_buffer.load_from_minari(dataset_specs[0][0], download=True)
        else:
            offline_buffer.load_mixed_datasets(dataset_specs, download=True)
        print(f"Loaded offline dataset: {offline_buffer.size} transitions")

        online_buffer = UniformReplayBufferAtari(obs_shape=obs_shape, capacity=50000)

        priority_offline_buffer = PriorityReplayBufferAtari(
            obs_shape=obs_shape,
            capacity=50000,
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )
        priority_online_buffer = PriorityReplayBufferAtari(
            obs_shape=obs_shape,
            capacity=50000,
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )

        priority_offline_buffer.load_from_buffer(offline_buffer, priority=warmup_priority)
        print(f"Initialized priority offline buffer with {priority_offline_buffer.size} transitions")

        # Density network
        density_net = DensityNetworkAtari.create(
            seed=args.seed + 1,
            obs_shape=obs_shape,
            num_actions=num_actions,
            cnn_features=tuple(args.cnn_features),
            cnn_kernels=tuple(args.cnn_kernels),
            cnn_strides=tuple(args.cnn_strides),
            encoder_dim=args.encoder_dim,
            encoder_lr=args.encoder_lr,
            head_lr=args.density_lr,
        )

    # WandB
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=vars(args),
        name=args.run_id,
    )

    # Initial data collection
    print(f"Collecting {args.collect_steps} initial online samples...")
    obs, _ = env.reset()

    for _ in trange(args.collect_steps, desc="Collecting"):
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = int(np.asarray(action[0]))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        online_buffer.insert(obs, action, next_obs, reward, mask)

        if args.algorithm == "a3rl":
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, warmup_priority
            )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Collection complete. Online buffer size: {online_buffer.size}")
    if args.algorithm == "a3rl":
        print(f"Priority offline buffer size: {priority_offline_buffer.size}")
        print(f"Priority online buffer size: {priority_online_buffer.size}")

    # Training
    if args.algorithm == "rlpd":
        agent = train_rlpd(args, agent, offline_buffer, online_buffer, env, eval_env)
    else:
        agent = train_a3rl(args, agent, density_net, offline_buffer, online_buffer,
                          priority_offline_buffer, priority_online_buffer, env, eval_env)

    print("Training complete!")


if __name__ == "__main__":
    main()
