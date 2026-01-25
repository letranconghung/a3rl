"""Training script for SAC with RLPD or A3RL algorithm modes."""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import minari
from tqdm import trange
from pathlib import Path
from datetime import datetime

from agents.sac import SACLearner
from agents.density import DensityNetwork
from data.buffer import UniformReplayBuffer, PriorityReplayBuffer, wrap_env


def parse_dataset_spec(spec_str):
    """Parse dataset specification string into (dataset_id, percentage) tuples.

    Format: 'dataset1:0.5,dataset2:1.0' or just 'dataset1' (defaults to 1.0)
    Returns: List of (dataset_id, percentage) tuples
    """
    if ',' in spec_str or ':' in spec_str:
        # Multiple datasets or explicit percentages
        specs = []
        for part in spec_str.split(','):
            if ':' in part:
                dataset_id, pct = part.rsplit(':', 1)
                specs.append((dataset_id.strip(), float(pct)))
            else:
                specs.append((part.strip(), 1.0))
        return specs
    else:
        # Single dataset, 100%
        return [(spec_str, 1.0)]


def combine_batches(offline_batch, online_batch):
    """Interleave offline and online batches for balanced minibatches."""
    combined = {}
    for k in offline_batch:
        off = offline_batch[k]
        on = online_batch[k]
        interleaved = jnp.stack([off, on], axis=1).reshape(-1, *off.shape[1:])
        combined[k] = interleaved
    return combined


def compute_a3rl_priorities_separate(
    offline_qs: jnp.ndarray,
    offline_qs_policy: jnp.ndarray,
    online_qs: jnp.ndarray,
    online_qs_policy: jnp.ndarray,
    offline_density: jnp.ndarray,
    online_density: jnp.ndarray,
    advantage_lambda: float,
    advantage_beta: float,
    priority_alpha: float,
):
    """
    Compute A3RL priorities with separate formulas for offline and online.

    Offline priority: (density(s,a)**priority_alpha / denom) * exp(lambda * advantage)
    Online priority: exp(lambda * advantage)

    Where denom = mean(density ** priority_alpha) over offline samples in the batch.

    Args:
        offline_qs: Q-values from critic on offline batch actions (num_qs, batch_size)
        offline_qs_policy: Q-values from critic on offline batch policy actions (num_qs, batch_size)
        online_qs: Q-values from critic on online batch actions (num_qs, batch_size)
        online_qs_policy: Q-values from critic on online batch policy actions (num_qs, batch_size)
        offline_density: Density weights for offline samples (batch_size,)
        online_density: Density weights for online samples (batch_size,)
        advantage_lambda: Temperature for exp scaling
        advantage_beta: Coefficient for advantage calculation (mean + beta * std)
        priority_alpha: Exponent for density weighting

    Returns:
        offline_priorities, online_priorities, metrics_dict
    """
    # Compute advantage for offline batch
    offline_advantages = offline_qs - offline_qs_policy  # (num_qs, batch_size)
    offline_adv_mean = offline_advantages.mean(axis=0)
    offline_adv_std = offline_advantages.std(axis=0)
    offline_advantage = offline_adv_mean + advantage_beta * offline_adv_std

    # Compute advantage for online batch
    online_advantages = online_qs - online_qs_policy  # (num_qs, batch_size)
    online_adv_mean = online_advantages.mean(axis=0)
    online_adv_std = online_advantages.std(axis=0)
    online_advantage = online_adv_mean + advantage_beta * online_adv_std

    # Normalize advantages as Z-score using offline batch statistics
    offline_advantage_mean = offline_advantage.mean()
    offline_advantage_std = offline_advantage.std()
    offline_advantage_zscore = (offline_advantage - offline_advantage_mean) / jnp.maximum(offline_advantage_std, 1e-8)
    online_advantage_zscore = (online_advantage - offline_advantage_mean) / jnp.maximum(offline_advantage_std, 1e-8)
    # Clip Z-scores to (-3, 3)
    offline_advantage_zscore = jnp.clip(offline_advantage_zscore, -3.0, 3.0)
    online_advantage_zscore = jnp.clip(online_advantage_zscore, -3.0, 3.0)

    # Compute denom from offline samples
    density_raised = offline_density ** priority_alpha
    denom = density_raised.sum()

    # Offline priority: (density^alpha / denom) * exp(lambda * advantage_zscore)
    offline_advantage_scaled = advantage_lambda * offline_advantage_zscore
    offline_density_term = density_raised / jnp.maximum(denom, 1e-8)
    offline_advantage_term = jnp.exp(offline_advantage_scaled)
    offline_priorities = offline_density_term.squeeze() * offline_advantage_term

    # Online priority: exp(lambda * advantage_zscore)
    online_advantage_scaled = advantage_lambda * online_advantage_zscore
    online_advantage_term = jnp.exp(online_advantage_scaled)
    online_priorities = online_advantage_term.squeeze()

    # Ensure no NaN or inf values
    offline_priorities = jnp.nan_to_num(offline_priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    offline_priorities = jnp.clip(offline_priorities, 1e-5, 1e5)
    online_priorities = jnp.nan_to_num(online_priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    online_priorities = jnp.clip(online_priorities, 1e-5, 1e5)

    metrics = {
        "offline_advantage_mean": offline_advantage.mean(),
        "offline_advantage_std": offline_advantage.std(),
        "offline_advantage_min": offline_advantage.min(),
        "offline_advantage_max": offline_advantage.max(),
        "online_advantage_mean": online_advantage.mean(),
        "online_advantage_std": online_advantage.std(),
        "online_advantage_min": online_advantage.min(),
        "online_advantage_max": online_advantage.max(),
        "offline_advantage_zscore_mean": offline_advantage_zscore.mean(),
        "offline_advantage_zscore_std": offline_advantage_zscore.std(),
        "online_advantage_zscore_mean": online_advantage_zscore.mean(),
        "online_advantage_zscore_std": online_advantage_zscore.std(),
        "offline_advantage_term_mean": offline_advantage_term.mean(),
        "offline_advantage_term_std": offline_advantage_term.std(),
        "online_advantage_term_mean": online_advantage_term.mean(),
        "online_advantage_term_std": online_advantage_term.std(),
        "denom": denom,
        "offline_density_term_mean": offline_density_term.mean(),
        "offline_density_term_std": offline_density_term.std(),
        "offline_priority_mean": offline_priorities.mean(),
        "offline_priority_std": offline_priorities.std(),
        "online_priority_mean": online_priorities.mean(),
        "online_priority_std": online_priorities.std(),
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
            action = np.asarray(action[0])
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


def train_rlpd(args, agent, offline_buffer, online_buffer, env, eval_env):
    """RLPD training: uniform 50/50 sampling from offline and online buffers."""
    obs, _ = env.reset()

    for step in trange(args.max_env_steps, desc="Training (RLPD)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = np.asarray(action[0])

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

        # Sample 50/50 from offline and online (uniform)
        half_batch = (args.batch_size * args.utd_ratio) // 2
        offline_batch = offline_buffer.sample(half_batch)
        online_batch = online_buffer.sample(half_batch)

        # Combine batches (interleave)
        batch = combine_batches(offline_batch, online_batch)

        # SAC update (no weights for uniform sampling)
        agent, info = agent.update(batch, args.utd_ratio, weights=None)

        # Logging
        if step % args.log_interval == 0:
            wandb.log({
                "train/critic_loss": float(info["critic_loss"]),
                "train/actor_loss": float(info["actor_loss"]),
                "train/temperature": float(info["temperature"]),
                "train/entropy": float(info["entropy"]),
                "train/q_mean": float(info["q_mean"]),
                "train/env_steps": step,
            }, step=step)

        # Evaluation
        if step % args.eval_interval == 0:
            eval_info = evaluate(agent, eval_env)
            wandb.log(eval_info, step=step)
            print(f"\nStep {step}: eval_return={eval_info['eval/return_mean']:.2f}")

    return agent


def train_a3rl(args, agent, density_net, offline_buffer, online_buffer,
               priority_offline_buffer, priority_online_buffer, env, eval_env):
    """A3RL training with priority replay for offline and online buffers separately.

    Density network: samples uniformly 50/50 from offline and online buffers.
    SAC network: samples 50/50 from priority_offline and priority_online buffers (each with priority weighting).
    SAC updates: without importance sampling weights (uniform loss weighting).
    """
    obs, _ = env.reset()
    denom = 1.0  # Will be updated by density network

    warmup_steps = int(0.25 * args.max_env_steps)

    print(f"A3RL with separate priority offline/online buffers")
    print(f"First {warmup_steps} steps: warmup phase with uniform sampling")
    print(f"Remaining {args.max_env_steps - warmup_steps} steps: priority-weighted sampling from separate buffers")

    for step in trange(args.max_env_steps, desc="Training (A3RL)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = np.asarray(action[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to online_buffer AND priority_online_buffer
        online_buffer.insert(obs, action, next_obs, reward, mask)
        # During warmup: use 0.5 priority; after: use max_priority (updated by A3RL)
        if step < warmup_steps:
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, 1
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

        # ===== DENSITY UPDATE (Uniform 50/50 sampling from offline/online) =====
        # Sample uniformly from offline and online buffers
        half_total = (args.batch_size * args.utd_ratio) // 2
        density_offline_batch = offline_buffer.sample(half_total)
        density_online_batch = online_buffer.sample(half_total)

        # Combine batches (interleave)
        density_batch = combine_batches(density_offline_batch, density_online_batch)

        density_net, density_info = density_net.update(density_batch, args.utd_ratio)

        # ===== SAC UPDATE =====
        is_warmup_phase = step < warmup_steps

        if is_warmup_phase:
            # WARMUP: Sample uniformly 50/50 from offline and online buffers
            half_batch = (args.batch_size * args.utd_ratio) // 2
            sac_offline_batch = offline_buffer.sample(half_batch)
            sac_online_batch = online_buffer.sample(half_batch)
            sac_batch = combine_batches(sac_offline_batch, sac_online_batch)
        else:
            # A3RL: Sample 50/50 from priority_offline and priority_online (each with priority weighting)
            half_batch = (args.batch_size * args.utd_ratio) // 2
            sac_offline_batch, offline_indices, offline_weights = priority_offline_buffer.sample(
                half_batch, step=step, utd_ratio=None  # Don't normalize to utd_ratio
            )
            sac_online_batch, online_indices, online_weights = priority_online_buffer.sample(
                half_batch, step=step, utd_ratio=None
            )
            # Combine batches (interleave)
            sac_batch = combine_batches(sac_offline_batch, sac_online_batch)

        # SAC update WITHOUT importance sampling weights (always None for uniform loss weighting)
        agent, sac_info = agent.update(
            sac_batch, args.utd_ratio,
            weights=None,  # No importance sampling - uniform loss weighting
            offline_only_batch=None
        )

        # ===== PRIORITY UPDATE (A3RL phase only, after warmup) =====
        if not is_warmup_phase:
            last_batch_size = args.batch_size

            # Extract last minibatch indices from both offline and online
            last_offline_indices = offline_indices[-last_batch_size//2:]
            last_online_indices = online_indices[-last_batch_size//2:]

            # Get observations/actions for last minibatch (interleaved offline/online)
            last_obs = sac_batch["observations"][-last_batch_size:]
            last_actions = sac_batch["actions"][-last_batch_size:]

            # Forward pass density on last minibatch (returns separate for offline/online due to interleaving)
            last_density = density_net(last_obs, last_actions)
            # Split density values (interleaved as offline, online, offline, online, ...)
            offline_density = last_density[0::2]
            online_density = last_density[1::2]

            # Split Q-values and Q-policy from last minibatch (interleaved as offline, online, offline, online, ...)
            offline_qs = sac_info["qs"][:, -last_batch_size::2]
            online_qs = sac_info["qs"][:, -last_batch_size+1::2]
            offline_qs_policy = sac_info["qs_policy"][:, -last_batch_size::2]
            online_qs_policy = sac_info["qs_policy"][:, -last_batch_size+1::2]

            # Compute new priorities using separate formulas for offline and online
            offline_priorities, online_priorities, priority_metrics = compute_a3rl_priorities_separate(
                offline_qs,
                offline_qs_policy,
                online_qs,
                online_qs_policy,
                offline_density,
                online_density,
                args.advantage_lambda,
                args.advantage_beta,
                args.priority_alpha,
            )

            # Update priorities in separate buffers
            priority_offline_buffer.update_priorities(last_offline_indices, np.asarray(offline_priorities))
            priority_online_buffer.update_priorities(last_online_indices, np.asarray(online_priorities))

        # ===== LOGGING =====
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
                "train/is_warmup_phase": float(is_warmup_phase),
                "train/env_steps": step,
            }
            if not is_warmup_phase:
                for k, v in priority_metrics.items():
                    log_dict[f"priority/{k}"] = float(v)
            wandb.log(log_dict, step=step)

        # Evaluation
        if step % args.eval_interval == 0:
            eval_info = evaluate(agent, eval_env)
            wandb.log(eval_info, step=step)
            print(f"\nStep {step}: eval_return={eval_info['eval/return_mean']:.2f}")

    return agent


def main():
    parser = argparse.ArgumentParser()
    # Algorithm mode
    parser.add_argument("--algorithm", type=str, default="a3rl",
                        choices=["rlpd", "a3rl"],
                        help="Algorithm: 'rlpd' (uniform 50/50) or 'a3rl' (priority replay)")

    # Core
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset spec. Single: 'dataset_id' or mixed: 'dataset1:0.5,dataset2:1.0'")
    parser.add_argument("--max_env_steps", type=int, default=250_001)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd_ratio", type=int, default=10)
    parser.add_argument("--collect_steps", type=int, default=5000)

    # SAC hyperparameters
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--density_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--num_min_qs", type=int, default=2)

    # Priority replay (A3RL only)
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

    # Parse dataset specification
    dataset_specs = parse_dataset_spec(args.dataset_id)

    # Create readable dataset name for logging
    if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
        dataset_name = dataset_specs[0][0]
    else:
        dataset_name = "_".join([
            f"{ds}p{int(pct*100)}" for ds, pct in dataset_specs
        ])

    # Parameter string for naming
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

    # Environment (use first dataset to recover environment)
    first_dataset_id = dataset_specs[0][0]
    dataset = minari.load_dataset(first_dataset_id, download=True)
    env = wrap_env(dataset.recover_environment())
    eval_env = wrap_env(dataset.recover_environment())

    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Agent
    agent = SACLearner.create(
        seed=args.seed,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        num_qs=args.num_qs,
        num_min_qs=args.num_min_qs,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temp_lr=args.alpha_lr,
        discount=args.discount,
        tau=args.tau,
        use_layer_norm=True,
    )

    # Setup buffers based on algorithm
    if args.algorithm == "rlpd":
        # RLPD: Two uniform buffers
        offline_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)
        if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
            # Single dataset, load directly
            offline_buffer.load_from_minari(dataset_specs[0][0], download=True)
        else:
            # Mixed datasets
            offline_buffer.load_mixed_datasets(dataset_specs, download=True)
        print(f"Loaded offline dataset: {offline_buffer.size} transitions")

        online_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)

        density_net = None

    else:  # a3rl
        # A3RL: Two uniform buffers + two separate priority buffers (offline and online)
        offline_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)
        if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
            # Single dataset, load directly
            offline_buffer.load_from_minari(dataset_specs[0][0], download=True)
        else:
            # Mixed datasets
            offline_buffer.load_mixed_datasets(dataset_specs, download=True)
        print(f"Loaded offline dataset: {offline_buffer.size} transitions")

        online_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)

        # Create separate priority buffers for offline and online data
        priority_offline_buffer = PriorityReplayBuffer(
            obs_dim, action_dim,
            capacity=2_000_000,
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )
        priority_online_buffer = PriorityReplayBuffer(
            obs_dim, action_dim,
            capacity=2_000_000,
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )

        # Initialize priority_offline_buffer with offline data (priority = 1.0)
        priority_offline_buffer.load_from_buffer(offline_buffer, priority=1.0)
        print(f"Initialized priority offline buffer with {priority_offline_buffer.size} transitions")

        # Density network
        density_net = DensityNetwork.create(
            seed=args.seed + 1,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            lr=args.density_lr,
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
        action = np.asarray(action[0])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to online buffer (both modes)
        online_buffer.insert(obs, action, next_obs, reward, mask)

        # A3RL: Also add to priority_online_buffer with default_priority (only during initial collection)
        if args.algorithm == "a3rl":
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, 1.0
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
