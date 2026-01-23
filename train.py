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


def compute_a3rl_priorities(
    qs: jnp.ndarray,
    qs_policy: jnp.ndarray,
    density_weights: jnp.ndarray,
    denom: float,
    advantage_lambda: float,
    advantage_beta: float,
    priority_alpha: float,
    offline_qs: jnp.ndarray,
    offline_qs_policy: jnp.ndarray,
):
    """
    Compute A3RL priorities using advantage with Z-score normalization.

    New priority = (density(s,a)**priority_alpha / denom) * exp(lambda * advantage_zscore)

    Args:
        qs: Q-values from critic on batch actions (num_qs, batch)
        qs_policy: Q-values from critic on policy actions (num_qs, batch)
        density_weights: Density(s,a) for each sample in last minibatch
        denom: Mean of (density ** priority_alpha) on offline samples from density batch
        advantage_lambda: Temperature for exp scaling
        advantage_beta: Coefficient for advantage calculation (mean + beta * std)
        priority_alpha: Exponent for density weighting
        offline_qs: Q-values for offline-only batch actions (num_qs, offline_batch)
        offline_qs_policy: Q-values for offline-only batch policy actions (num_qs, offline_batch)

    Returns:
        priorities, metrics_dict
    """
    # Compute advantage for main batch
    advantages = qs - qs_policy  # (num_qs, batch)
    adv_mean = advantages.mean(axis=0)
    adv_std = advantages.std(axis=0)
    advantage = adv_mean + advantage_beta * adv_std

    # Compute advantage for offline batch (same beta)
    offline_advantages = offline_qs - offline_qs_policy  # (num_qs, offline_batch)
    offline_adv_mean = offline_advantages.mean(axis=0)
    offline_adv_std = offline_advantages.std(axis=0)
    offline_advantage = offline_adv_mean + advantage_beta * offline_adv_std

    # Normalize as Z-score using offline batch statistics
    offline_advantage_mean = offline_advantage.mean()
    offline_advantage_std = offline_advantage.std()
    advantage_zscore = (advantage - offline_advantage_mean) / jnp.maximum(offline_advantage_std, 1e-8)
    # Clip Z-score to (-3, 3)
    advantage_zscore = jnp.clip(advantage_zscore, -3.0, 3.0)

    # Priority = (density^alpha / denom) * exp(lambda * advantage_zscore)
    advantage_zscore_scaled = advantage_lambda * advantage_zscore

    density_term = (density_weights ** priority_alpha) / jnp.maximum(denom, 1e-8)
    advantage_term = jnp.exp(advantage_zscore_scaled)

    priorities = density_term.squeeze() * advantage_term
    # Ensure no NaN or inf values
    priorities = jnp.nan_to_num(priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    priorities = jnp.clip(priorities, 1e-5, 1e5)

    metrics = {
        "advantage_mean": advantage.mean(),
        "advantage_std": advantage.std(),
        "advantage_min": advantage.min(),
        "advantage_max": advantage.max(),
        "offline_advantage_mean": offline_advantage_mean,
        "offline_advantage_std": offline_advantage_std,
        "advantage_zscore_mean": advantage_zscore.mean(),
        "advantage_zscore_std": advantage_zscore.std(),
        "advantage_zscore_min": advantage_zscore.min(),
        "advantage_zscore_max": advantage_zscore.max(),
        "density_term_mean": density_term.mean(),
        "density_term_std": density_term.std(),
        "advantage_term_mean": advantage_term.mean(),
        "advantage_term_std": advantage_term.std(),
        "priority_mean": priorities.mean(),
        "priority_std": priorities.std(),
        "priority_min": priorities.min(),
        "priority_max": priorities.max(),
    }

    return priorities, metrics


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


def train_a3rl(args, agent, density_net, offline_buffer, online_buffer, priority_buffer,
               offline_size, env, eval_env):
    """A3RL training with RLPD warmup phase.

    First 25% of steps: Follow RLPD (uniform 50/50 sampling) but maintain density learning
                        and priority buffer with 0.5 priority for online samples.
    Remaining 75% of steps: Switch to A3RL (priority-weighted sampling with density-based priorities).
    """
    obs, _ = env.reset()
    denom = 1.0  # Will be updated by density network

    warmup_steps = int(0.25 * args.max_env_steps)

    print(f"A3RL with RLPD warmup: first {warmup_steps} steps use uniform 50/50 sampling")
    print(f"Then switch to priority-weighted A3RL sampling for remaining {args.max_env_steps - warmup_steps} steps")

    for step in trange(args.max_env_steps, desc="Training (A3RL)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = np.asarray(action[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to BOTH online_buffer AND priority_buffer
        online_buffer.insert(obs, action, next_obs, reward, mask)
        # During warmup: use 0.5 priority; after: use max_priority (updated by A3RL)
        if step < warmup_steps:
            priority_buffer.insert_with_priority(obs, action, next_obs, reward, mask, 0.5 * len(offline_buffer)/warmup_steps)
        else:
            priority_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, priority_buffer.max_priority
            )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            if "episode" in info:
                wandb.log({
                    "train/episode_return": info["episode"]["r"],
                    "train/episode_length": info["episode"]["l"],
                }, step=step)

        # ===== DENSITY UPDATE (RLPD-style: UTD_ratio steps) =====
        # Sample 50/50 from offline and online buffers (uniform)
        # Total samples = batch_size * utd_ratio, split evenly between offline/online
        half_total = (args.batch_size * args.utd_ratio) // 2
        density_offline_batch = offline_buffer.sample(half_total)
        density_online_batch = online_buffer.sample(half_total)

        # Combine batches (interleave) - same pattern as SAC
        density_batch = combine_batches(density_offline_batch, density_online_batch)

        density_net, density_info = density_net.update(density_batch, args.utd_ratio)

        # Compute denom = mean(density(s,a) ** priority_alpha) on last minibatch offline samples
        denom = (density_info["offline_weights"] ** args.priority_alpha).mean()

        # ===== SAC UPDATE =====
        is_warmup_phase = step < warmup_steps

        if is_warmup_phase:
            # WARMUP: Sample uniformly 50/50 from offline and online buffers
            half_batch = (args.batch_size * args.utd_ratio) // 2
            sac_offline_batch = offline_buffer.sample(half_batch)
            sac_online_batch = online_buffer.sample(half_batch)
            sac_batch = combine_batches(sac_offline_batch, sac_online_batch)
            sac_weights = None  # Uniform sampling
            sac_indices = None  # Not needed for warmup
            offline_sample_pct = 0.5  # Exactly 50/50 during warmup
        else:
            # A3RL: Sample from priority_buffer (priority-weighted)
            total_sac_samples = args.batch_size * args.utd_ratio
            sac_batch, sac_indices, sac_weights = priority_buffer.sample(
                total_sac_samples, step=step, utd_ratio=args.utd_ratio
            )
            # Compute percentage of offline samples (indices < offline_size are offline)
            offline_sample_pct = (sac_indices < offline_size).mean()

        # Sample offline-only batch for normalization statistics (no backprop)
        offline_only_batch = offline_buffer.sample(args.batch_size)

        # SAC update with IS weights and offline-only batch for normalization
        agent, sac_info = agent.update(
            sac_batch, args.utd_ratio,
            weights=sac_weights,
            offline_only_batch=offline_only_batch
        )

        # ===== PRIORITY UPDATE (A3RL phase only, after warmup) =====
        if not is_warmup_phase:
            last_batch_size = args.batch_size
            last_indices = sac_indices[-last_batch_size:]

            # Get observations/actions for last minibatch
            last_obs = sac_batch["observations"][-last_batch_size:]
            last_actions = sac_batch["actions"][-last_batch_size:]

            # Forward pass density on last minibatch
            last_density = density_net(last_obs, last_actions)

            # Compute new priorities using advantage with Z-score normalization
            new_priorities, priority_metrics = compute_a3rl_priorities(
                sac_info["qs"],
                sac_info["qs_policy"],
                last_density,
                denom,
                args.advantage_lambda,
                args.advantage_beta,
                args.priority_alpha,
                sac_info["offline_qs"],
                sac_info["offline_qs_policy"],
            )

            # Update priorities in priority_buffer
            priority_buffer.update_priorities(last_indices, np.asarray(new_priorities))

        # ===== LOGGING =====
        if step % args.log_interval == 0:
            priority_entropy = priority_buffer.compute_entropy()
            log_dict = {
                "train/critic_loss": float(sac_info["critic_loss"]),
                "train/actor_loss": float(sac_info["actor_loss"]),
                "train/temperature": float(sac_info["temperature"]),
                "train/entropy": float(sac_info["entropy"]),
                "train/q_mean": float(sac_info["q_mean"]),
                "train/density_loss": float(density_info["density_loss"]),
                "train/offline_weight": float(density_info["offline_weight"]),
                "train/online_weight": float(density_info["online_weight"]),
                "train/denom": float(denom),
                "train/max_priority": priority_buffer.max_priority,
                "train/priority_entropy": float(priority_entropy),
                "train/offline_sample_pct": float(offline_sample_pct),
                "train/is_warmup_phase": float(is_warmup_phase),
                "train/env_steps": step,
            }
            if not is_warmup_phase:
                log_dict["train/is_weight_mean"] = float(sac_weights.mean())
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
        dataset_name = "mixed_" + "_".join([
            f"{ds.split('/')[-1]}p{int(pct*100)}" for ds, pct in dataset_specs
        ])

    # Parameter string for naming
    if args.algorithm == "rlpd":
        param_str = "rlpd"
    else:
        param_str = f"a3rl_a{args.priority_alpha}_l{args.advantage_lambda}_b{args.advantage_beta}"

    if args.run_id is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        args.run_id = f"{dataset_name.replace('/', '_')}_seed{args.seed}_{param_str}_{timestamp}"

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
        priority_buffer = None

    else:  # a3rl
        # A3RL: Two uniform buffers + one priority buffer
        offline_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)
        if len(dataset_specs) == 1 and dataset_specs[0][1] == 1.0:
            # Single dataset, load directly
            offline_buffer.load_from_minari(dataset_specs[0][0], download=True)
        else:
            # Mixed datasets
            offline_buffer.load_mixed_datasets(dataset_specs, download=True)
        print(f"Loaded offline dataset: {offline_buffer.size} transitions")

        online_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)

        priority_buffer = PriorityReplayBuffer(
            obs_dim, action_dim,
            capacity=4_000_000,  # Holds both offline + online
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )

        # Initialize priority_buffer with offline data (priority = 1.0)
        priority_buffer.load_from_buffer(offline_buffer, priority=1.0)
        print(f"Initialized priority buffer with {priority_buffer.size} offline transitions")

        # Default priority for initial collection only
        default_priority = offline_buffer.size / args.collect_steps
        print(f"Default priority for initial collection: {default_priority:.2f}")

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

        # A3RL: Also add to priority buffer with default_priority (only during initial collection)
        if args.algorithm == "a3rl":
            priority_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, default_priority
            )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Collection complete. Online buffer size: {online_buffer.size}")
    if args.algorithm == "a3rl":
        print(f"Priority buffer size: {priority_buffer.size}")

    # Training
    if args.algorithm == "rlpd":
        agent = train_rlpd(args, agent, offline_buffer, online_buffer, env, eval_env)
    else:
        # Pass offline_size to track percentage of offline samples in priority buffer
        offline_size = offline_buffer.size
        agent = train_a3rl(args, agent, density_net, offline_buffer, online_buffer,
                          priority_buffer, offline_size, env, eval_env)

    print("Training complete!")


if __name__ == "__main__":
    main()
