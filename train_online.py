"""Online-only A3RL training script with importance sampling."""

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
from data.buffer import UniformReplayBuffer, PriorityReplayBuffer, wrap_env


def compute_online_priorities(
    online_qs_from_batch: jnp.ndarray,
    online_qs_policy_from_batch: jnp.ndarray,
    priority_online_qs: jnp.ndarray,
    priority_online_qs_policy: jnp.ndarray,
    advantage_lambda: float,
    advantage_beta: float,
):
    """
    Compute priorities for online-only samples using advantage-based weighting.

    Online priority: exp(lambda * advantage_zscore)

    Args:
        online_qs_from_batch: Q-values from critic on online batch actions (num_qs, batch_size)
        online_qs_policy_from_batch: Q-values from critic on online batch policy actions (num_qs, batch_size)
        priority_online_qs: Q-values from critic on priority online batch actions (num_qs, batch_size)
        priority_online_qs_policy: Q-values from critic on priority online batch policy actions (num_qs, batch_size)
        advantage_lambda: Temperature for exp scaling
        advantage_beta: Coefficient for advantage calculation (mean + beta * std)

    Returns:
        online_priorities, metrics_dict
    """
    # Compute advantage for online batch (for normalization statistics)
    online_advantages = online_qs_from_batch - online_qs_policy_from_batch  # (num_qs, batch_size)
    online_adv_mean = online_advantages.mean(axis=0)
    online_adv_std = online_advantages.std(axis=0)
    online_advantage_stats = online_adv_mean + advantage_beta * online_adv_std

    # Normalize using statistics from online batch
    online_advantage_mean = online_advantage_stats.mean()
    online_advantage_std = online_advantage_stats.std()

    # Compute advantages for priority buffer samples
    priority_online_advantages = priority_online_qs - priority_online_qs_policy
    priority_online_adv_mean = priority_online_advantages.mean(axis=0)
    priority_online_adv_std = priority_online_advantages.std(axis=0)
    priority_online_advantage = priority_online_adv_mean + advantage_beta * priority_online_adv_std

    # Normalize priority buffer advantages using batch statistics
    priority_online_advantage_zscore = (priority_online_advantage - online_advantage_mean) / jnp.maximum(online_advantage_std, 1e-8)

    # Clip Z-scores to (-3, 3)
    priority_online_advantage_zscore = jnp.clip(priority_online_advantage_zscore, -3.0, 3.0)

    # Online priority: exp(lambda * advantage_zscore)
    online_advantage_scaled = advantage_lambda * priority_online_advantage_zscore
    online_advantage_term = jnp.exp(online_advantage_scaled)
    online_priorities = online_advantage_term.squeeze()

    # Ensure no NaN or inf values
    online_priorities = jnp.nan_to_num(online_priorities, nan=1.0, posinf=1e5, neginf=1e-5)
    online_priorities = jnp.clip(online_priorities, 1e-5, 1e5)

    metrics = {
        "online_advantage_stats_mean": online_advantage_stats.mean(),
        "online_advantage_stats_std": online_advantage_stats.std(),
        "priority_online_advantage_mean": priority_online_advantage.mean(),
        "priority_online_advantage_std": priority_online_advantage.std(),
        "priority_online_advantage_min": priority_online_advantage.min(),
        "priority_online_advantage_max": priority_online_advantage.max(),
        "priority_online_advantage_zscore_mean": priority_online_advantage_zscore.mean(),
        "priority_online_advantage_zscore_std": priority_online_advantage_zscore.std(),
        "online_advantage_term_mean": online_advantage_term.mean(),
        "online_advantage_term_std": online_advantage_term.std(),
        "online_priority_mean": online_priorities.mean(),
        "online_priority_std": online_priorities.std(),
    }

    return online_priorities, metrics


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


def train_online_a3rl(args, agent, online_buffer, priority_online_buffer, env, eval_env):
    """Online-only A3RL training with priority replay and importance sampling.

    SAC network: samples from priority_online_buffer (with priority weighting).
    SAC updates: WITH importance sampling weights from priority buffer.
    """
    obs, _ = env.reset()

    warmup_steps = int(0.25 * args.max_env_steps)
    warmup_priority = float(np.exp(args.advantage_lambda * 3.0))

    print(f"Online-only A3RL with priority replay and importance sampling")
    print(f"First {warmup_steps} steps: warmup phase with uniform sampling (priority={warmup_priority:.4f})")
    print(f"Remaining {args.max_env_steps - warmup_steps} steps: priority-weighted sampling with IS weights")

    for step in trange(args.max_env_steps, desc="Training (Online A3RL)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = np.asarray(action[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to online_buffer AND priority_online_buffer
        online_buffer.insert(obs, action, next_obs, reward, mask)
        # During warmup: use warmup_priority; after: use max_priority (updated by A3RL)
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

        # ===== SAC UPDATE =====
        is_warmup_phase = step < warmup_steps

        if is_warmup_phase:
            # WARMUP: Sample uniformly from online buffer
            sac_batch = online_buffer.sample(args.batch_size * args.utd_ratio)

            # SAC update WITHOUT importance sampling weights (uniform loss weighting during warmup)
            agent, sac_info = agent.update(
                sac_batch, args.utd_ratio,
                weights=None,  # No importance sampling during warmup
                offline_only_batch=None,
                online_only_batch=None
            )
        else:
            # A3RL: Sample from priority_online buffer (with priority weighting)
            sac_batch, online_indices, online_weights = priority_online_buffer.sample(
                args.batch_size * args.utd_ratio, step=step, utd_ratio=None
            )

            # Sample separate online batch for normalization statistics
            norm_batch_size = args.batch_size
            norm_online_batch = online_buffer.sample(norm_batch_size)

            agent, sac_info = agent.update(
                sac_batch, args.utd_ratio,
                weights=None,
                offline_only_batch=None,
                online_only_batch=norm_online_batch
            )

        # ===== PRIORITY UPDATE (A3RL phase only, after warmup) =====
        if not is_warmup_phase:
            last_batch_size = args.batch_size

            # Extract last minibatch indices
            last_online_indices = online_indices[-last_batch_size:]

            # Get Q-values and Q-policy from last minibatch
            priority_online_qs = sac_info["qs"][:, -last_batch_size:]
            priority_online_qs_policy = sac_info["qs_policy"][:, -last_batch_size:]

            # Get Q-values from normalization batch
            online_qs_from_batch = sac_info["online_qs"]
            online_qs_policy_from_batch = sac_info["online_qs_policy"]

            # Compute new priorities
            online_priorities, priority_metrics = compute_online_priorities(
                online_qs_from_batch,
                online_qs_policy_from_batch,
                priority_online_qs,
                priority_online_qs_policy,
                args.advantage_lambda,
                args.advantage_beta,
            )

            # Update priorities in buffer
            priority_online_buffer.update_priorities(last_online_indices, np.asarray(online_priorities))

        # ===== LOGGING =====
        if step % args.log_interval == 0:
            online_entropy = priority_online_buffer.compute_entropy()
            log_dict = {
                "train/critic_loss": float(sac_info["critic_loss"]),
                "train/actor_loss": float(sac_info["actor_loss"]),
                "train/temperature": float(sac_info["temperature"]),
                "train/entropy": float(sac_info["entropy"]),
                "train/q_mean": float(sac_info["q_mean"]),
                "train/online_max_priority": priority_online_buffer.max_priority,
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


def train_pure_sac(args, agent, online_buffer, env, eval_env):
    """Pure online SAC training with uniform sampling (no priority replay)."""
    obs, _ = env.reset()

    print(f"Pure online SAC with uniform sampling")

    for step in trange(args.max_env_steps, desc="Training (Pure SAC)"):
        # Environment interaction
        action, agent = agent.sample_actions(obs[None].astype(np.float32))
        action = np.asarray(action[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = 0.0 if (terminated or truncated) else 1.0

        # Add to online buffer
        online_buffer.insert(obs, action, next_obs, reward, mask)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            if "episode" in info:
                wandb.log({
                    "train/episode_return": info["episode"]["r"],
                    "train/episode_length": info["episode"]["l"],
                }, step=step)

        # Sample uniformly from online buffer
        sac_batch = online_buffer.sample(args.batch_size * args.utd_ratio)

        # SAC update without importance sampling weights
        agent, sac_info = agent.update(
            sac_batch, args.utd_ratio,
            weights=None,
            offline_only_batch=None,
            online_only_batch=None
        )

        # ===== LOGGING =====
        if step % args.log_interval == 0:
            log_dict = {
                "train/critic_loss": float(sac_info["critic_loss"]),
                "train/actor_loss": float(sac_info["actor_loss"]),
                "train/temperature": float(sac_info["temperature"]),
                "train/entropy": float(sac_info["entropy"]),
                "train/q_mean": float(sac_info["q_mean"]),
                "train/env_steps": step,
            }
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
                        choices=["pure", "a3rl"],
                        help="Algorithm: 'pure' (uniform SAC) or 'a3rl' (priority replay with IS)")

    # Core
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset to use for recovering the environment")
    parser.add_argument("--max_env_steps", type=int, default=250_001)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd_ratio", type=int, default=10)
    parser.add_argument("--collect_steps", type=int, default=5000)

    # SAC hyperparameters
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--num_min_qs", type=int, default=2)

    # Priority replay (A3RL only)
    parser.add_argument("--priority_beta_start", type=float, default=0.4)
    parser.add_argument("--priority_beta_frames", type=int, default=None)
    parser.add_argument("--advantage_lambda", type=float, default=1.0)
    parser.add_argument("--advantage_beta", type=float, default=-0.2)

    # Logging
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--wandb_entity", type=str, default="hung-active-rlpd")
    parser.add_argument("--wandb_project", type=str, default="a3rl_online")
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

    # Parameter string for naming
    if args.algorithm == "pure":
        param_str = "pure_sac"
    else:
        param_str = f"online_a3rl_l{args.advantage_lambda}_b{args.advantage_beta}"

    if args.run_id is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        args.run_id = f"{args.wandb_version}_{args.dataset_id.replace('/', '_')}_seed{args.seed}_{param_str}_{timestamp}"

    if args.wandb_group is None:
        args.wandb_group = f"{args.wandb_version}_{args.dataset_id.replace('/', '_')}_{param_str}"

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Environment (use dataset to recover environment)
    dataset = minari.load_dataset(args.dataset_id, download=True)
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
    online_buffer = UniformReplayBuffer(obs_dim, action_dim, capacity=2_000_000)

    if args.algorithm == "a3rl":
        # Calculate warmup priority: exp(lambda * 3)
        warmup_priority = float(np.exp(args.advantage_lambda * 3.0))

        # Create priority buffer for A3RL
        priority_online_buffer = PriorityReplayBuffer(
            obs_dim, action_dim,
            capacity=2_000_000,
            beta_start=args.priority_beta_start,
            beta_frames=args.priority_beta_frames,
        )
    else:
        priority_online_buffer = None
        warmup_priority = None

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

        # Add to online buffer
        online_buffer.insert(obs, action, next_obs, reward, mask)

        # Add to priority buffer if using A3RL
        if args.algorithm == "a3rl":
            priority_online_buffer.insert_with_priority(
                obs, action, next_obs, reward, mask, warmup_priority
            )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Collection complete. Online buffer size: {online_buffer.size}")
    if args.algorithm == "a3rl":
        print(f"Priority online buffer size: {priority_online_buffer.size}")

    # Training
    if args.algorithm == "pure":
        agent = train_pure_sac(args, agent, online_buffer, env, eval_env)
    else:
        agent = train_online_a3rl(args, agent, online_buffer, priority_online_buffer, env, eval_env)

    print("Training complete!")


if __name__ == "__main__":
    main()
