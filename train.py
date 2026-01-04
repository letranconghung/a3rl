import argparse
import numpy as np
import torch
import wandb
import minari
from agents.sac import SAC
from agents.density import DensityNetwork
import gymnasium as gym
from tqdm import trange
from data.minari_dataset import ReplayBuffer, FlattenAndNormalizeWrapper
from pathlib import Path
from datetime import datetime
import os

def evaluate(env: gym.Env, agent, num_episodes=10, device='cuda'):
    """Evaluate the agent on the environment"""
    episode_returns = []
    episode_lengths = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
                action, _ = agent.actor.sample(obs_tensor, deterministic=True)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'eval/return_mean': np.mean(episode_returns),
        'eval/return_std': np.std(episode_returns),
        'eval/length_mean': np.mean(episode_lengths)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--max_env_steps', type=int, default=250_001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--utd_ratio', type=int, default=5)
    parser.add_argument('--collect_steps', type=int, default=5000)  # Initial online collection steps
    parser.add_argument('--advantage_beta', type=float, default=1.0)  # LCB coefficient
    parser.add_argument('--device', type=str, default='cuda')
    # SAC learning rates
    parser.add_argument('--actor_lr', type=float, default=3e-4,
                       help='Learning rate for actor network')
    parser.add_argument('--critic_lr', type=float, default=3e-4,
                       help='Learning rate for critic network')
    parser.add_argument('--alpha_lr', type=float, default=3e-4,
                       help='Learning rate for temperature parameter')
    parser.add_argument('--wandb_entity', type=str, default="hung-active-rlpd")
    parser.add_argument('--wandb_project', type=str, default="a3rl_main")
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Unique run identifier for checkpoint naming')
    # Density network and priority replay arguments
    parser.add_argument('--density_lr', type=float, default=3e-4,
                       help='Learning rate for density network')
    parser.add_argument('--priority_alpha', type=float, default=0.0,
                       help='Priority exponent (0=uniform, >0=prioritized). Default: 0')
    parser.add_argument('--priority_beta_start', type=float, default=0.4,
                       help='Initial importance sampling exponent')
    parser.add_argument('--priority_beta_frames', type=int, default=None,
                       help='Steps to anneal beta to 1.0 (default: max_env_steps)')
    args = parser.parse_args()

    # Set default beta_frames to max_env_steps if not specified
    if args.priority_beta_frames is None:
        args.priority_beta_frames = args.max_env_steps

    # Enable TF32 for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create unique run identifier if not provided
    if args.run_id is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        args.run_id = f"{args.dataset_id.replace('/', '_')}_seed{args.seed}_{timestamp}"

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Load Minari dataset and create environment
    dataset = minari.load_dataset(args.dataset_id, download=True)
    env = FlattenAndNormalizeWrapper(dataset.recover_environment())
    eval_env = FlattenAndNormalizeWrapper(dataset.recover_environment())

    # Seed environments for reproducibility
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1000)  # Different seed for eval to avoid correlation

    # Get dimensions after wrapping
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create offline buffer and load dataset (with priority support)
    offline_buffer = ReplayBuffer(
        obs_dim, action_dim, capacity=2_000_000, device=args.device,
        alpha=args.priority_alpha,
        beta_start=args.priority_beta_start,
        beta_frames=args.priority_beta_frames
    )
    offline_buffer.load_from_minari(args.dataset_id, download=True)

    # Create online replay buffer (for finetuning, no priority needed)
    online_buffer = ReplayBuffer(obs_dim, action_dim, capacity=args.max_env_steps, device=args.device)

    # Initialize SAC agent
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
        hidden_dims=(256, 256),
        num_qs=10,  # Use 10 Q networks for ensemble advantage estimation
        num_min_qs=2,  # Still use 2 for target Q computation
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        discount=0.99,
        tau=0.005,
        use_layer_norm=True,
    )

    # Initialize density network
    density_net = DensityNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        lr=args.density_lr,
        device=args.device
    )

    # Compile models for faster execution
    print("Compiling models with torch.compile...")
    agent.actor = torch.compile(agent.actor)
    agent.critic = torch.compile(agent.critic)
    agent.target_critic = torch.compile(agent.target_critic)
    density_net = torch.compile(density_net)
    print("Compilation complete!")

    # Initialize training state
    start_step = 0
    wandb_run_id = None

    # Load checkpoint if resuming
    if args.resume_from is not None:
        print(f"Loading checkpoint from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.log_alpha = checkpoint['log_alpha']
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        online_buffer.load_state_dict(checkpoint['online_buffer'])
        # Load density network if available (backward compatible)
        if 'density_state_dict' in checkpoint:
            density_net.load_state_dict(checkpoint['density_state_dict'])
            density_net.optimizer.load_state_dict(checkpoint['density_optimizer'])
        # Load offline buffer priorities if available
        if 'offline_buffer' in checkpoint:
            offline_buffer.load_state_dict(checkpoint['offline_buffer'])
        start_step = checkpoint['step'] + 1  # Resume from step after checkpoint
        wandb_run_id = checkpoint.get('wandb_run_id', None)
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        print(f"Resumed from checkpoint at step {checkpoint['step']}, continuing from step {start_step}")

    # Initialize WandB
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=vars(args),
        name=args.run_id,  # Use run_id as the WandB run name
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None
    )

    # Check if training is already complete
    if start_step >= args.max_env_steps:
        print(f"Training already complete! Start step {start_step} >= max steps {args.max_env_steps}")
        return

    # Initial online data collection phase (skip if resuming)
    if start_step == 0:
        print(f"Collecting {args.collect_steps} online samples using SAC agent...")
        obs, info = env.reset()
        done = True

        for _ in trange(args.collect_steps, desc="Collecting"):
            if done:
                obs, info = env.reset()
                done = False

            # Use SAC agent to collect (with exploration noise)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=args.device).unsqueeze(0)
                action, _ = agent.actor.sample(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add to online buffer
            online_buffer.add(obs, action, next_obs, reward, float(done))
            obs = next_obs

        print(f"Collection complete! Online buffer size: {online_buffer.size}")
    else:
        print(f"Skipping initial collection (resuming from step {start_step})")

    # Training loop
    obs, info = env.reset()
    done = True

    for total_env_steps in trange(start_step, args.max_env_steps, initial=start_step, total=args.max_env_steps, desc="Training"):
        # Online interaction
        if done:
            obs, info = env.reset()
            done = False

        # Sample action
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=args.device).unsqueeze(0)
            action, _ = agent.actor.sample(obs_tensor, deterministic=False)
            action = action.cpu().numpy()[0]

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add to online buffer
        online_buffer.add(obs, action, next_obs, reward, float(done))
        obs = next_obs

        if done and 'episode' in info:
            wandb.log({
                'train/episode_return': info['episode']['r'],
                'train/episode_length': info['episode']['l']
            }, step=total_env_steps)

        # Perform UTD ratio updates
        for _ in range(args.utd_ratio):
            # Mixed training with 0.5 offline ratio
            offline_size = int(args.batch_size * 0.5)
            online_size = args.batch_size - offline_size

            # Sample from buffers (returns batch, indices, weights)
            offline_batch, offline_indices, offline_weights = offline_buffer.sample(
                offline_size, device=args.device, step=total_env_steps
            )
            online_batch, _, _ = online_buffer.sample(online_size, device=args.device)

            # Update density network (before concatenation)
            density_info = density_net.update(offline_batch, online_batch)

            # Update offline buffer priorities using density weights (already computed in update)
            if args.priority_alpha > 0:
                offline_buffer.update_priorities(offline_indices, density_info['offline_weight_tensor'])

            # Concatenate batches for SAC updates
            batch = {
                'obs': torch.cat([offline_batch['obs'], online_batch['obs']], dim=0),
                'actions': torch.cat([offline_batch['actions'], online_batch['actions']], dim=0),
                'next_obs': torch.cat([offline_batch['next_obs'], online_batch['next_obs']], dim=0),
                'rewards': torch.cat([offline_batch['rewards'], online_batch['rewards']], dim=0),
                'dones': torch.cat([offline_batch['dones'], online_batch['dones']], dim=0)
            }

            # Construct importance sampling weights for SAC updates
            # offline_weights: from PER (or None if alpha=0)
            # online_weights: uniform (no priority replay for online buffer)
            # Normalize so total offline weight = 0.5, total online weight = 0.5
            if offline_weights is not None:
                # Normalize offline IS weights to sum to 0.5
                offline_is_weights = offline_weights / offline_weights.sum() * 0.5
                # Uniform weights for online, summing to 0.5
                online_is_weights = torch.full((online_size,), 0.5 / online_size, device=args.device)
                # Concatenate: [offline, online]
                combined_weights = torch.cat([offline_is_weights, online_is_weights], dim=0)
            else:
                # No priority replay: use None (reverts to standard MSE/mean)
                combined_weights = None

            # Update agent with weights
            critic_info = agent.update_critic(batch, weights=combined_weights)
            actor_info = agent.update_actor(batch, weights=combined_weights)
            temp_info = agent.update_temperature(actor_info['entropy'])

            # Compute ensemble advantages with LCB
            with torch.no_grad():
                q_batch = critic_info['q_values']  # (num_qs=10, batch, 1)
                q_policy = actor_info['q_values_policy']  # (num_qs=10, batch, 1)

                # Advantage per Q network
                advantages = q_batch - q_policy  # (10, batch, 1)

                # LCB across ensemble
                advantage_mean = advantages.mean(dim=0)
                advantage_std = advantages.std(dim=0)
                advantage_lcb = advantage_mean - args.advantage_beta * advantage_std

            # Logging
            if total_env_steps % args.log_interval == 0:
                log_dict = {
                    'train/critic_loss': critic_info['critic_loss'],
                    'train/q_mean': critic_info['q_mean'],
                    'train/actor_loss': actor_info['actor_loss'],
                    'train/entropy': actor_info['entropy'],
                    'train/alpha': temp_info['alpha'],
                    'train/temp_loss': temp_info['temp_loss'],
                    'train/advantage_mean': advantage_mean.mean().item(),
                    'train/advantage_std': advantage_std.mean().item(),
                    'train/advantage_lcb': advantage_lcb.mean().item(),
                    'train/env_steps': total_env_steps,
                    # Density network metrics
                    'train/density_loss': density_info['density_loss'],
                    'train/offline_weight': density_info['offline_weight'],
                    'train/online_weight': density_info['online_weight'],
                }
                # Log priority replay metrics if enabled
                if args.priority_alpha > 0:
                    log_dict['train/priority_beta'] = offline_buffer._get_beta(total_env_steps)
                    log_dict['train/max_priority'] = offline_buffer.max_priority
                wandb.log(log_dict, step=total_env_steps)

        # Evaluation and checkpointing
        if total_env_steps % args.eval_interval == 0:
            eval_metrics = evaluate(eval_env, agent, num_episodes=10, device=args.device)
            wandb.log(eval_metrics, step=total_env_steps)

            # Save checkpoint at the evaluation step
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{total_env_steps}.pt"
            checkpoint = {
                'step': total_env_steps,  # Saved at this step
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'target_critic_state_dict': agent.target_critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'log_alpha': agent.log_alpha,
                'alpha_optimizer': agent.alpha_optimizer.state_dict(),
                'online_buffer': online_buffer.state_dict(),
                'offline_buffer': offline_buffer.state_dict(),  # Include priorities
                'density_state_dict': density_net.state_dict(),
                'density_optimizer': density_net.optimizer.state_dict(),
                'args': vars(args),
                'wandb_run_id': wandb.run.id,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")

            # Keep only the last checkpoint to save space
            for old_ckpt in sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))[:-1]:
                old_ckpt.unlink()

    print("Training complete!")


if __name__ == '__main__':
    main()
