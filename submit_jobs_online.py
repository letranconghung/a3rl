#!/usr/bin/env python3
"""Submit SLURM jobs for online-only training (pure SAC vs online A3RL)."""

import argparse
import json
import subprocess
from itertools import product
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for online-only training")
    parser.add_argument("--dataset_ids", type=str, nargs="+", required=True,
                       help="Dataset IDs (used only to recover environment)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds (default: 0, 1, 2)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without submitting")
    parser.add_argument("--max_concurrent", type=int, default=None,
                       help="Maximum number of concurrent array tasks (throttle). None = no limit")
    parser.add_argument("--version", type=str, default="v01",
                       help="Version prefix for WandB group name (default: v01)")
    parser.add_argument("--algorithm", type=str, nargs="+", default=["a3rl"],
                       choices=["pure", "a3rl"],
                       help="Algorithm(s) to run (default: a3rl)")
    parser.add_argument("--advantage_lambda", type=float, nargs="+", default=[1.0],
                       help="Advantage lambda values (default: 1.0, only used for a3rl)")
    parser.add_argument("--advantage_beta", type=float, nargs="+", default=[-0.2],
                       help="Advantage beta (UCB coefficient) values (default: -0.2, only used for a3rl)")

    args = parser.parse_args()

    # Generate all combinations
    # For pure SAC: only dataset and seed matter
    # For online A3RL: dataset, seed, and advantage params
    configs = []
    if len(args.algorithm) > 1:
        raise ValueError("More than 1 algorithm supplied!")

    for algo in args.algorithm:
        if algo == "pure":
            # Pure SAC doesn't use advantage params
            for dataset_id, seed in product(args.dataset_ids, args.seeds):
                configs.append((algo, dataset_id, seed, None, None))
        else:  # a3rl
            for dataset_id, seed, adv_lambda, adv_beta in product(
                args.dataset_ids, args.seeds,
                args.advantage_lambda, args.advantage_beta
            ):
                configs.append((algo, dataset_id, seed, adv_lambda, adv_beta))

    # Create logs directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    logs_dir = Path(f"logs/{timestamp}")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build complete config data for all jobs
    configs_data = []
    for algo, dataset_id, seed, adv_lambda, adv_beta in configs:
        dataset_name = dataset_id.replace('/', '_')

        if algo == "pure":
            param_str = "pure_sac"
        else:
            param_str = f"online_a3rl_l{adv_lambda}_b{adv_beta}"

        wandb_group = f"{args.version}_{dataset_name}_{param_str}"
        run_id = f"{args.version}_{dataset_name}_seed{seed}_{param_str}_{timestamp}"

        configs_data.append({
            'algo': algo,
            'dataset_id': dataset_id,
            'seed': seed,
            'adv_lambda': adv_lambda,
            'adv_beta': adv_beta,
            'run_id': run_id,
            'wandb_group': wandb_group,
            'wandb_version': args.version
        })

    # Write config file
    config_file = logs_dir / f"configs_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(configs_data, f, indent=2)

    # Build array job script
    # Create compact job name
    ds_names = ','.join([d.split('/')[1]+'-'+d.split('/')[-1][:-3] for d in args.dataset_ids])
    algo_str = ','.join(args.algorithm)

    if args.algorithm[0] == "pure":
        job_base_name = f"{args.version}||{algo_str}||ds={ds_names}"
    else:
        job_base_name = f"{args.version}||{algo_str}||ds={ds_names}||l={','.join([f'{l:.2f}' for l in args.advantage_lambda])}||b={','.join([f'{b:.2f}' for b in args.advantage_beta])}"

    throttle_str = f"%{args.max_concurrent}" if args.max_concurrent else ""
    array_range = f"0-{len(configs_data) - 1}"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_base_name}
#SBATCH --output={logs_dir}/%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array={array_range}{throttle_str}

# Read config for this array task
CONFIG_FILE={config_file.absolute()}
TASK_ID=$SLURM_ARRAY_TASK_ID

conda activate a3rl_jax
cd /share/data/ripl/hung/a3rl_workspace/a3rl

# Use Python to read config and execute training
python -c "
import json
import subprocess
import sys

with open('$CONFIG_FILE') as f:
    configs = json.load(f)

config = configs[$TASK_ID]

# Build training command
cmd = [
    'python', 'train_online.py',
    '--algorithm', config['algo'],
    '--dataset_id', config['dataset_id'],
    '--seed', str(config['seed']),
    '--run_id', config['run_id'],
    '--wandb_group', config['wandb_group'],
    '--wandb_version', config['wandb_version']
]

# Add a3rl-specific params
if config['algo'] == 'a3rl' and config['adv_lambda'] is not None:
    cmd.extend([
        '--advantage_lambda', str(config['adv_lambda']),
        '--advantage_beta', str(config['adv_beta'])
    ])

# Run training
subprocess.run(cmd, check=True)
"
"""

    if args.dry_run:
        print("DRY RUN - Array job configuration:\n")
        print(f"Config file: {config_file}")
        print(f"Array job name: {job_base_name}")
        print(f"Array range: {array_range}{throttle_str}")
        print(f"Total tasks: {len(configs_data)}")
        if args.max_concurrent:
            print(f"Max concurrent: {args.max_concurrent}")
        print(f"All configurations:")
        for i, cfg in enumerate(configs_data):
            print(f"\n  Task {i}:")
            print(f"    Algorithm: {cfg['algo']}, Dataset: {cfg['dataset_id']}, Seed: {cfg['seed']}")
            if cfg['algo'] == 'a3rl':
                print(f"    Params: lambda={cfg['adv_lambda']}, beta={cfg['adv_beta']}")
            print(f"    Run ID: {cfg['run_id']}")
        print(f"\nLogs will be written to: {logs_dir}/<job_id>_<task_id>.out")
    else:
        # Submit array job
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✓ Submitted array job: {job_base_name}")
            print(f"  Job ID: {job_id}")
            print(f"  Tasks: {len(configs_data)} (array {array_range}{throttle_str})")
            print(f"  Config: {config_file}")
            print(f"  Logs: {logs_dir}/<job_id>_<task_id>.out")
        else:
            print(f"✗ Failed to submit array job: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
