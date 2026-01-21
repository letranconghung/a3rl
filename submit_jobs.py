#!/usr/bin/env python3
"""Simple script to submit multiple SLURM jobs for a3rl training with automatic checkpointing."""

import argparse
import subprocess
from itertools import product
from datetime import datetime
from pathlib import Path


def create_dataset_name(dataset_spec):
    """Create a readable name from dataset specification for file/group naming."""
    if ',' in dataset_spec or ':' in dataset_spec:
        # Mixed dataset
        parts = []
        for part in dataset_spec.split(','):
            if ':' in part:
                ds, pct = part.rsplit(':', 1)
                parts.append(f"{ds.split('/')[-1]}p{int(float(pct)*100)}")
            else:
                parts.append(f"{part.split('/')[-1]}p100")
        return "mixed_" + "_".join(parts)
    else:
        # Single dataset
        return dataset_spec


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for a3rl training")
    parser.add_argument("--dataset_ids", type=str, nargs="+", required=True,
                       help="Dataset specs. Single: 'dataset_id' or mixed: 'dataset1:0.5,dataset2:1.0'")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds (default: 0, 1, 2)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without submitting")
    parser.add_argument("--num_chains", type=int, default=1,
                       help="Number of chained jobs (default: 3 for 8hr limit)")
    parser.add_argument("--time_limit", type=str, default="07:45:00",
                       help="Time limit per job in HH:MM:SS format (default: 07:45:00)")
    parser.add_argument("--version", type=str, default="v01",
                       help="Version prefix for WandB group name (default: v01)")
    parser.add_argument("--algorithm", type=str, nargs="+", default=["a3rl"],
                       choices=["rlpd", "a3rl"],
                       help="Algorithm(s) to run (default: a3rl)")
    parser.add_argument("--priority_alpha", type=float, nargs="+", default=[0.2],
                       help="Priority alpha values (default: 0.6, only used for a3rl)")
    parser.add_argument("--advantage_lambda", type=float, nargs="+", default=[1.0],
                       help="Advantage lambda values (default: 1.0, only used for a3rl)")
    parser.add_argument("--advantage_beta", type=float, nargs="+", default=[1.0],
                       help="Advantage beta (UCB coefficient) values (default: 1.0, only used for a3rl)")

    args = parser.parse_args()

    # Generate all combinations
    # For RLPD: only dataset and seed matter
    # For A3RL: dataset, seed, and priority params
    configs = []
    for algo in args.algorithm:
        if algo == "rlpd":
            # RLPD doesn't use priority params
            for dataset_id, seed in product(args.dataset_ids, args.seeds):
                configs.append((algo, dataset_id, seed, None, None, None))
        else:  # a3rl
            for dataset_id, seed, p_alpha, adv_lambda, adv_beta in product(
                args.dataset_ids, args.seeds,
                args.priority_alpha, args.advantage_lambda, args.advantage_beta
            ):
                configs.append((algo, dataset_id, seed, p_alpha, adv_lambda, adv_beta))

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")

    # Create logs directory

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    logs_dir = Path(f"logs/{timestamp}")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for algo, dataset_id, seed, p_alpha, adv_lambda, adv_beta in configs:
        # Create readable dataset name for naming
        dataset_name = create_dataset_name(dataset_id)

        # Create unique run_id for this training run
        if algo == "rlpd":
            param_str = "rlpd"
        else:
            param_str = f"a3rl_a{p_alpha}_l{adv_lambda}_b{adv_beta}"

        # Group name: version_dataset_params (no seed, no timestamp)
        wandb_group = f"{args.version}_{dataset_name.replace('/', '_')}_{param_str}"
        run_id = f"{dataset_name.replace('/', '_')}_seed{seed}_{param_str}_{timestamp}"

        job_base_name = f"{algo}_{dataset_name.replace('/', '_')}_s{seed}_{param_str}"

        if args.dry_run:
            print(f"Training run: {run_id}")
            print(f"  Algorithm: {algo}, Dataset: {dataset_id}, Seed: {seed}")
            if algo == "a3rl":
                print(f"  Params: alpha={p_alpha}, lambda={adv_lambda}, beta={adv_beta}")
            print(f"  WandB group: {wandb_group}")
            print(f"  Chained jobs: {args.num_chains}")

        previous_job_id = None

        for chain_idx in range(args.num_chains):
            job_name = f"{job_base_name}_p{chain_idx + 1}"

            # Build the training command
            train_cmd = (
                f"python train.py "
                f"--algorithm {algo} "
                f"--dataset_id {dataset_id} "
                f"--seed {seed} "
                f"--run_id {run_id} "
                f"--wandb_group {wandb_group} "
                f"--wandb_version {args.version}"
            )

            # Add priority params only for a3rl
            if algo == "a3rl":
                train_cmd += (
                    f" --priority_alpha {p_alpha} "
                    f"--advantage_lambda {adv_lambda} "
                    f"--advantage_beta {adv_beta}"
                )

            # Add resume flag for continuation jobs
            if chain_idx > 0:
                checkpoint_path = f"checkpoints/{run_id}/checkpoint_step_*.pt"
                train_cmd += f" --resume_from $(ls -t {checkpoint_path} 2>/dev/null | head -1)"

            # Build sbatch script using heredoc style
            dependency_line = f"#SBATCH --dependency=afterany:{previous_job_id}" if previous_job_id else ""

            sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{timestamp}_{job_name}_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time={args.time_limit}
{dependency_line}

conda activate a3rl_jax
cd /share/data/ripl/hung/a3rl_workspace/a3rl
{train_cmd}
"""

            if args.dry_run:
                dep_str = f" (depends on {previous_job_id})" if previous_job_id else ""
                print(f"  Job {chain_idx + 1}/{args.num_chains}: {job_name}{dep_str}")
                print(f"    {train_cmd}")
            else:
                # Submit script via stdin to sbatch
                result = subprocess.run(
                    ["sbatch"],
                    input=sbatch_script,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # Extract job ID from sbatch output (format: "Submitted batch job 12345")
                    job_id = result.stdout.strip().split()[-1]
                    previous_job_id = job_id
                    dep_str = f" (depends on previous)" if chain_idx > 0 else ""
                    print(f"✓ Submitted {job_name}: Job ID {job_id}{dep_str}")
                else:
                    print(f"✗ Failed {job_name}: {result.stderr.strip()}")
                    break  # Don't submit remaining chain if one fails

        if args.dry_run:
            print(f"Total training runs: {len(configs)}")
            print(f"Jobs per run: {args.num_chains} (chained)")
            print(f"Time limit per job: {args.time_limit}")
            print(f"Total SLURM jobs: {len(configs) * args.num_chains}\n")
            print()


if __name__ == "__main__":
    main()
