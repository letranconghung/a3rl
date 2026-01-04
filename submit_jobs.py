#!/usr/bin/env python3
"""Simple script to submit multiple SLURM jobs for a3rl training with automatic checkpointing."""

import argparse
import subprocess
from itertools import product
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for a3rl training")
    parser.add_argument("--dataset_ids", type=str, nargs="+", required=True,
                       help="Dataset IDs (e.g., 'mujoco/halfcheetah/medium-v0')")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds (default: 0, 1, 2)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without submitting")
    parser.add_argument("--num_chains", type=int, default=3,
                       help="Number of chained jobs (default: 3 for 8hr limit)")

    args = parser.parse_args()

    # Generate all combinations
    configs = list(product(args.dataset_ids, args.seeds))

    print(f"Total training runs: {len(configs)}")
    print(f"Jobs per run: {args.num_chains} (chained)")
    print(f"Total SLURM jobs: {len(configs) * args.num_chains}\n")

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id, seed in configs:
        # Create unique run_id for this training run
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_id = f"{dataset_id.replace('/', '_')}_seed{seed}_{timestamp}"

        job_base_name = f"a3rl_{dataset_id.replace('/', '_')}_s{seed}"

        if args.dry_run:
            print(f"Training run: {run_id}")
            print(f"  Dataset: {dataset_id}, Seed: {seed}")
            print(f"  Chained jobs: {args.num_chains}")

        previous_job_id = None

        for chain_idx in range(args.num_chains):
            job_name = f"{job_base_name}_p{chain_idx + 1}"

            # Build the training command
            train_cmd = (
                f"python train.py "
                f"--dataset_id {dataset_id} "
                f"--seed {seed} "
                f"--run_id {run_id}"
            )

            # Add resume flag for continuation jobs
            if chain_idx > 0:
                checkpoint_path = f"checkpoints/{run_id}/checkpoint_step_*.pt"
                train_cmd += f" --resume_from $(ls -t {checkpoint_path} 2>/dev/null | head -1)"

            # Build sbatch script using heredoc style
            dependency_line = f"#SBATCH --dependency=afterany:{previous_job_id}" if previous_job_id else ""

            sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
{dependency_line}

conda activate a3rl
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
            print()


if __name__ == "__main__":
    main()
