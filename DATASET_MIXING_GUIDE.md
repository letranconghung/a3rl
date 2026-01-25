# Dataset Mixing Feature Guide

## Overview
You can now train with mixed offline datasets by specifying multiple dataset IDs with optional sampling percentages.

## Usage

### Single Dataset (unchanged behavior)
```bash
python train.py --dataset_id mujoco/halfcheetah/medium-v0 --algorithm rlpd
```

### Single Dataset with Sampling
Sample only 50% of transitions from a dataset:
```bash
python train.py --dataset_id mujoco/halfcheetah/medium-v0:0.5 --algorithm rlpd
```

### Mixed Datasets
Combine multiple datasets with different sampling percentages:
```bash
# 50% of medium dataset + 100% of expert dataset
python train.py \
  --dataset_id "mujoco/halfcheetah/medium-v0:0.5,mujoco/halfcheetah/expert-v0:1.0" \
  --algorithm rlpd
```


python train.py \
  --dataset_id "mujoco/halfcheetah/medium-v0:0.5,mujoco/halfcheetah/expert-v0:1.0" \
  --algorithm rlpd

```bash
# Multiple datasets, all 100% (default)
python train.py \
  --dataset_id "dataset1,dataset2,dataset3" \
  --algorithm a3rl
```

## Format Specification

### Dataset Spec Format
```
dataset_id[:percentage][,dataset_id[:percentage]...]
```

- `dataset_id`: Minari dataset identifier (e.g., `mujoco/halfcheetah/medium-v0`)
- `percentage`: Optional float between 0.0 and 1.0 (default: 1.0)
  - `1.0` = use all transitions
  - `0.5` = randomly sample 50% of transitions
  - `0.1` = randomly sample 10% of transitions

### Examples
- Single dataset: `mujoco/halfcheetah/medium-v0`
- Half of one dataset: `mujoco/halfcheetah/medium-v0:0.5`
- Two datasets mixed: `mujoco/halfcheetah/medium-v0:0.5,mujoco/halfcheetah/expert-v0:1.0`
- Three datasets: `dataset1:0.3,dataset2:0.5,dataset3:1.0`

## Submit Jobs with Mixed Datasets

### Using submit_jobs.py
```bash
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/medium-v0:0.5,mujoco/halfcheetah/expert-v0:1.0" \
  --seeds 0 1 2 \
  --algorithm rlpd a3rl \
  --dry_run
```

### Multiple mixed dataset combinations
```bash
python submit_jobs.py \
  --dataset_ids \
    "mujoco/halfcheetah/medium-v0" \
    "mujoco/halfcheetah/medium-v0:0.5,mujoco/halfcheetah/expert-v0:0.5" \
    "mujoco/halfcheetah/medium-v0:0.3,mujoco/halfcheetah/expert-v0:0.7" \
  --seeds 0 1 2 \
  --algorithm a3rl
```

"mujoco/halfcheetah/simple-v0:0.98,mujoco/halfcheetah/expert-v0:0.02" "mujoco/ant/simple-v0:0.98,mujoco/ant/expert-v0:0.02" "mujoco/humanoid/simple-v0:0.98,mujoco/humanoid/expert-v0:0.02" "mujoco/humanoidstandup/simple-v0:0.98,mujoco/humanoidstandup/expert-v0:0.02" "mujoco/walker2d/simple-v0:0.98,mujoco/walker2d/expert-v0:0.02" "mujoco/hopper/simple-v0:0.98,mujoco/hopper/expert-v0:0.02"


# CURRENT
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0:0.5,mujoco/halfcheetah/expert-v0:0.5" "mujoco/ant/simple-v0:0.5,mujoco/ant/expert-v0:0.5" "mujoco/humanoid/simple-v0:0.5,mujoco/humanoid/expert-v0:0.5" "mujoco/humanoidstandup/simple-v0:0.5,mujoco/humanoidstandup/expert-v0:0.5" "mujoco/walker2d/simple-v0:0.5,mujoco/walker2d/expert-v0:0.5" "mujoco/hopper/simple-v0:0.5,mujoco/hopper/expert-v0:0.5" "mujoco/hopper/simple-v0" "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" "D4RL/antmaze/umaze-v1" "D4RL/pointmaze/umaze-v2"\
  --seeds 0 1 2 --algorithm rlpd \
  --version rlpd_v13

**The classic ones**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0:0.5,mujoco/halfcheetah/expert-v0:0.5" "mujoco/ant/simple-v0:0.5,mujoco/ant/expert-v0:0.5" "mujoco/humanoid/simple-v0:0.5,mujoco/humanoid/expert-v0:0.5" "mujoco/humanoidstandup/simple-v0:0.5,mujoco/humanoidstandup/expert-v0:0.5" "mujoco/walker2d/simple-v0:0.5,mujoco/walker2d/expert-v0:0.5" "mujoco/hopper/simple-v0:0.5,mujoco/hopper/expert-v0:0.5" \
  "mujoco/hopper/simple-v0" "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" "D4RL/antmaze/umaze-v1" "D4RL/pointmaze/umaze-v2"\
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 0.3 1 1.5 \
  --advantage_beta -0.2 \
  --version a3rl_v13

(sent)

**The unbalanced ones**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0:0.95,mujoco/halfcheetah/expert-v0:0.05" "mujoco/ant/simple-v0:0.95,mujoco/ant/expert-v0:0.05" "mujoco/humanoid/simple-v0:0.95,mujoco/humanoid/expert-v0:0.05" "mujoco/humanoidstandup/simple-v0:0.95,mujoco/humanoidstandup/expert-v0:0.05" "mujoco/walker2d/simple-v0:0.95,mujoco/walker2d/expert-v0:0.05" "mujoco/hopper/simple-v0:0.95,mujoco/hopper/expert-v0:0.05" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 0.3 1 1.5 \
  --advantage_beta -0.2 \
  --version a3rl_v13

python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0:0.95,mujoco/halfcheetah/expert-v0:0.05" "mujoco/ant/simple-v0:0.95,mujoco/ant/expert-v0:0.05" "mujoco/humanoid/simple-v0:0.95,mujoco/humanoid/expert-v0:0.05" "mujoco/humanoidstandup/simple-v0:0.95,mujoco/humanoidstandup/expert-v0:0.05" "mujoco/walker2d/simple-v0:0.95,mujoco/walker2d/expert-v0:0.05" "mujoco/hopper/simple-v0:0.95,mujoco/hopper/expert-v0:0.05" \
  --seeds 0 1 2 --algorithm rlpd \
  --version rlpd_v13
(sent)

**The D4RL antmaze except for umaze ones**
python submit_jobs.py \
  --dataset_ids "D4RL/antmaze/medium-diverse-v1" "D4RL/antmaze/large-play-v1" "D4RL/antmaze/large-diverse-v1" "D4RL/antmaze/umaze-diverse-v1" "D4RL/antmaze/medium-play-v1" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 0.3 1 1.5 \
  --advantage_beta -0.2 \
  --version a3rl_v13

python submit_jobs.py \
  --dataset_ids "D4RL/antmaze/medium-diverse-v1" "D4RL/antmaze/large-play-v1" "D4RL/antmaze/large-diverse-v1" "D4RL/antmaze/umaze-diverse-v1" "D4RL/antmaze/medium-play-v1" \
  --seeds 0 1 2 --algorithm rlpd \
  --version rlpd_v13
(sent)

**Version 14: simple with only lambda 1. Density now normalized to sum.**
python submit_jobs.py \
  --dataset_ids "mujoco/hopper/simple-v0" "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" "D4RL/antmaze/umaze-v1"\
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 1 \
  --advantage_beta -0.2 \
  --version a3rl_v14

**the new environments**
python submit_jobs.py \
  --dataset_ids "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/hammer/expert-v2" "D4RL/relocate/expert-v2" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 1 \
  --advantage_beta -0.2 \
  --version a3rl_v14

python submit_jobs.py \
    --dataset_ids "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/hammer/expert-v2" "D4RL/relocate/expert-v2" \
    --seeds 0 1 2 --algorithm rlpd \
    --version rlpd_v13

**extra hyperparameter search on the promising medium and expert**
python submit_jobs.py \
  --dataset_ids "mujoco/hopper/simple-v0" "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" "D4RL/antmaze/umaze-v1"\
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 1 \
  --advantage_beta -0.2 \
  --version a3rl_v14














## Deprecated
python submit_jobs.py \
  --dataset_ids "mujoco/hopper/medium-v0" "mujoco/halfcheetah/medium-v0" "mujoco/walker2d/medium-v0" "mujoco/ant/medium-v0" "mujoco/swimmer/medium-v0" "mujoco/humanoidstandup/medium-v0" "mujoco/humanoid/medium-v0" \
  --seeds 3 4  --algorithm rlpd \
  --version rlpd_v08 \

python submit_jobs.py \
  --dataset_ids "mujoco/hopper/medium-v0" "mujoco/halfcheetah/medium-v0" "mujoco/walker2d/medium-v0" "mujoco/ant/medium-v0" "mujoco/swimmer/medium-v0" "mujoco/humanoidstandup/medium-v0" "mujoco/humanoid/medium-v0" \
  --seeds 0 1 2 \
  --advantage_lambda 0.1 0.3 1 \
  --algorithm a3rl \
  --version a3rl_v09 \

## How It Works

1. **Parsing**: The `parse_dataset_spec()` function parses the dataset specification string
2. **Loading**: For each dataset:
   - Load all transitions from the Minari dataset
   - If percentage < 1.0, randomly sample that percentage of transitions
3. **Combining**: All sampled transitions are combined into a single offline buffer
4. **Naming**: Mixed datasets get automatic naming for logging:
   - Single: `mujoco/halfcheetah/medium-v0`
   - Mixed: `mixed_medium-v0p50_expert-v0p100`

## Implementation Details

### Modified Files
- `data/buffer.py`: Added `load_mixed_datasets()` method and `sample_percentage` parameter
- `train.py`: Added `parse_dataset_spec()` and dataset loading logic
- `submit_jobs.py`: Added `create_dataset_name()` and updated argument parsing

### Buffer Methods
```python
# Load single dataset with sampling
buffer.load_from_minari(
    dataset_id="mujoco/halfcheetah/medium-v0",
    download=True,
    sample_percentage=0.5
)

# Load mixed datasets
buffer.load_mixed_datasets(
    dataset_specs=[
        ("mujoco/halfcheetah/medium-v0", 0.5),
        ("mujoco/halfcheetah/expert-v0", 1.0)
    ],
    download=True
)
```

## Notes
- Sampling is done **randomly** without replacement
- The random seed is set via `np.random.seed(args.seed)` before loading
- All datasets must be compatible (same observation/action spaces)
- The first dataset is used to recover the environment for online interaction
