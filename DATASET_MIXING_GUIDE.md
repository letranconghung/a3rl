****# Dataset Mixing Feature Guide

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


# Run logs
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

### jan 25:
**run again on simple, version 15**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/hopper/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" \
  "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/hammer/expert-v2" "D4RL/relocate/expert-v2" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 1 \
  --advantage_beta -0.2 \
  --version a3rl_v15 

**more hyperparameter search on medium and expert, version 15**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/expert-v0" "mujoco/walker2d/expert-v0" "mujoco/hopper/expert-v0" "mujoco/ant/expert-v0" "mujoco/humanoidstandup/expert-v0" "mujoco/humanoid/expert-v0" \
  "mujoco/halfcheetah/medium-v0" "mujoco/walker2d/medium-v0" "mujoco/hopper/medium-v0" "mujoco/ant/medium-v0" "mujoco/humanoidstandup/medium-v0" "mujoco/humanoid/medium-v0" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 0.6 \
  --advantage_lambda 0.5 1 \
  --advantage_beta -0.2 \
  --version a3rl_v15

**RLPD on medium and expert**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/expert-v0" "mujoco/walker2d/expert-v0" "mujoco/hopper/expert-v0" "mujoco/ant/expert-v0" "mujoco/humanoidstandup/expert-v0" "mujoco/humanoid/expert-v0" \
  "mujoco/halfcheetah/medium-v0" "mujoco/walker2d/medium-v0" "mujoco/hopper/medium-v0" "mujoco/ant/medium-v0" "mujoco/humanoidstandup/medium-v0" "mujoco/humanoid/medium-v0" \
  --seeds 0 1 2 --algorithm rlpd \
  --version rlpd_v13

### jan 27:
**run again on simple, version 16**
python submit_jobs.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/walker2d/simple-v0" "mujoco/hopper/simple-v0" "mujoco/ant/simple-v0" "mujoco/humanoidstandup/simple-v0" "mujoco/humanoid/simple-v0" \
  "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/hammer/expert-v2" "D4RL/relocate/expert-v2" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 \
  --advantage_lambda 1 \
  --advantage_beta -0.2 \
  --version a3rl_v16

**run again on weak environments: "mujoco/humanoid/medium-v0" "mujoco/humanoidstandup/expert-v0" "mujoco/hopper/simple-v0"**
python submit_jobs.py \
  --dataset_ids "mujoco/humanoid/medium-v0" "mujoco/humanoidstandup/expert-v0" "mujoco/hopper/simple-v0" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 0.6 1 \
  --advantage_lambda 0.5 1 1.5\
  --advantage_beta -0.2 \
  --version a3rl_v16

### jan 28:
#### part 1
**run on the remaining D4RL tasks**
python submit_jobs.py \
  --dataset_ids "D4RL/pen/human-v2" "D4RL/pen/cloned-v2" \
  "D4RL/kitchen/partial-v2" "D4RL/kitchen/complete-v2" "D4RL/kitchen/mixed-v2" \
  "D4RL/door/human-v2" "D4RL/door/cloned-v2" \
  "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
  "D4RL/relocate/human-v2" "D4RL/relocate/cloned-v2" \
  "D4RL/antmaze/umaze-diverse-v1" "D4RL/antmaze/medium-play-v1" "D4RL/antmaze/medium-diverse-v1" \
  --seeds 0 1 2 --algorithm a3rl \
  --priority_alpha 0.2 0.6 \
  --advantage_lambda 1 1.5 \
  --advantage_beta -0.2 \
  --version a3rl_v16

**run RLPD on these new tasks**
python submit_jobs.py \
  --dataset_ids "D4RL/pen/human-v2" "D4RL/pen/cloned-v2" \
  "D4RL/kitchen/partial-v2" "D4RL/kitchen/complete-v2" "D4RL/kitchen/mixed-v2" \
  "D4RL/door/human-v2" "D4RL/door/cloned-v2" \
  "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
  "D4RL/relocate/human-v2" "D4RL/relocate/cloned-v2" \
  "D4RL/antmaze/umaze-diverse-v1" "D4RL/antmaze/medium-play-v1" "D4RL/antmaze/medium-diverse-v1" \
  --seeds 0 1 2 --algorithm rlpd \
  --version rlpd_v13

#### part 2
**ablation studies on the good environments**
This was generated from `python ablation_alpha,lambda,beta.py`

**ablation studies on TD error, and TD density** 
****
python submit_jobs_td.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" \
  --seeds 0 --algorithm a3rl \
  --priority_alpha 0.2 0 \
  --advantage_lambda 1 \
  --version abl_v16

**more seeds for ablation studies**
ran with seed 1, 2 for ablation_alpha,lambda,beta.py

**ablation studies on TD error, and TD density more seeds**
python submit_jobs_td.py \
  --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" \
  --seeds 1 2 --algorithm a3rl \
  --priority_alpha 0.2 0 \
  --advantage_lambda 1 \
  --version abl_v16

#### part 3
**running PEX**

cd /share/data/ripl/hung/a3rl_workspace/a3rl/PEX

python submit_jobs_pex.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" \
--seeds 0 1 2 --algorithms pex \
--offline_steps 1000001 --online_steps 250001 \
--dry_run

#### part 4
**PEX on more good ones, but shorter offline pretraining**
python submit_jobs_pex.py \
--dataset_ids "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 --algorithms pex \
--offline_steps 500001 --online_steps 250001


#### part 5
**BOORL on the good ones**
python submit_jobs_boorl.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 --offline_steps 500001 --online_steps 250001 --version boorl_v02


#### part 6
**only online A3RL vs online SAC. ablate on more good ones**
python submit_jobs_online.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2  \
--version onlineabl_v01 \
--algorithm a3rl

python submit_jobs_online.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2  \
--version onlineabl_v01 \
--algorithm pure

#### part 7
**run emergency ablation studies again**
python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0  \
--algorithm a3rl \
--version hparamabl_v01 \
--priority_alpha 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0  \
--algorithm a3rl \
--version hparamablbeta_v01 \
--advantage_beta 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0  \
--algorithm a3rl \
--version hparamabllambda_v01 \
--advantage_lambda 0

### feb 02
#### run alpha, beta, lambda ablation studies (again)
python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 \
--algorithm a3rl \
--version abl_alpha_v01 \
--priority_alpha 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 \
--algorithm a3rl \
--version abl_beta_v01 \
--advantage_beta 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 \
--algorithm a3rl \
--version abl_lambda_v01 \
--advantage_lambda 0


#### run BOORL chain
python submit_jobs_boorl.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 \
--offline_version boorl_off_v01 \
--online_version boorl_on_v01


### feb 15
#### run main results on 17 good environments, 7 seeds
python submit_jobs.py \
    --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
    --seeds 3 4 5 6 7 8 9 --algorithm a3rl \
    --priority_alpha 0.2 0.6 \
    --advantage_lambda 1 1.5 \
    --advantage_beta -0.2 \
    --version a3rl_v16

#### run ablation alpha on 17 good environments, 7 seeds
python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 3 4 5 6 7 8 9 \
--algorithm a3rl \
--version abl_alpha_v01 \
--priority_alpha 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 3 4 5 6 7 8 9 \
--algorithm a3rl \
--version abl_beta_v01 \
--advantage_beta 0

python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 3 4 5 6 7 8 9 \
--algorithm a3rl \
--version abl_lambda_v01 \
--advantage_lambda 0


#### run RLPD on 17 good environments, 7 seeds
python submit_jobs.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 3 4 5 6 7 8 9 \
--algorithm rlpd \
--version rlpd_v13


### feb 21
#### run PEX on 17 good environments, 10 seeds
cd /share/data/ripl/hung/a3rl_workspace/a3rl/PEX

python submit_jobs_pex.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 3 4 5 6 7 8 9 --algorithms pex \
--offline_steps 1000001 --online_steps 250001 \
--version pex_v02

#### run BOORL on 17 good environments, 10 seeds
python submit_jobs_boorl.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 3 4 5 6 7 8 9 \
--offline_version boorl_off_v02 \
--online_version boorl_on_v02

#### run PEX again on 17 good environments, 10 seeds. there were some issues with cuda.
python submit_jobs_pex.py \
--dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
--seeds 0 1 2 3 4 5 6 7 8 9 --algorithms pex \
--offline_steps 1000001 --online_steps 250001 \
--version pex_v03


#### re run online BOORL
python submit_jobs_online.py \
    --dataset_ids "D4RL/door/expert-v2" \
    --seeds 5 6 7 8 9 \
    --online_steps 250001 \
    --online_version boorl_on_v02 \
    --exclude_nodes g2, g10, g11, g12, g13, g4, g8, g9, g17, g18, g1, g6, g14, g15, g19, g22

#### rerun ablation density
python submit_jobs.py \
    --dataset_ids "mujoco/halfcheetah/simple-v0" "mujoco/halfcheetah/medium-v0" "mujoco/hopper/medium-v0" "mujoco/hopper/expert-v0" "mujoco/humanoid/medium-v0" "mujoco/humanoid/expert-v0" "mujoco/humanoidstandup/medium-v0" "D4RL/antmaze/umaze-v1" "D4RL/pen/expert-v2" "D4RL/door/expert-v2" "D4RL/relocate/expert-v2" "D4RL/pen/cloned-v2" "D4RL/pen/human-v2" "D4RL/door/cloned-v2" "D4RL/door/human-v2" "D4RL/hammer/cloned-v2" "D4RL/hammer/human-v2" \
    --seeds 0 1 2 --algorithm a3rl \
    --priority_alpha 0 \
    --advantage_lambda 0.6 \
    --advantage_beta 0 \
    --version a3rl_v16
