import numpy

runs = {
    "mujoco/halfcheetah/simple-v0": "a3rl_v16_mujoco_halfcheetah_simple-v0_a3rl_a0.2_l1.0_b-0.2",
    "mujoco/halfcheetah/medium-v0": "a3rl_v15_mujoco_halfcheetah_medium-v0_a3rl_a0.6_l1.0_b-0.2",
    # "mujoco/hopper/medium-v0": "a3rl_v15_mujoco_hopper_medium-v0_a3rl_a0.2_l1.0_b-0.2",
    # "mujoco/hopper/expert-v0": "a3rl_v15_mujoco_hopper_expert-v0_a3rl_a0.6_l0.5_b-0.2",
    "mujoco/humanoid/medium-v0": "a3rl_v16_mujoco_humanoid_medium-v0_a3rl_a1.0_l1.5_b-0.2",
    "mujoco/humanoid/expert-v0": "a3rl_v15_mujoco_humanoid_expert-v0_a3rl_a0.6_l1.0_b-0.2",
    # "mujoco/humanoidstandup/medium-v0": "a3rl_v15_mujoco_humanoidstandup_medium-v0_a3rl_a0.6_l0.5_b-0.2",
    "D4RL/antmaze/umaze-v1": "a3rl_v16_D4RL_antmaze_umaze-v1_a3rl_a0.2_l1.0_b-0.2",
    "D4RL/pen/expert-v2": "a3rl_v16_D4RL_pen_expert-v2_a3rl_a0.2_l1.0_b-0.2",
    "D4RL/relocate/expert-v2": "a3rl_v15_D4RL_relocate_expert-v2_a3rl_a0.2_l1.0_b-0.2",
    "D4RL/door/expert-v2": "a3rl_v16_D4RL_door_expert-v2_a3rl_a0.2_l1.0_b-0.2",
    
}


cmd = 'python submit_jobs.py --dataset_ids DATASET_ID --seeds 0 1 2 --algorithm a3rl --priority_alpha ALPHA --advantage_lambda LAMBDA --advantage_beta BETA --version VERSION'
if __name__ == "__main__":
    for env, run in runs.items():
        arr = run.split('_')
        a = float(arr[-3][1:])
        l = float(arr[-2][1:])
        b = float(arr[-1][1:])
        # print(f'run {run} has {a=}, {l=}, {b=}.')
        

        # density ablation
        print()
        print(cmd.replace("VERSION","abl_v17").replace("DATASET_ID", env).replace("ALPHA", str(0)).replace("LAMBDA", str(l)).replace("BETA", str(b)))

        # advantage ablation
        print()
        print(cmd.replace("VERSION","abl_v17").replace("DATASET_ID", env).replace("ALPHA", str(a)).replace("LAMBDA", str(0)).replace("BETA", str(b)))
        
        # beta ablation
        print()
        print(cmd.replace("VERSION","abl_v17").replace("DATASET_ID", env).replace("ALPHA", str(a)).replace("LAMBDA", str(l)).replace("BETA", str(0)))
        
    print()
        