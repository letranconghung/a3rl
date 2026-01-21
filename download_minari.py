import minari

if __name__ == "__main__":
    # List all MuJoCo datasets from HuggingFace
    datasets = [
        "mujoco/hopper/expert-v0", "mujoco/hopper/simple-v0", "mujoco/hopper/medium-v0",
        "mujoco/halfcheetah/simple-v0", "mujoco/halfcheetah/expert-v0", "mujoco/halfcheetah/medium-v0",
        "mujoco/walker2d/simple-v0", "mujoco/walker2d/medium-v0", "mujoco/walker2d/expert-v0",
        "mujoco/ant/medium-v0", "mujoco/ant/expert-v0", "mujoco/ant/simple-v0",
        "mujoco/pusher/expert-v0", "mujoco/pusher/medium-v0",
        "mujoco/invertedpendulum/expert-v0", "mujoco/invertedpendulum/medium-v0",
        "mujoco/inverteddoublependulum/expert-v0", "mujoco/inverteddoublependulum/medium-v0",
        "mujoco/swimmer/medium-v0", "mujoco/swimmer/expert-v0",
        "mujoco/humanoidstandup/medium-v0", "mujoco/humanoidstandup/expert-v0", "mujoco/humanoidstandup/simple-v0",
        "mujoco/reacher/expert-v0", "mujoco/reacher/medium-v0",
        "mujoco/humanoid/medium-v0", "mujoco/humanoid/simple-v0", "mujoco/humanoid/expert-v0",
    ]

    # for name in datasets:
    #     print(f"Downloading {name}...")
    #     minari.download_dataset(f"hf://farama-minari/{name}")
    #     print(f"Done: {name}\n")

    print(' '.join([f"\"{d}\"" for d in datasets if "medium" in d]))