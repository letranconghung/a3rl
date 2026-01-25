import minari

if __name__ == "__main__":
    # List all MuJoCo datasets from HuggingFace
    datasets = [
        
    ]

    # for name in datasets:
    #     print(f"Downloading {name}...")
    #     minari.download_dataset(f"hf://farama-minari/{name}")
    #     print(f"Done: {name}\n")

    print(' '.join([f"\"{d}\"" for d in datasets if "medium" in d]))