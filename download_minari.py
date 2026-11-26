import minari

if __name__ == "__main__":

    # Retrieve the dictionary of remote datasets
    remote_datasets = minari.list_remote_datasets()

    # List only the names (IDs)
    dataset_names = list(remote_datasets.keys())

    datasets = [d for d in dataset_names if 'D4RL' in d]
    for name in datasets:
        print(f"Downloading {name}...")
        minari.download_dataset(f"hf://farama-minari/{name}")
        print(f"Done: {name}\n")

    # print(' '.join([f"\"{d}\"" for d in datasets if "medium" in d]))