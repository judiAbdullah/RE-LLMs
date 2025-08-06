import os
import sys
import argparse
from dotenv import dotenv_values
from datasets import load_from_disk

def load_dataset(dataset_path):
    """Load a dataset from a given path."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    return load_from_disk(dataset_path)

def show_samples(dataset, num_samples=10):
    """Print a specified number of samples from the dataset."""
    for i in range(num_samples):
        for key in dataset[i]:
            print(f"{key}: {dataset[i][key]}")
            print('-'*30)
        print('*'*60)

def main():
    parser = argparse.ArgumentParser(description="Load and display dataset samples.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to load.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to display.")

    args = parser.parse_args()

    # Determine the parent directory and load .env config
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.append(parent_dir)

    config = dotenv_values(os.path.join(parent_dir, ".env"))

    # Map dataset names to paths from the .env configuration
    dataset_paths = {
        "filteredDataJava": config.get("filteredDataJava"),
        "filteredDataPython": config.get("filteredDataPython"),
        "evalDatasetpathJava": config.get("evalDatasetpathJava"),
        "evalDatasetpathPython": config.get("evalDatasetpathPython"),
        # Add more datasets as needed
    }

    # Validate the dataset name
    if args.dataset not in dataset_paths:
        print(f"Error: Dataset '{args.dataset}' not found in configuration.")
        print(f"Available datasets: {', '.join(dataset_paths.keys())}")
        sys.exit(1)

    dataset_path = os.path.join(parent_dir, dataset_paths[args.dataset].lstrip(os.sep))

    try:
        # Load the dataset
        dataset = load_dataset(dataset_path)
        print(f"Loaded dataset '{args.dataset}' from {dataset_path}\n")
        print(dataset)
        print(f"{args.samples} printing")
        # Show samples
        if 'test' in dataset:
            show_samples(dataset['test'], num_samples=args.samples)
        else:
            show_samples(dataset, num_samples=args.samples)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
