import pandas as pd
import os
import random
import argparse


def sample_random_from_parquet(parquet_file_path, num_samples, output_dir=None, seed=0):
    """
    Randomly select samples from a given parquet file and save as a new parquet file.

    Args:
        parquet_file_path: str, path to input parquet file
        num_samples: int, number of samples to select
        output_dir: str, directory to save output file (default: same as input)
        seed: int, random seed for reproducibility
    """
    # Load parquet dataset
    print(f"Loading dataset from {parquet_file_path} ...")
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
    total = len(df)
    print(f"Total records in dataset: {total}")

    # Ensure num_samples does not exceed total
    num_samples = min(num_samples, total)
    print(f"Randomly selecting {num_samples} samples ...")

    # Random sampling
    sampled_df = df.sample(n=num_samples, random_state=seed)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(parquet_file_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(parquet_file_path))[0].split("_")[0]
    output_file = os.path.join(output_dir, f"{base_name}_random{num_samples}.parquet")

    # Save sampled dataset
    sampled_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Saved sampled dataset to {output_file}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample data from a parquet file")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to select")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output parquet file (default: same as input)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    sample_random_from_parquet(args.input, args.num, args.output_dir, args.seed)
