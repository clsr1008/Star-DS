"""
Script to prepare Hendrycks MATH training dataset for DeepScaler-style training.

This script processes the local Hendrycks MATH dataset into a standardized format.
It loads all training parquet files from the 7 categories, extracts final boxed answers,
adds instruction prompts, and saves the combined data as a single parquet file.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string."""
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process Hendrycks MATH examples."""
    def process_fn(example: Dict[str, Any], idx: int, data_source: str) -> Optional[Dict[str, Any]]:
        question = example.get('problem', "").strip()
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"

        # Extract final answer from solution
        raw_solution = example.get('solution', "")
        answer = extract_solution(raw_solution)

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                # 'level': example.get('level', None),
                # 'type': example.get('type', None)
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Hendrycks MATH dataset')
    parser.add_argument('--data_dir', default='data/hendrycks_math',
                        help='Directory containing Hendrycks MATH category folders')
    parser.add_argument('--local_dir', default='data/train/rlvr',
                        help='Local directory to save processed dataset')
    parser.add_argument('--hdfs_dir', default=None,
                        help='Optional HDFS directory to copy dataset to')
    args = parser.parse_args()

    data_dir = args.data_dir
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    makedirs(local_dir, exist_ok=True)

    # All 7 category folders
    categories = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')

    idx_counter = 0
    for cat in categories:
        train_path = os.path.join(data_dir, cat, 'train-00000-of-00001.parquet')
        if not os.path.exists(train_path):
            print(f"[WARN] Missing file: {train_path}")
            continue
        df = pd.read_parquet(train_path)
        for _, row in df.iterrows():
            processed_example = process_fn(row.to_dict(), idx_counter, f"hendrycks_math_{cat}")
            if processed_example is not None:
                train_data.append(processed_example)
                idx_counter += 1

    # Save merged training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'math_full.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
