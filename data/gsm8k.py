"""
Script to prepare GSM8K training dataset for DeepScaler-style training.

This script processes the GSM8K dataset into a standardized format.
It loads a jsonlines file, extracts final answers, adds instruction prompts,
and saves the processed data as a parquet file.
"""

import argparse
import os
import json
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def extract_final_answer(answer_str: str) -> str:
    """Extract the final answer from GSM8K's answer field (after '####')."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def make_map_fn(split: str):
    """Create a mapping function to process GSM8K examples."""
    def process_fn(example: Dict[str, Any], data_source: str) -> Optional[Dict[str, Any]]:
        question = example.get('question', "").strip()
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"

        raw_answer = example.get('answer', "")
        answer = extract_final_answer(raw_answer)
        index = example.get("idx", None)  # index is taken directly from JSON file

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
                'index': index,
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GSM8K dataset')
    parser.add_argument('--data_path', default='Qwen2.5-Eval/evaluation/data/gsm8k/train.jsonl',
                        help='Path to GSM8K jsonlines file')
    parser.add_argument('--local_dir', default='data/train/rlvr_gsm8k',
                        help='Local directory to save processed dataset')
    parser.add_argument('--hdfs_dir', default=None,
                        help='Optional HDFS directory to copy dataset to')
    args = parser.parse_args()

    data_path = args.data_path
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    makedirs(local_dir, exist_ok=True)

    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')

    print(f"Loading GSM8K from {data_path} ...")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            processed_example = process_fn(example, "gsm8k")
            if processed_example is not None:
                train_data.append(processed_example)

    # Save processed training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    output_file = os.path.join(local_dir, 'gsm8k_full.parquet')
    train_df.to_parquet(output_file)

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"✅ Saved processed dataset to {output_file}")
