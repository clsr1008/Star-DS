"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))
    # 从一个可能包含多次 \boxed{...} 的解答文本里，提取最后一个盒住的答案，并去掉 \boxed{} 外壳
    # 在 deepscaler 数据集中未被使用，应该适用于 math 数据集


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, data_source: str) -> Optional[Dict[str, Any]]:
        # example: 原始数据集（raw dataset）的样本（字典结构），期望至少包含键 'problem' 和 'answer'
        # 原始数据集 DeepScaleR-Preview-Dataset 在huggingface的链接 https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
        question = example.pop('problem') # 从 example 里取出题干并删除该键
        instruction = "Let's think step by step and output the final answer within \\boxed{}." # 把“链式思考+用 \boxed{} 给出最终答案”的格式要求贴到题干后面
        question = f"{question} {instruction}" # 拼接形成最终喂给模型的 prompt
        answer = example.pop('answer') # 取出标准答案

        data = { # 构造标准化样本结构
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
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('train/deepscaler'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    train_datasets = [TrainDataset.DEEPSCALER] # 从库里面调用加载 raw dataset，可以在huggingface上查看
    # 应该是 rllm 库 https://github.com/rllm-org/rllm/tree/main （但是目前还没有装上）
    train_dataset = load_dataset(train_datasets[0])
    test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    # 测试集也是用类似的方式加载
    
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = [] # 声明并初始化一个空列表，里面会装统一格式化后的样本字典
    process_fn = make_map_fn('train') # 返回一个闭包函数 process_fn
    for idx, example in enumerate(train_dataset): # 枚举训练数据集 train_dataset 的每一个样本
        processed_example = process_fn(example, idx, "deepscaler") # 用刚才拿到的 process_fn 处理样本
        if processed_example is not None:
            train_data.append(processed_example)

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, test_dataset.value.lower())
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data) # 把 Python 列表转成 pandas DataFrame
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet')) # 把 DataFrame 保存为 Parquet 文件
    # data/train/one_shot_rlvr/dsr_sub.parquet 就是经过这样处理后的数据集

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)