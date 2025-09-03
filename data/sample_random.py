import pandas as pd
import os
import random
import argparse


def sample_random_from_parquet(parquet_file_path, num_samples, output_dir=None):
    """
    从指定 parquet 文件中随机选择样本，并保存为新 parquet 文件

    参数:
        parquet_file_path: str, 输入的 parquet 文件路径
        num_samples: int, 要抽取的样本数量
        output_dir: str, 输出目录（默认与输入文件相同）
    """
    # 加载 parquet 数据
    print(f"Loading dataset from {parquet_file_path} ...")
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
    total = len(df)
    print(f"Total records in dataset: {total}")

    # 防止样本数超过总数
    num_samples = min(num_samples, total)
    print(f"Randomly selecting {num_samples} samples ...")

    # 随机采样
    sampled_df = df.sample(n=num_samples, random_state=42)

    # 确定输出路径
    if output_dir is None:
        output_dir = os.path.dirname(parquet_file_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(parquet_file_path))[0].split("_")[0]
    output_file = os.path.join(output_dir, f"{base_name}_random{num_samples}.parquet")

    # 保存文件
    sampled_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Saved sampled dataset to {output_file}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample data from a parquet file")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to select")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output parquet file (default: same as input)")

    args = parser.parse_args()

    sample_random_from_parquet(args.input, args.num, args.output_dir)
