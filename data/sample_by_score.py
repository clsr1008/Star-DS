import os
import argparse
import pandas as pd
import json
import numpy as np


def sample_by_score(parquet_file_path, score_file_path, score_field, num_samples,
                    mode="top", output_dir=None, print_only=True):
    """
    根据 JSON 文件里的分数字段筛选样本，并在 Parquet 主数据中提取这些样本保存，
    同时打印出 question、answer 和对应分数。

    参数:
        parquet_file_path: str, 主数据 parquet 文件
        score_file_path: str, JSON 文件（包含 index 和对应分数）
        score_field: str, 筛选字段
        num_samples: int, 抽取的样本数
        mode: str, "top" 表示取最高的样本，"bottom" 表示取最低的样本
        output_dir: str, 保存路径
        print_only: bool, 如果 True 只打印不保存 parquet
    """
    print(f"Loading main dataset from {parquet_file_path} ...")
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
    print(f"Total records in dataset: {len(df)}")

    print(f"Loading scores from {score_file_path} ...")
    with open(score_file_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    score_df = pd.DataFrame(scores)  # 转成 DataFrame，里面有 index 和对应分数
    print(f"Loaded {len(score_df)} score records.")

    # 如果是 hybrid，则计算新的字段
    if score_field == "hybrid":
        if not {"uncertainty_score", "reward_variance"}.issubset(score_df.columns):
            raise ValueError("计算 hybrid_score 需要同时存在 'uncertainty_score' 和 'reward_variance' 字段")
        score_df["hybrid"] = score_df["uncertainty_score"] + np.sqrt(score_df["reward_variance"])

    if score_field not in score_df.columns:
        raise ValueError(f"Field '{score_field}' not found in score file. Available fields: {list(score_df.columns)}")

    # 按分数排序
    ascending = True if mode == "bottom" else False
    score_sorted = score_df.sort_values(by=score_field, ascending=ascending)

    # 选出目标 index
    num_samples = min(num_samples, len(score_sorted))
    selected_scores = score_sorted.head(num_samples)
    selected_indices = set(selected_scores["index"].tolist())

    # 在主数据中过滤
    sampled_df = df[df["extra_info"].apply(lambda x: x["index"] in selected_indices)].copy()
    # 把 extra_info["index"] 提取成一列
    sampled_df["index"] = sampled_df["extra_info"].apply(lambda x: x["index"])

    # 🔹 打印 question, answer 和分数
    print(f"\n📊 Top/Bottom {num_samples} samples by '{score_field}' ({mode}):\n")
    merged = sampled_df.merge(selected_scores, on="index")
    for _, row in merged.iterrows():
        question = row["prompt"][0]["content"]
        answer = row["reward_model"]["ground_truth"]
        score = row[score_field]
        print(f"Index={row['extra_info']['index']} | Score={score:.4f}")
        print(f"Q: {question}")
        print(f"A: {answer}\n{'-'*80}")

    # 🔹 如果不是 print_only，则保存 parquet
    if not print_only:
        if output_dir is None:
            output_dir = os.path.dirname(parquet_file_path)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(parquet_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_{score_field}_{mode}{num_samples}.parquet")

        sampled_df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"\n✅ Saved sampled dataset to {output_file}")

    return sampled_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample data from a parquet file by external score JSON")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file (main data)")
    parser.add_argument("--score_file", type=str, required=True, help="Path to JSON file containing scores")
    parser.add_argument("--field", type=str, required=True,
                        choices=["uncertainty_score", "reward_variance", "hybrid", "ifd", "ppl", "score", "token_length"],
                        help="Field name to sort and select samples")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to select")
    parser.add_argument("--mode", type=str, choices=["top", "bottom"], default="top",
                        help="Select top or bottom samples (default: top)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output parquet file (default: same as input)")
    parser.add_argument("--print_only", action="store_true", help="Only print results without saving parquet")

    args = parser.parse_args()
    sample_by_score(args.input, args.score_file, args.field, args.num,
                    args.mode, args.output_dir, print_only=args.print_only)
