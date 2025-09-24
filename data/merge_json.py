import os
import json
import argparse
from glob import glob

def merge_json_files(input_pattern, output_file):
    all_results = {}

    # 找到所有匹配的 json 文件
    files = sorted(glob(input_pattern))
    print(f"Found {len(files)} files to merge.")

    for fpath in files:
        print(f"[Loading] {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for record in data:
                idx = record["index"]
                # 如果重复，则后面的覆盖前面的
                all_results[idx] = record

    # 转成列表并按 index 排序
    merged_results = sorted(all_results.values(), key=lambda x: x["index"])

    # 保存合并后的文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Merged {len(files)} files into {output_file}, total {len(merged_results)} unique samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON result files into one.")
    parser.add_argument("--input_pattern", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_scores_*.json",
                        help="Glob pattern to match input JSON files.")
    parser.add_argument("--output_file", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_scores_merged.json",
                        help="Path to save merged JSON file.")
    args = parser.parse_args()

    merge_json_files(args.input_pattern, args.output_file)
