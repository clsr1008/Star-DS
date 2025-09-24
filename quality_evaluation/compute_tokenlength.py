import torch
from transformers import AutoTokenizer
import pandas as pd
import json
import argparse
from tqdm import tqdm


class TokenLengthEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[Init] Loaded tokenizer {model_name}")

    def compute_token_length(self, text: str) -> int:
        """计算单个文本的 token length"""
        encodings = self.tokenizer(text, return_tensors="pt")
        return encodings["input_ids"].size(1)


def main(args):
    evaluator = TokenLengthEvaluator(model_name=args.model_name)

    # 读取 parquet 文件
    df = pd.read_parquet(args.data_path)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_index = row["extra_info"]["index"]
        question = row["prompt"][0]["content"]
        answer = row["reward_model"]["ground_truth"]

        text = f"Q: {question} A: {answer}"

        try:
            token_len = evaluator.compute_token_length(text)
        except Exception as e:
            print(f"[Error] Sample {sample_index} failed: {e}")
            token_len = None

        results.append({
            "index": int(sample_index),
            "token_length": token_len
        })

    # 保存到 json 文件
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute token length for each sample in a parquet file.")
    parser.add_argument("--data_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full.parquet",
                        help="Input parquet file path.")
    parser.add_argument("--output_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_tokenlength.json",
                        help="Output json file path.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="HF tokenizer model name.")

    args = parser.parse_args()
    main(args)
