import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import math
import argparse
from tqdm import tqdm


class PPLEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        self.model.eval()
        print(f"[Init] Loaded model {model_name} on {device}")

    def compute_ppl(self, text: str) -> float:
        """计算单个文本的 PPL"""
        encodings = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            neg_log_likelihood = outputs.loss.item()
        return math.exp(neg_log_likelihood)


def main(args):
    evaluator = PPLEvaluator(model_name=args.model_name, device=args.device)

    # 读取 parquet 文件
    df = pd.read_parquet(args.data_path)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_index = row["extra_info"]["index"]
        question = row["prompt"][0]["content"]
        answer = row["reward_model"]["ground_truth"]

        text = f"Q: {question} A: {answer}"

        try:
            ppl = evaluator.compute_ppl(text)
        except Exception as e:
            print(f"[Error] Sample {sample_index} failed: {e}")
            ppl = None

        results.append({
            "index": int(sample_index),
            "ppl": ppl
        })

    # 保存到 json 文件
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PPL for each sample in a parquet file.")
    parser.add_argument("--data_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full.parquet",
                        help="Input parquet file path.")
    parser.add_argument("--output_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_ppl.json",
                        help="Output json file path.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HF model name for perplexity calculation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")

    args = parser.parse_args()
    main(args)
