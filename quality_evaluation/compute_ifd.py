import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import argparse
from tqdm import tqdm


class IFDEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        self.model.eval()
        print(f"[Init] Loaded model {model_name} on {device}")

    def compute_nll(self, context: str, continuation: str) -> float:
        """
        计算 continuation 在 context 条件下的平均 NLL
        """
        # 编码 context
        context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.model.device).long()
        # 编码 continuation
        with self.tokenizer.as_target_tokenizer():
            cont_ids = self.tokenizer(continuation, return_tensors="pt").input_ids.to(self.model.device).long()

        # 拼接 input_ids
        input_ids = torch.cat([context_ids, cont_ids], dim=1)
        # labels: context 部分 mask 为 -100
        labels_ids = torch.cat([torch.full_like(context_ids, -100), cont_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels_ids)
            nll = outputs.loss.item()  # 平均 NLL

        return nll

    def compute_ifd(self, question: str, answer: str) -> float:
        """
        计算 IFD(Q, A) = s(A|Q) / s(A)
        """
        try:
            nll_cond = self.compute_nll(f"Q: {question}\nA:", answer)  # s(A|Q)
            nll_direct = self.compute_nll("", answer)  # s(A)
            return nll_cond / nll_direct
        except Exception as e:
            print(f"[Error] Failed IFD calc: {e}")
            return None


def main(args):
    evaluator = IFDEvaluator(model_name=args.model_name, device=args.device)

    # 读取 parquet 文件
    df = pd.read_parquet(args.data_path)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_index = row["extra_info"]["index"]
        question_raw = row["prompt"][0]["content"]
        # 去掉固定结尾提示
        question_clean = question_raw.replace(
            "Let's think step by step and output the final answer within \\boxed{}.", ""
        ).strip()
        # 加上直接输出答案的引导
        question = question_clean + " Please directly output the final answer."
        # 答案前面加前缀
        answer = "The answer is " + row["reward_model"]["ground_truth"]

        try:
            ifd_score = evaluator.compute_ifd(question, answer)
        except Exception as e:
            print(f"[Error] Sample {sample_index} failed: {e}")
            ifd_score = None

        results.append({
            "index": int(sample_index),
            "ifd": ifd_score
        })

    # 保存到 json 文件
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IFD for each sample in a parquet file.")
    parser.add_argument("--data_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full.parquet",
                        help="Input parquet file path.")
    parser.add_argument("--output_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_ifd.json",
                        help="Output json file path.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HF model name for IFD calculation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")

    args = parser.parse_args()
    main(args)
