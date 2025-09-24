import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import argparse
import numpy as np
import pandas as pd
import json
from typing import List, Tuple
import regex
import time
from tqdm import tqdm
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv


def extract_answer(rollout: str) -> Tuple[str, int]:
    """
    Extract the final answer from a rollout.

    Returns:
        answer (str): the extracted answer
        ans_idx (int): ending index of the answer in the rollout
    """
    # Try extracting the last \boxed{} or \fbox{}
    boxed = last_boxed_only_string(rollout)
    if boxed:
        ans = remove_boxed(boxed).strip()
        ans_idx = rollout.rfind(boxed) + len(boxed)
        return ans, ans_idx

    # Truncate before code snippets
    match = re.search(r"\b(code|python)\b", rollout, flags=re.IGNORECASE)
    if match:
        rollout = rollout[:match.start()]

    # Fallback: last number including \frac{a}{b}
    number_iter = list(re.finditer(
        r"(\\frac\{[+\-]?\d+\}\{[+\-]?\d+\}|[-+]?\d+(?:\.\d+)?)",
        rollout
    ))
    if number_iter:
        last_match = number_iter[-1]
        ans = last_match.group(0)
        ans_idx = last_match.end()

        # Check if number is inside an interval
        right_context = rollout[ans_idx:].lstrip()
        if right_context.startswith((")", "]")):
            left_part = rollout[:last_match.start()]
            left_bracket_pos = max(left_part.rfind("("), left_part.rfind("["))
            if left_bracket_pos != -1:
                interval_text = rollout[left_bracket_pos: ans_idx + 1]
                ans = interval_text
                ans_idx = ans_idx + 1
                return ans, ans_idx

        return ans, ans_idx

    # If nothing found, return empty
    return "", len(rollout)


def is_sequential(steps, mode="step"):
    """
    Check whether steps are sequential starting from 1.

    Args:
        steps: list of step strings
        mode: "step" for "Step N", "num" for numbered list like 1., 2., ...
    """
    numbers = []
    for s in steps:
        if mode == "step":
            m = re.search(r"Step\s*(\d+)", s, re.IGNORECASE)
        else:
            m = re.search(r"^(\d+)\.", s.strip())
        if m:
            numbers.append(int(m.group(1)))
        else:
            return False

    return numbers == list(range(1, len(numbers) + 1))


def load_prompt(file_path: str) -> str:
    """Load prompt template from a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _check_eq(x, tokens):
    """Check if token matches any in tokens list (ignore parentheses, spaces, case, special chars)"""
    x = x.replace("Ġ", "")
    x = regex.sub(r'[\(\)\s]', ' ', x).strip()
    return any(x == t for t in tokens)


def clean_question(question: str) -> str:
    """Remove fixed suffix from math question if present"""
    suffix = "Let's think step by step and output the final answer within \\boxed{}."
    if question.strip().endswith(suffix):
        return question.strip()[: -len(suffix)].strip()
    return question.strip()


class SampleQualityEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B", device="cuda"):
        """Initialize model and tokenizer for sample evaluation"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        print("[Init] Model loaded successfully.")

    def generate_rollouts(self, prompt: str, num_rollouts: int = 5, max_new_tokens: int = 2048) -> List[str]:
        """Generate multiple rollouts for a given prompt"""
        print(f"\n[Generate] Prompt ...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=1,
            temperature=0.5,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_rollouts,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        rollouts = [self.tokenizer.decode(o, skip_special_tokens=True)[len(prompt):] for o in outputs]
        for i, r in enumerate(rollouts):
            print(f"[Generate] Rollout {i + 1}:\n{r}\n")
        return rollouts

    def split_steps(self, rollout: str) -> Tuple[List[str], str]:
        """Split rollout into individual steps and extract final answer"""
        answer, last_ans_idx = extract_answer(rollout)
        first_step_idx = 0
        for m in re.finditer(r"(Step\s*\d+|^\d+\.)", rollout, re.IGNORECASE | re.MULTILINE):
            first_step_idx = m.start()
            break
        step_text = rollout[first_step_idx:last_ans_idx].strip()
        if not step_text:
            step_text = rollout.strip()

        # Attempt step patterns
        step_pattern1 = re.compile(r"(Step\s*\d+.*?)(?=Step\s*\d+|$)", re.DOTALL | re.IGNORECASE)
        steps1 = step_pattern1.findall(step_text)

        step_pattern2 = re.compile(r"^\d+\.\s+.*?(?=^\d+\.|\Z)", re.DOTALL | re.MULTILINE)
        steps2 = step_pattern2.findall(step_text)

        step_pattern3 = re.compile(r"(\d+\.\s.*?)(?=(?:\d+\.)|$)", re.DOTALL)
        steps3 = step_pattern3.findall(step_text)

        if steps1 and is_sequential(steps1, mode="step"):
            steps = steps1
        elif steps2 and is_sequential(steps2, mode="num"):
            steps = steps2
        elif steps3 and is_sequential(steps3, mode="num"):
            steps = steps3
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", step_text) if p.strip()]
            if len(paragraphs) > 1:
                steps = paragraphs
            else:
                steps = [line.strip() for line in step_text.split("\n") if line.strip()]

        return steps, answer

    def self_eval_confidence(self, steps: list[str], question: str = "") -> list[float]:
        """Compute per-step confidence scores (0~1) for a sequence of steps"""
        scores = []
        template = load_prompt("quality_evaluation/prompts/self_eval.txt")
        steps = [s.strip() for s in steps if s.strip()]
        question = clean_question(question)

        for t, current_step in enumerate(steps):
            previous_steps = "\n".join(steps[:t]) or "None"
            eval_prompt = template.format(
                question=question,
                previous_steps=previous_steps,
                current_step=current_step
            )
            print(f"\n[SelfEval] Evaluating step {t + 1}/{len(steps)}...")
            print(eval_prompt)

            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.3,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            logprobs_per_step = outputs.scores
            gen_token_ids = outputs.sequences[0][input_len:]

            r_tokens = ['A', 'correct', 'Correct']
            w_tokens = ['B', 'wrong', 'Wrong', 'incorrect', 'Incorrect']

            uncertainty = 1.0
            for gen_token_id, step_logits in reversed(list(zip(gen_token_ids, logprobs_per_step))):
                token_str = self.tokenizer.convert_ids_to_tokens([gen_token_id])[0]
                if not _check_eq(token_str, w_tokens + r_tokens):
                    continue
                probs = torch.softmax(step_logits[0], dim=-1).cpu().numpy()
                correct = sum(probs[self.tokenizer.encode(k, add_special_tokens=False)[0]]
                              for k in r_tokens if k in self.tokenizer.vocab)
                wrong = sum(probs[self.tokenizer.encode(k, add_special_tokens=False)[0]]
                            for k in w_tokens if k in self.tokenizer.vocab)
                uncertainty = wrong / (correct + wrong + 1e-9)
                break
            scores.append(uncertainty)
        return scores

    def rollout_confidence(self, rollout: str, question: str) -> float:
        """Compute confidence of a single rollout"""
        steps, _ = self.split_steps(rollout)
        step_confs = self.self_eval_confidence(steps, question)
        print("[Confidence] Step confs:", step_confs)
        rollout_conf = float(np.mean(step_confs)) if len(step_confs) > 0 else 0.0
        print(f"[Confidence] Rollout uncertainty = {rollout_conf:.4f}")
        return rollout_conf

    def rollout_reward(self, rollout: str, ground_truth: str) -> float:
        answer, _ = extract_answer(rollout)
        print("[Reward] Extracted answer:", answer)
        if is_equiv(answer, ground_truth):
            rollout_reward = 1.0
        else:
            rollout_reward = 0.0
        print(f"[Reward] Rollout reward = {rollout_reward:.1f}")
        return rollout_reward

    def sample_quality(self, question: str, ground_truth: str, num_rollouts: int = 8) -> float:
        """
        Evaluate a new sample's quality.

        Args:
            question: the math question
            ground_truth: the correct answer
            num_rollouts: number of rollouts to generate

        Returns:
            conf_score: confidence score (average step uncertainty)
            reward_score: variance of rollout rewards
            num_correct: number of rollouts matching ground truth
        """
        print("\n================ Evaluating New Sample ================")
        start_time = time.time()

        base_prompt = load_prompt("quality_evaluation/prompts/math_cot.txt")
        prompt = f"{base_prompt}{question}\nA: "
        rollouts = self.generate_rollouts(prompt, num_rollouts=num_rollouts)

        # ============ Step 1. Select one rollout to compute confidence ============
        def rollout_steps_count(rollout):
            steps, _ = self.split_steps(rollout)
            return len(steps)

        rollouts_sorted = sorted(
            rollouts,
            key=lambda r: (
                0 if 3 <= rollout_steps_count(r) <= 8 else
                1 if rollout_steps_count(r) > 8 else
                2,  # primary sorting: [3-8] < >8 < <3
                abs(rollout_steps_count(r) - 5)  # secondary: closer to 5 steps is better
            )
        )
        chosen_rollout = rollouts_sorted[0]
        print(f"[Selection] Chosen rollout has {rollout_steps_count(chosen_rollout)} steps.")

        # Compute confidence only on the chosen rollout
        confs = [self.rollout_confidence(chosen_rollout, question)]
        conf_score = float(np.mean(confs))
        print(f"[Sample] Final sample uncertainty score = {conf_score:.4f}")

        # ============ Step 2. Compute reward for all rollouts ============
        rewards = [self.rollout_reward(r, ground_truth) for r in rollouts]
        print(rewards)
        num_correct = sum(1 for r in rewards if r == 1.0)
        reward_score = float(np.var(rewards))
        print(f"[Sample] Final sample reward variance = {reward_score:.4f}")

        print("======================================================\n")
        end_time = time.time()
        print(f"[Timing] Sample evaluation took {end_time - start_time:.2f} seconds.")

        return conf_score, reward_score, num_correct



def main(args):
    evaluator = SampleQualityEvaluator()

    # Load the parquet file
    df = pd.read_parquet(args.data_path)

    # Determine processing range
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(df)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Automatically modify output filename if a slice is specified
    base, ext = os.path.splitext(args.output_path)
    if args.start_idx is not None or args.end_idx is not None:
        args.output_path = f"{base}_{start_idx}_{end_idx}{ext}"

    # Load existing results if available
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Consider only samples with complete scores as processed
        processed_indices = {
            r["index"] for r in results
            if r.get("uncertainty_score") is not None and r.get("reward_variance") is not None
        }
        print(f"[Resume] {len(processed_indices)} samples already processed (with valid scores).")
    else:
        results = []
        processed_indices = set()

    # Iterate over the samples
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_index = row["extra_info"]["index"]
        if sample_index in processed_indices:
            continue  # skip already processed samples

        question = row["prompt"][0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]

        print(f"\n[Processing] Sample index = {sample_index}")

        try:
            uncertainty_score, reward_variance, num_correct = evaluator.sample_quality(
                question, ground_truth, num_rollouts=args.num_rollouts
            )
        except Exception as e:
            print(f"[Error] Sample {sample_index} failed: {e}")
            uncertainty_score, reward_variance, num_correct = None, None, None

        results.append({
            "index": sample_index,
            "uncertainty_score": uncertainty_score,
            "reward_variance": reward_variance,
            "num_correct": num_correct
        })

        # Save results after each sample to prevent data loss on interruption
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample quality evaluation with resume support.")
    parser.add_argument("--data_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full.parquet",
                        help="Input parquet file path.")
    parser.add_argument("--output_path", type=str, default="data/train/rlvr_gsm8k/gsm8k_full_with_scores.json",
                        help="Output JSON file path.")
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="Number of rollouts per sample.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive) of samples to process.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (exclusive) of samples to process.")

    args = parser.parse_args()
    main(args)