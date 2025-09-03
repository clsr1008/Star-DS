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
    提取最终答案，返回答案本身和在 rollout 中的结束位置。

    Returns:
        answer (str): 答案内容
        ans_idx (int): 答案在 rollout 中的结尾索引
    """

    # 1. 尝试提取最后的 \boxed{} 或 \fbox{}
    boxed = last_boxed_only_string(rollout)
    if boxed:
        ans = remove_boxed(boxed).strip()
        ans_idx = rollout.rfind(boxed) + len(boxed)
        # print(111111)
        return ans, ans_idx

    # 找到 "code" 或 "python" 等字样的位置
    match = re.search(r"\b(code|python)\b", rollout, flags=re.IGNORECASE)
    if match:
        rollout = rollout[:match.start()]

    # 2. 匹配形如 <var> = 2 或 <var> = 4/2 或 <var> = \frac{29}{4}
    # eq_iter = list(re.finditer(
    #     r"\b([a-zA-Z])\s*=\s*(?:\\frac\{([+\-]?\d+)\}\{([+\-]?\d+)\}|([+\-]?\d+(?:/\d+)?(?:\.\d+)?))(?![a-zA-Z])",
    #     rollout
    # ))
    # if eq_iter:
    #     last_match = eq_iter[-1]
    #     if last_match.group(2) and last_match.group(3):
    #         ans = f"\\frac{{{last_match.group(2)}}}{{{last_match.group(3)}}}"
    #         ans_idx = last_match.end()
    #     else:
    #         ans = last_match.group(4).strip()
    #         ans_idx = last_match.end(4)
    #     return ans, ans_idx

    # 3. 匹配区间形式 (a, b) 或 [a, b]
    # interval_iter = list(re.finditer(
    #     r"[\(\[]\s*(?:\\frac\{[+\-]?\d+\}\{[+\-]?\d+\}|[-+]?\d+(?:\.\d+)?)(?:\s*,\s*(?:\\frac\{[+\-]?\d+\}\{[+\-]?\d+\}|[-+]?\d+(?:\.\d+)?))+\s*[\)\]]",
    #     rollout
    # ))
    # if interval_iter:
    #     last_match = interval_iter[-1]
    #     ans = last_match.group(0)
    #     ans_idx = last_match.end()
    #     print(2222222)
    #     return ans, ans_idx

    # 4. 兜底：最后一个数字（包括 \frac{a}{b}）
    number_iter = list(re.finditer(
        r"(\\frac\{[+\-]?\d+\}\{[+\-]?\d+\}|[-+]?\d+(?:\.\d+)?)",
        rollout
    ))
    if number_iter:
        last_match = number_iter[-1]
        ans = last_match.group(0)
        ans_idx = last_match.end()

        # 检查这个数字是否在区间里
        right_context = rollout[ans_idx:].lstrip()
        if right_context.startswith((")", "]")):
            # 向左找到最近的 ( 或 [
            left_part = rollout[:last_match.start()]
            left_bracket_pos = max(left_part.rfind("("), left_part.rfind("["))
            if left_bracket_pos != -1:
                interval_text = rollout[left_bracket_pos: ans_idx + 1]  # 包括右括号
                ans = interval_text
                ans_idx = ans_idx + 1
                print("3333_interval")
                return ans, ans_idx

        # print(3333333)
        return ans, ans_idx

    # 4. 实在没有就返回空
    return "", len(rollout)

def is_sequential(steps, mode="step"):
    """
    检查 step 是否从1开始并连续递增
    :param steps: 匹配到的 step 列表
    :param mode: "step" 表示 Step N 格式，"num" 表示 1., 2. 格式
    """
    numbers = []
    for s in steps:
        if mode == "step":
            m = re.search(r"Step\s*(\d+)", s, re.IGNORECASE)
        else:  # num 格式
            m = re.search(r"^(\d+)\.", s.strip())
        if m:
            numbers.append(int(m.group(1)))
        else:
            return False  # 有没解析出数字的，直接判失败

    # 检查是否连续 1,2,3...
    return numbers == list(range(1, len(numbers) + 1))

def load_prompt(file_path: str) -> str:
    """直接读取 prompt 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _check_eq(x, tokens):
    """判断token是否匹配（忽略括号、空格、大小写、前导空格符号 Ġ）"""
    x = x.replace("Ġ", "")  # 去掉 Ġ
    x = regex.sub(r'[\(\)\s]', ' ', x).strip()
    if any(x == t for t in tokens):
        return True
    return False

def clean_question(question: str) -> str:
    suffix = "Let's think step by step and output the final answer within \\boxed{}."
    if question.strip().endswith(suffix):
        return question.strip()[: -len(suffix)].strip()
    return question.strip()


class SampleQualityEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-1.5B", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        print("[Init] Model loaded successfully.")

    def generate_rollouts(self, prompt: str, num_rollouts: int = 5, max_new_tokens: int = 2048) -> List[str]:
        """生成多条 rollouts"""
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
            eos_token_id = self.tokenizer.eos_token_id,
        )
        rollouts = [self.tokenizer.decode(o, skip_special_tokens=True)[len(prompt):] for o in outputs]
        for i, r in enumerate(rollouts):
            print(f"[Generate] Rollout {i+1}:\n{r}\n")
        return rollouts

    def split_steps(self, rollout: str) -> Tuple[List[str], str]:
        """
        拆分 rollout 为 steps 和最终答案
        - 去掉第一个 step 之前的内容
        - 去掉最后一个 step 之后的内容（保留最后 \boxed{} 作为答案）
        - 支持 Step N:, 编号列表 1., 2., ... 和换行分割
        """
        # 1. 提取最终答案
        answer, last_ans_idx = extract_answer(rollout)

        # 2. 找到第一个 step 的起始位置
        first_step_idx = None
        for m in re.finditer(r"(Step\s*\d+|^\d+\.)", rollout, re.IGNORECASE | re.MULTILINE):
            first_step_idx = m.start()
            break
        if first_step_idx is None:
            # 如果没有显式 Step/编号列表，则从第一行开始
            lines = rollout.strip().split("\n")
            for line in lines:
                if line.strip() and "\\boxed" not in line:
                    first_step_idx = rollout.find(line)
                    break
        if first_step_idx is None:
            first_step_idx = 0  # 兜底

        # 3. 截取中间文本作为 step 区域
        step_text = rollout[first_step_idx:last_ans_idx].strip()
        if not step_text:
            step_text = rollout.strip()

        # 4. 尝试 Step N: 匹配
        step_pattern1 = re.compile(r"(Step\s*\d+.*?)(?=Step\s*\d+|$)", re.DOTALL | re.IGNORECASE)
        steps1 = step_pattern1.findall(step_text)

        # 5. 尝试编号列表匹配
        step_pattern2 = re.compile(r"^\d+\.\s+.*?(?=^\d+\.|\Z)", re.DOTALL | re.MULTILINE)
        steps2 = step_pattern2.findall(step_text)

        # 匹配连续行内或换行的编号列表 Step 1., 2., ...
        step_pattern3 = re.compile(r"(\d+\.\s.*?)(?=(?:\d+\.)|$)", re.DOTALL)
        steps3 = step_pattern3.findall(step_text)

        # 6. 没有匹配 → 按行拆分
        if steps1 and is_sequential(steps1, mode="step"):
            steps = steps1
        elif steps2 and is_sequential(steps2, mode="num"):
            steps = steps2
        elif steps3 and is_sequential(steps3, mode="num"):
            steps = steps3
        else:
            # 按段落拆分（空行分隔）
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", step_text) if p.strip()]
            if len(paragraphs) > 1:
                # 多段落，直接每段作为一个 step
                steps = paragraphs
            else:
                # 单段落 → 按行拆分
                steps = [line.strip() for line in step_text.split("\n") if line.strip()]

        return steps, answer

    def self_eval_confidence(self, steps: list[str], question: str = "") -> list[float]:
        """
        输入：
            steps: [s1, s2, ..., sn]  每一步的推理文本
            question: 原始数学题目
        输出：
            [c1, c2, ..., cn] 每一步的 correctness score (0~1, 即 P(A))
        """
        scores = []
        template = load_prompt("quality_evaluation/prompts/self_eval.txt")
        steps = [s.strip() for s in steps if s.strip()]
        question = clean_question(question)

        for t in range(len(steps)):
            previous_steps = "\n".join(steps[:t]) or "None"
            current_step = steps[t]

            eval_prompt = template.format(
                question=question,
                previous_steps=previous_steps,
                current_step=current_step
            )

            print(f"\n[SelfEval] Evaluating step {t + 1}/{len(steps)}...")
            print(eval_prompt)

            # ============ 1. 编码输入并生成 ============
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,  # 开启采样
                    temperature=0.3,  # 控制多样性
                    top_p=1.0,  # nucleus sampling
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs['input_ids'].shape[1]  # prompt 长度
            logprobs_per_step = outputs.scores  # 每步生成的 logits

            # 1. 获取生成序列
            generated_ids = outputs.sequences[0]  # 第一个 batch 的完整序列
            # 2. 如果你想只看新生成的 token，可以去掉 prompt 部分
            new_generated_ids = generated_ids[input_len:] # 只保留模型新生成的部分
            # 3. 转为文字
            generated_text = self.tokenizer.decode(new_generated_ids, skip_special_tokens=True)
            print("生成文本：")
            print(generated_text)

            # 取生成的tokens和对应的 top_logprobs
            # tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0][input_len:])
            # print("tokens:", tokens)
            # print("len_tokens", len(tokens))
            # # print("logprobs_per_step:", logprobs_per_step)
            # print("len_tuple",len(logprobs_per_step))
            # print(logprobs_per_step[0].shape) # torch.Size([1, 151936])

            # ============ 2. 定义候选 token ============
            r_tokens = ['A', 'correct', 'Correct']
            w_tokens = ['B', 'wrong', 'Wrong', 'incorrect', 'Incorrect']

            uncertainty = 1.0
            # count = 0
            # ============ 3. 遍历生成的token和top_logprobs ============
            gen_token_ids = outputs.sequences[0][input_len:]  # 只取新生成的 token
            for gen_token_id, step_logits in reversed(list(zip(gen_token_ids, logprobs_per_step))):
                # count += 1
                token_str = self.tokenizer.convert_ids_to_tokens([gen_token_id])[0]
                if not _check_eq(token_str, w_tokens + r_tokens):
                    continue
                # softmax得到概率分布
                probs = torch.softmax(step_logits[0], dim=-1).cpu().numpy()
                # 累加 A 和 B 的概率
                correct = sum(probs[self.tokenizer.encode(k, add_special_tokens=False)[0]]
                              for k in r_tokens if k in self.tokenizer.vocab)
                wrong = sum(probs[self.tokenizer.encode(k, add_special_tokens=False)[0]]
                            for k in w_tokens if k in self.tokenizer.vocab)
                # 这里取 normalized 概率更合理
                uncertainty = wrong / (correct + wrong + 1e-9) # 注意我们衡量的是不确定度
                break
            scores.append(uncertainty)
            # print("实际循环次数:", count)
        return scores

    # def rollout_score(self, rollout: str, question: str, ground_truth: str) -> float:
    #     print("\n[Rollout] Evaluating rollout...")
    #     steps, answer = self.split_steps(rollout)
    #     print("answer:",answer)
    #     # print("\n[Split Steps] Steps:")
    #     # for i, step in enumerate(steps, 1):
    #     #     print(f"Step {i}:\n{step}")
    #     # print("\n[Split Steps] Final Answer:")
    #     # print(answer)
    #     # return
    #
    #     # ============ Step 1. 自评置信度 ============
    #     step_confs = self.self_eval_confidence(steps, question)
    #     print(step_confs)
    #     rollout_conf = float(np.mean(step_confs)) if len(step_confs) > 0 else 0.0
    #     print(f"[Rollout] Rollout confidence = {rollout_conf:.4f}")
    #
    #     # ============ Step 2. 计算 reward ============
    #     answer, _ = extract_answer(rollout)
    #     if is_equiv(answer, ground_truth):
    #         rollout_reward = 1.0
    #     else:
    #         rollout_reward = 0.0
    #     print(f"[Rollout] Rollout reward = {rollout_reward:.1f}")
    #
    #     return rollout_conf, rollout_reward

    def rollout_confidence(self, rollout: str, question: str) -> float:
        """计算 rollout 的置信度"""
        steps, _ = self.split_steps(rollout)
        step_confs = self.self_eval_confidence(steps, question)
        print("[Confidence] Step confs:", step_confs)
        rollout_conf = float(np.mean(step_confs)) if len(step_confs) > 0 else 0.0
        print(f"[Confidence] Rollout uncertainty = {rollout_conf:.4f}")
        return rollout_conf

    def rollout_reward(self, rollout: str, ground_truth: str) -> float:
        """计算 rollout 的 reward"""
        answer, _ = extract_answer(rollout)
        print("[Reward] Extracted answer:", answer)
        if is_equiv(answer, ground_truth):
            rollout_reward = 1.0
        else:
            rollout_reward = 0.0
        print(f"[Reward] Rollout reward = {rollout_reward:.1f}")
        return rollout_reward

    def sample_quality(self, question: str, ground_truth: str, num_rollouts: int = 8) -> float:
        print("\n================ Evaluating New Sample ================")
        start_time = time.time()
        base_prompt = load_prompt("quality_evaluation/prompts/math_cot.txt")
        prompt = f"{base_prompt}{question}\nA: "
        rollouts = self.generate_rollouts(prompt, num_rollouts=num_rollouts)

        # confs, rewards = zip(*[self.rollout_score(r, question, ground_truth) for r in rollouts])
        # confs = [self.rollout_confidence(r, question) for r in rollouts]

        # ============ Step 1. 选择一个 rollout 来计算 confidence ============
        def rollout_steps_count(rollout):
            steps, _ = self.split_steps(rollout)
            return len(steps)

        rollouts_sorted = sorted(
            rollouts,
            key=lambda r: (
                0 if 3 <= rollout_steps_count(r) <= 8 else
                1 if rollout_steps_count(r) > 8 else
                2,  # 先分类: [3-8] < >8 < <3
                abs(rollout_steps_count(r) - 5)  # 次要指标: 越接近5步越好
            )
        )
        chosen_rollout = rollouts_sorted[0]
        print(f"[Selection] Chosen rollout has {rollout_steps_count(chosen_rollout)} steps.")

        # 只对 chosen_rollout 计算 conf
        confs = [self.rollout_confidence(chosen_rollout, question)]
        conf_score = float(np.mean(confs))
        print(f"[Sample] Final sample uncertainty score = {conf_score:.4f}")

        # ============ Step 2. 对所有 rollout 计算 reward ============
        rewards = [self.rollout_reward(r, ground_truth) for r in rollouts]
        print(rewards)
        num_correct = sum(1 for r in rewards if r == 1.0)  # 统计 reward=1.0 的个数
        reward_score = float(np.var(rewards))  # 方差
        print(f"[Sample] Final sample reward variance = {reward_score:.4f}")


        print("======================================================\n")
        end_time = time.time()
        print(f"[Timing] Sample evaluation took {end_time - start_time:.2f} seconds.")

        return conf_score, reward_score, num_correct



def main(args):
    evaluator = SampleQualityEvaluator()

    # 读取 parquet 文件
    df = pd.read_parquet(args.data_path)
    # 确定处理区间
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(df)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # 如果指定了分片范围，则自动修改输出文件名
    base, ext = os.path.splitext(args.output_path)
    if args.start_idx is not None or args.end_idx is not None:
        args.output_path = f"{base}_{start_idx}_{end_idx}{ext}"

    # 读取已完成的结果（如果存在）
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # 只将分数完整的样本视作已处理
        processed_indices = {
            r["index"] for r in results
            if r.get("uncertainty_score") is not None and r.get("reward_variance") is not None
        }
        print(f"[Resume] {len(processed_indices)} samples already processed (with valid scores).")
    else:
        results = []
        processed_indices = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        sample_index = row["extra_info"]["index"]
        if sample_index in processed_indices:
            continue  # 跳过已处理的

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

        # 每处理完一条就保存一次，防止再次中断丢失
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample quality evaluation with resume support.")
    parser.add_argument("--data_path", type=str, default="data/train/rlvr/math_full.parquet",
                        help="Input parquet file path.")
    parser.add_argument("--output_path", type=str, default="data/train/rlvr/math_full_with_scores.json",
                        help="Output json file path.")
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="Number of rollouts per sample.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive) of samples to process.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (exclusive) of samples to process.")

    args = parser.parse_args()
    main(args)