<div align="center">

# STEP-LEVEL UNCERTAINTY-AWARE REASONING DATA SELECTION IN REINFORCEMENT LEARNING FOR LLM MULTI-STEP REASONING


## Setup


### Train Enviroment
Our training pipeline is adapted from [verl](https://github.com/volcengine/verl) and  [rllm(DeepScaleR)](https://github.com/agentica-project/rllm). The installation commands that we verified as viable are as follows:
```bash
conda create -y -n rlvr_train python=3.10
conda activate rlvr_train
pip install -e .
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install ray vllm==0.6.3
pip install flash-attn --no-build-isolation
pip install wandb matplotlib
pip install huggingface_hub
```
If you are using H100 nodes and see errors like `CUDA error: device kernel image is invalid`, please refer to [this issue](https://github.com/ypwang61/One-Shot-RLVR/issues/22#issuecomment-3066442183) for fixing the problem.

### Eval Enviroment
Our evaluation pipeline for math reasoning tasks is adapted from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). The installation commands that we verified as viable are as follows:
```bash
conda create -y -n rlvr_eval python=3.10
conda activate rlvr_eval
cd Qwen2.5-Eval/evaluation
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
pip install wandb matplotlib
pip install -U transformers
pip install vllm==0.6.3
```

## Data Quality Evaluation

```bash
# our proposed method
python quality_evaluation/evaluate.py

# Random sampling
python data/sample_random.py --input data/train/rlvr_gsm8k/gsm8k_full.parquet --num 1000

# ppl perplexity
python quality_evaluation/compute_ppl.py

# token length
python quality_evaluation/compute_tokenlength.py

# IFD score
python quality_evaluation/compute_ifd.py

# Sample by score
python data/sample_by_score.py --input data/train/rlvr/math_full.parquet --score_file data/train/rlvr/math_full_with_scores_merged.json --field uncertainty_score --num 5 --mode top

```

## Training Models

```bash
conda activate rlvr_train
bash scripts/train/training_1.5b_math_full.sh
```

Please change `data.train_files` and `trainer.experiment_name` in the training script when trying other training subsets.

## Evaluation

### Eval Scripts for Qwen Models
To run evaluation for our method on 6 common math reasoning benchmarks (MATH500, AIME24, AMC23, Minerva Math, OlympiadBench, AIME25), we can follow the commands:
```bash
conda activate rlvr_eval
cd Qwen2.5-Eval/evaluation
bash sh/eval_one_experiment_all_ckpts.sh
```
Here for AIME24, AMC23, and AIME25, we evaluate the pass@8 results.
Please adjust the experiment name in `Qwen2.5-Eval/evaluation/sh/eval_one_experiment_all_ckpts.sh` when using other training examples. 

## Acknowledgements
- Our training experiments are powered by a modified fork of [rllm(DeepScaleR)](https://github.com/agentica-project/rllm) and [verl](https://github.com/volcengine/verl).
- Our evaluation experiments are based on a modified fork of [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
- Our model is trained on top of [`Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [`Qwen2.5-Math-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-7B), [`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
