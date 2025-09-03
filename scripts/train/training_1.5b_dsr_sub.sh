##!/bin/bash
#set -x
#
# CHECKPOINTS_DIR=./checkpoints
#
#export VLLM_ATTENTION_BACKEND=XFORMERS
#export CUDA_VISIBLE_DEVICES=0,1
#
#python3 -m verl.trainer.main_ppo \
# # 数据路径参数
# data.train_files=data/train/one_shot_rlvr/dsr_sub.parquet \
# data.val_files=data/test/math500.parquet \
# data.train_batch_size=128 \
# data.val_batch_size=530 \
# data.max_prompt_length=1024 \
# # prompt 长度最大为 1024
# data.max_response_length=3072 \
# # 生成响应最大为 3072 token
# # 模型与优化器设置
# actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
# # 使用 Qwen2.5-Math-1.5B 作为 actor 模型
# actor_rollout_ref.model.use_remove_padding=True \
# # 启用 remove padding 优化速度
# actor_rollout_ref.model.enable_gradient_checkpointing=True \
# # 启用 gradient checkpointing 节省显存
# # PPO算法细节
# actor_rollout_ref.actor.optim.lr=1e-6 \
# actor_rollout_ref.actor.ppo_mini_batch_size=128 \
# # PPO 的 mini-batch 设置为 128
# actor_rollout_ref.actor.use_dynamic_bsz=True \
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
# # 设定 PPO 最大处理 token 数量以防 GPU OOM
# actor_rollout_ref.actor.use_kl_loss=True \
# # 使用 KL 损失来约束输出分布的漂移
# actor_rollout_ref.actor.kl_loss_coef=0.001 \
# actor_rollout_ref.actor.kl_loss_type=low_var_kl \
# # 一种更稳定的 KL 变体
# actor_rollout_ref.actor.fsdp_config.param_offload=False \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
# # rollout 设置（即采样过程）
# actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
# # 使用 tensor parallel 加速大模型处理
# actor_rollout_ref.rollout.name=vllm \
# # 使用 vLLM 做推理 rollout
# actor_rollout_ref.rollout.temperature=0.6 \
# # temperature 控制生成多样性
# +actor_rollout_ref.rollout.val_temperature=0.6 \
# actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
# actor_rollout_ref.rollout.n=8 \
# # 训练时每个样本 rollout 数为 8，验证为 1
# +actor_rollout_ref.rollout.n_val=1 \
# # 参考模型（ref model）设置， 参考模型用于计算 PPO 的 KL 散度
# actor_rollout_ref.ref.fsdp_config.param_offload=True \
# # 参考模型用于计算 PPO 的 KL 散度
# # 奖励模型和算法控制
# reward_model.reward_manager='naive' \
# # 奖励管理器使用简单策略
# algorithm.adv_estimator=grpo \
# # 使用 GRPO（Generalized Return-based Policy Optimization）替代标准 PPO
# algorithm.kl_ctrl.kl_coef=0.001 \
# # KL 控制项权重为 0.001
# # Trainer设置（训练控制器）
# trainer.critic_warmup=0 \
# trainer.logger=['console','wandb'] \
# # 使用 wandb + 控制台做日志记录
# trainer.project_name='verl_few_shot'\
# trainer.experiment_name='Qwen2.5-Math-1.5B-dsr_sub'\
# trainer.checkpoints_dir=$CHECKPOINTS_DIR \
# +trainer.val_before_train=True \
# # 每轮训练前先在验证集上跑一次，检查质量
# trainer.n_gpus_per_node=2 \
# # 单节点 8 GPU 训练
# trainer.nnodes=1 \
# trainer.save_freq=20 \
# trainer.test_freq=20 \
# trainer.default_hdfs_dir=null \
# trainer.total_epochs=2000 2>&1 | tee verl_demo.log
# # 总训练轮数 2000

# world_size：训练可用的 GPU 数量(通常 = trainer.n_gpus_per_node * trainer.nnodes)
# infer_tp（推理的 tensor-parallel 大小）: 对应脚本里的 actor_rollout_ref.rollout.tensor_model_parallel_size
# 要求 world_size % infer_tp == 0（即 GPU 数必须能被 tensor-parallel 的分片数整除），否则无法把推理并行切分成整块

#!/bin/bash
set -x

CHECKPOINTS_DIR=./checkpoints

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=db1e7422d34fb72c311641f26a9652e39fe5414c

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/one_shot_rlvr/dsr_sub.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_few_shot'\
 trainer.experiment_name='Qwen2.5-Math-1.5B-dsr_sub'\
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=2000 2>&1 | tee verl_demo.log