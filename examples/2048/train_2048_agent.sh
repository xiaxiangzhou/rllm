#!/bin/bash

# Train a 2048 agent using GRPO
# This script uses the default PPO trainer configuration with GRPO algorithm

python examples/2048/train_2048_agent.py \
    --config-name ppo_trainer \
    trainer=grpo \
    trainer.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.output_dir=outputs/2048_agent \
    trainer.num_train_epochs=3 \
    trainer.per_device_train_batch_size=2 \
    trainer.per_device_eval_batch_size=4 \
    trainer.gradient_accumulation_steps=8 \
    trainer.learning_rate=1e-5 \
    trainer.warmup_ratio=0.1 \
    trainer.logging_steps=10 \
    trainer.save_steps=100 \
    trainer.eval_steps=100 \
    trainer.save_total_limit=3 \
    trainer.bf16=true \
    trainer.gradient_checkpointing=true \
    trainer.ddp_find_unused_parameters=false \
    +env.env_args.max_steps=100 \
    +agent.agent_args.max_steps=100 \
    +agent.agent_args.use_accumulate_history=True

