#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J lora-sft-ds
#SBATCH -o lora_sft_ds.%j.out
#SBATCH -e lora_sft_ds.%j.err

source ~/.bashrc
conda activate gpt_oss

# logs that help debug deepspeed init
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TRANSFORMERS_VERBOSITY=info

# run deepspeed + point hf trainer at the deepspeed json config
deepspeed --num_gpus=1 lora_sft_deepspeed.py \
  --model_name openai/gpt-oss-20b \
  --train_file data/wikipedia_20251101_simple/train.jsonl \
  --val_file data/wikipedia_20251101_simple/val.jsonl \
  --adapter_name en-simple \
  --deepspeed_config ds_zero3.json \
  --max_seq_len 256 \
  --per_device_train_batch_size 1 \
  --grad_accum 1
