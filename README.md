# Fine-Tuning GPT-OSS-20B with Multilingual Dolma Data

This repository builds on our work extending the [Dolma toolkit](https://github.com/maieutic-nlp/dolma) into a multilingual framework. Our goal is to test and demonstrate the toolkit’s capabilities by processing and using a Spanish dataset for downstream experiments. As an initial proof of concept, we fine-tune [GPT-OSS-20B](https://platform.openai.com/docs/models/gpt-oss-20b) on this Spanish data to validate the quality and usability of our multilingual Dolma outputs.

### Setup
Create a new environment.
```
conda create -n gpt_oss python=3.10.13
```

Install the required packages.
```
cd dolma-lm
conda activate gpt_oss
pip install -r requirements.txt
```

### Benchmarking
The benchmarking step is used to evaluate baseline model performance using [MMLU-ProX](https://mmluprox.github.io/), an extended version of MMLU that tests reasoning across multiple languages, before fine-tuning. This provides a reference point to measure the effect of training with data processed by our multilingual Dolma toolkit.
1. Allocate a GPU node. 
```
salloc -A a100acct -p gpu-a100 --gres=gpu:1 -t 01:00:00
```
2. Enter the node.
```
srun --jobid=<JOBID> --pty bash
```
3. Load CUDA and activate the environment. 
```
module load cuda/12.1
conda activate gpt_oss
```
4. Run the benchmark.
```
python scripts/benchmark_<LANGUAGE>.py
```

### Fine-Tuning
1. Download and prepare the dataset with the Dolma toolkit. After processing, copy the entire `wikipedia/` folder (including `v0/documents/*.gz`) into this repository on your local machine.
2. Run `split_wiki.py` to create train/validation splits. This script loads `wikipedia/v0/documents/*.gz`, keeps only the text field, creates a 98% / 2% split, and writes:
```
data/wikipedia_<DATE>_<LANG>/
    ├── train.jsonl
    └── val.jsonl
```
You can edit the following defaults in `split_wiki.py`:
* DATE and LANG
* data_path (path to the shards)
* train/val ratio (train/val ratio, defaults to 0.02)
3. Create a DeepSpeed configuration file. This repository expects a DeepSpeed ZeRO-3 config file named `ds_zero3.json` in the directory where training is launched. This file controls how model parameters, gradients, and optimizer states are sharded across GPUs during training.
4. Run LoRA supervised fine-tuning with Slurm and DeepSpeed. Submit the provided Slurm job script to launch training:
```
sbatch run_lora_sft_ds.slurm
```
The Slurm script runs `lora_sft_deepspeed.py` with DeepSpeed enabled and passes the path to `ds_zero3.json` via `--deepspeed_config`. To customize a run (e.g., change the adapter name, sequence length, batch size, or number of GPUs), edit the Slurm script before submission. Training logs are written to:
```
lora_sft_ds.<JOBID>.out
lora_sft_ds.<JOBID>.err
```
Monitor job status and logs with:
```
squeue -u $USER
tail -f lora_sft_ds.<JOBID>.out
tail -f lora_sft_ds.<JOBID>.err
```
5. LoRA adapters are saved under:
```
outputs/adapter-<adapter_name>/adapters/
```
You can reuse the same base model and train multiple adapters (e.g. `en-simple`, `en-es`, `multilingual`) by changing `--adapter_name`.

**Notes**
- `gpt-oss-20b` is distributed with MXFP4 quantization and is dequantized to bf16 for training in `lora_sft_deepspeed.py`.
- DeepSpeed ZeRO-3 is used to shard model parameters, gradients, and optimizer states across GPUs.
- Single-GPU jobs are intended for debugging DeepSpeed initialization; full training requires multiple GPUs due to bf16 memory requirements.

### References
- https://huggingface.co/openai/gpt-oss-20b
- https://cookbook.openai.com/articles/gpt-oss/run-transformers
- https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
- https://mmluprox.github.io/
- https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite#overview