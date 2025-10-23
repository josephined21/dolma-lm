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

### References
- https://huggingface.co/openai/gpt-oss-20b
- https://cookbook.openai.com/articles/gpt-oss/run-transformers
- https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
- https://mmluprox.github.io/
- https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite#overview