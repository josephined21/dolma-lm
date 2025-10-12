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

### References
- https://huggingface.co/openai/gpt-oss-20b
- https://cookbook.openai.com/articles/gpt-oss/run-transformers
- https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers