# sft for gpt-oss-20b with deepspeed zero-3
# - gpt-oss loads as mxfp4 by default; we dequantize to bf16
# - deepspeed handles sharding, so do not use device_map when enabled
# - this is meant to validate deepspeed wiring; oom on small setups is expected

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_name", default="openai/gpt-oss-20b")

    # data
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)

    # output
    p.add_argument("--adapter_name", default="en-simple")
    p.add_argument("--output_dir", default="outputs")

    # training
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # deepspeed
    p.add_argument("--deepspeed_config", default=None)

    return p.parse_args()


def expand_paths(spec: str):
    p = Path(spec)

    # directory -> all jsonl/jsonl.gz under it
    if p.exists() and p.is_dir():
        files = sorted([*p.rglob("*.jsonl"), *p.rglob("*.jsonl.gz")])
        if not files:
            raise FileNotFoundError(f"no jsonl files under {spec}")
        return [str(x) for x in files]

    # comma list
    if "," in spec:
        out = [s.strip() for s in spec.split(",") if s.strip()]
        if not out:
            raise FileNotFoundError(f"empty file list: {spec}")
        return out

    # glob
    if any(ch in spec for ch in "*?[]"):
        matches = sorted(str(x) for x in Path().glob(spec))
        if not matches:
            raise FileNotFoundError(f"glob matched no files: {spec}")
        return matches

    # single file
    if not p.exists():
        raise FileNotFoundError(f"file not found: {spec}")
    return [spec]


def detect_lora_targets(model):
    # common projection names across decoder-only transformer impls
    candidates = [
        # attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        "dense", "out_proj",
        # mlp
        "gate_proj", "up_proj", "down_proj",
        "w1", "w2", "w3", "wi", "wo",
    ]

    present = set()
    for n, _ in model.named_modules():
        leaf = n.split(".")[-1]
        if leaf in candidates:
            present.add(leaf)

    return [c for c in candidates if c in present] or ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_model(model_name: str, use_deepspeed: bool):
    # gpt-oss comes with a built-in quantization_config (mxfp4)
    cfg = AutoConfig.from_pretrained(model_name)
    has_builtin_quant = getattr(cfg, "quantization_config", None) is not None

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # if the repo ships mxfp4, dequantize to bf16 for training
    if has_builtin_quant:
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    # deepspeed zero-3 and hf device_map/low_cpu_mem_usage don't mix
    if not use_deepspeed:
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = False

    # some environments don't support attn_implementation; fall back cleanly
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs,
        )
    except (TypeError, ValueError):
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def main():
    args = parse_args()
    use_deepspeed = args.deepspeed_config is not None

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = load_model(args.model_name, use_deepspeed)

    targets = detect_lora_targets(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)

    train_files = expand_paths(args.train_file)
    val_files = expand_paths(args.val_file)
    ds = load_dataset("json", data_files={"train": train_files, "val": val_files})

    run_dir = os.path.join(args.output_dir, f"adapter-{args.adapter_name}")
    os.makedirs(run_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        max_seq_length=args.max_seq_len,
        tokenizer=tok,
    )

    trainer.train()

    # saves peft adapter weights (not a merged full model)
    model.save_pretrained(os.path.join(run_dir, "adapters"))
    tok.save_pretrained(run_dir)


if __name__ == "__main__":
    main()
