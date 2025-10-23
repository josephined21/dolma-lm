import torch, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
model.eval()

lang = "es"
try:
    ds = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="test")
except:
    ds = load_dataset("li-lab/MMLU-ProX-Lite", lang, split="validation")

LETTER_CHOICES = [chr(ord('A') + i) for i in range(10)]

def to_letter(ans):
    """normalize ground-truth to uppercase letter a-j"""
    if isinstance(ans, str):
        a = ans.strip().upper()
        if a in LETTER_CHOICES:
            return a
        if a.isdigit():
            idx = int(a)
            if 0 <= idx < len(LETTER_CHOICES):
                return LETTER_CHOICES[idx]
    elif isinstance(ans, int):
        if 0 <= ans < len(LETTER_CHOICES):
            return LETTER_CHOICES[ans]
    return None

def format_question_es(q, options):
    """format spanish question and options"""
    lines = [f"pregunta: {q}", "opciones:"]
    for i, opt in enumerate(options[:len(LETTER_CHOICES)]):
        lines.append(f"{LETTER_CHOICES[i]}. {opt}")
    lines.append("\nresponde solo con la letra (a-j) de la opción correcta.")
    return "\n".join(lines)

def apply_chat_template(user_content: str):
    """apply chat template if available"""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "eres un asistente que responde con una sola letra a-j."},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return user_content

def evaluate(split, prompt_builder, batch_q=2, cand_batch=10):
    """evaluate accuracy using likelihood-based multiple choice"""
    import torch
    n_total, n_correct = 0, 0
    per_subject = defaultdict(lambda: {"n": 0, "c": 0})
    CANDS = [f" {chr(ord('A') + i)}" for i in range(10)]

    def build_labeled_inputs(prompts, labels):
        """build inputs and labels where prompt tokens are masked"""
        seqs = [p + l for p, l in zip(prompts, labels)]
        enc = tokenizer(seqs, return_tensors="pt", padding=True).to(model.device)
        enc_prompt = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)
        labels_t = enc["input_ids"].clone()
        labels_t[:] = -100
        attn_lens = enc["attention_mask"].sum(dim=1)
        for i in range(labels_t.size(0)):
            start = int(prompt_lens[i].item())
            end = int(attn_lens[i].item())
            if end > start:
                labels_t[i, start:end] = enc["input_ids"][i, start:end]
        return enc["input_ids"], enc["attention_mask"], labels_t

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def score_batch(input_ids, attn_mask, label_ids):
        """return average nll over candidate suffix"""
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = label_ids[:, 1:]
        B, Tm1, V = shift_logits.shape
        losses = ce_loss(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1)
        ).view(B, Tm1)
        mask = (shift_labels != -100)
        denom = mask.sum(dim=1).clamp_min(1)
        seq_loss = (losses * mask).sum(dim=1) / denom
        return seq_loss.cpu()

    for start_idx in range(0, len(split), batch_q):
        batch = [split[i] for i in range(start_idx, min(start_idx + batch_q, len(split)))]
        prompts, golds, subjects = [], [], []
        for ex in batch:
            q = ex.get("question") or ""
            options = []
            for i in range(10):
                opt = ex.get(f"option_{i}")
                if opt is None:
                    continue
                s = str(opt).strip()
                if s:
                    options.append(s)
            gold = to_letter(ex.get("answer"))
            subj = ex.get("category") or "unknown"
            if not q or len(options) < 2 or gold not in LETTER_CHOICES:
                continue
            user_prompt = format_question_es(q, options)
            full_prompt = prompt_builder(user_prompt)
            prompts.append(full_prompt)
            golds.append(gold)
            subjects.append(subj)
        if not prompts:
            continue
        flat_prompts, flat_labels = [], []
        for p in prompts:
            flat_prompts.extend([p] * 10)
            flat_labels.extend(CANDS)
        input_ids, attn_mask, label_ids = build_labeled_inputs(flat_prompts, flat_labels)
        seq_loss = score_batch(input_ids, attn_mask, label_ids)
        num_q = len(prompts)
        losses = seq_loss.view(num_q, 10)
        pred_idx = losses.argmin(dim=1).tolist()
        preds = [LETTER_CHOICES[i] for i in pred_idx]
        for pred, gold, subj in zip(preds, golds, subjects):
            n_total += 1
            per_subject[subj]["n"] += 1
            if pred == gold:
                n_correct += 1
                per_subject[subj]["c"] += 1

    overall = (n_correct / n_total * 100.0) if n_total else 0.0
    per_subject_acc = {k: (v["c"] / v["n"] * 100.0) if v["n"] else 0.0 for k, v in per_subject.items()}
    return overall, per_subject_acc, n_total

overall, per_subject_acc, n = evaluate(ds, prompt_builder=apply_chat_template, batch_q=2, cand_batch=10)
print(f"{lang} — overall accuracy: {overall:.2f}% on {n} items")
best = sorted(per_subject_acc.items(), key=lambda kv: -kv[1])[:5]
worst = sorted(per_subject_acc.items(), key=lambda kv: kv[1])[:5]
print("top subjects:", best)
print("bottom subjects:", worst)
