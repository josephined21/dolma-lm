from datasets import load_dataset
import os

DATE = "20251101" # TODO: set date
LANG = "simple" # TODO: set language code

# input shards
data_path = "../wikipedia/v0/documents/*.gz" # TODO: set path to wikipedia shards

# output folder
out_dir = f"data/wikipedia_{DATE}_{LANG}"
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset("json", data_files=data_path, split="train")

# keep only the text column
if "text" in ds.column_names:
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

# 98/2 train/val split
split = ds.train_test_split(test_size=0.02, seed=42, shuffle=True)

train_path = f"{out_dir}/train.jsonl"
val_path = f"{out_dir}/val.jsonl"

split["train"].to_json(train_path)
split["test"].to_json(val_path)

print(f"done. wrote:\n  {train_path}\n  {val_path}")
