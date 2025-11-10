# train_and_evaluate.py
"""
End-to-end pipeline for C++ vulnerability classification.

Steps:
1. Load and tokenize dataset
2. Fine-tune model
3. Evaluate on validation/test sets
4. Run compiler-based feedback evaluation
5. Log rewards for reinforcement-based improvement
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from data_processing import get_dataset, tokenize_dataset
from code_chunker import build_chunks
from code_evaluator import CodeEvaluator
from feedback_loop import feedback_learning_step

# Configuration

MODEL_NAME = "microsoft/phi-2"  # can swap with CodeLlama or StarCoder
OUTPUT_DIR = "./finetuned_model"
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Step 1: Dataset loading and tokenization

print("[1] Loading dataset...")
train_ds, val_ds, test_ds = get_dataset()
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

print("[2] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # handle missing pad_token

print("[3] Tokenizing dataset...")
tokenized_train = tokenize_dataset(train_ds, tokenizer)
tokenized_val = tokenize_dataset(val_ds, tokenizer)
tokenized_test = tokenize_dataset(test_ds, tokenizer)

# Step 2: Model fine-tuning

print("[4] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_steps=50,
    logging_dir=LOG_DIR,
    logging_steps=20,
    report_to="none",  # disable wandb if not configured
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print("[5] Fine-tuning model...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("[✔] Model fine-tuning complete.")

# Step 3: Evaluate on validation/test sets

print("[6] Evaluating on test set...")
eval_results = trainer.evaluate(eval_dataset=tokenized_test)
print(json.dumps(eval_results, indent=2))

with open(os.path.join(LOG_DIR, "eval_results.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

# Step 4: Run compiler-based feedback on generated outputs

print("[7] Running compiler feedback loop...")

# Example directory containing sample C++ files to test
SAMPLE_DIR = "./sample_cpp"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Create a simple example C++ file if none exist
example_cpp = os.path.join(SAMPLE_DIR, "example.cpp")
if not os.path.exists(example_cpp):
    with open(example_cpp, "w") as f:
        f.write(
            '#include <iostream>\nint main(){char buf[4]; strcpy(buf,"overflow!"); return 0;}'
        )

# Evaluate compiler feedback and log reward
reward = feedback_learning_step(model, tokenizer, SAMPLE_DIR)
print(f"[Reward] Compiler feedback reward: {reward}")

# Step 5: Save summary and logs

summary = {
    "model": MODEL_NAME,
    "training_samples": len(train_ds),
    "validation_samples": len(val_ds),
    "test_samples": len(test_ds),
    "eval_metrics": eval_results,
    "last_feedback_reward": reward,
}

with open(os.path.join(LOG_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[✔] Training and evaluation pipeline complete.")
print(f"Summary written to {os.path.join(LOG_DIR, 'summary.json')}")
