from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from data_processing import get_dataset, tokenize_dataset
from code_chunker import build_chunks
from code_evaluator import CodeEvaluator
from feedback_loop import feedback_learning_step
import argparse
import torch
import json
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./finetuned_model")
    p.add_argument("--output_dir", default="./finetuned_model")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("[1] Loading test dataset...")
    _, _, test = get_dataset()
    print(f"Test: {len(test)}")

    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[3] Tokenizing dataset...")
    test_dataset = tokenize_dataset(test, tokenizer)

    print("[4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    print("[5] Preparing trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to=args.report_to,
        do_train=False,
        do_eval=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )

    print("[6] Evaluating...")
    eval_results = trainer.evaluate()
    print(json.dumps(eval_results, indent=2))

    with open(os.path.join(args.log_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print("[7] Running compiler feedback loop...")
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

    summary = {
        "model": model_path,
        "test_samples": len(test_dataset),
        "eval_metrics": eval_results,
        "last_feedback_reward": reward,
    }

    with open(os.path.join(args.log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {os.path.join(args.log_dir, 'summary.json')}")
    print("[✔] Evaluation complete.")

if __name__ == "__main__":
    main()