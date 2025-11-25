import argparse
import os
import random
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data_processing import get_dataset
from performance import clean_label
from feedback_loop import feedback_learning_step   # << NEW


def parse_args():
    p = argparse.ArgumentParser()

    # Can now accept HF model name OR local folder
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   help="HF model OR local checkpoint folder.")

    p.add_argument("--output_dir", default="./rl_model")
    p.add_argument("--log_dir", default="./logs")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=32)

    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None)

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_messages(code: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a security analyzer for C/C++ code. "
                "Given a code snippet, classify it into one of the known vulnerability "
                "categories or 'safe'. Only respond with the classification label."
            ),
        },
        {"role": "user", "content": code},
    ]


def build_prompt(tokenizer, code: str) -> str:
    messages = build_messages(code)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(args):
    print(f"[1] Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2] Loading model from {args.model_name}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config if args.load_in_4bit else None,
        device_map="auto",
    )

    if args.use_lora:
        print("[2.1] Preparing LoRA...")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return tokenizer, model


def collate_fn(batch, tokenizer):
    codes = [ex["code"] for ex in batch]
    labels = [ex["output"] for ex in batch]

    prompts = [build_prompt(tokenizer, code) for code in codes]
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels_text": labels,
        "codes": codes,
    }


def rl_step(model, tokenizer, batch, args, baseline):
    """
    RL step:

    - Sample a prediction from the model.
    - Compute reward using feedback_learning_step(model, tokenizer, code_path, ...),
      which runs the compiler + sanitizers and returns a scalar reward.
    - Use REINFORCE with a moving baseline.
    """
    device = model.device

    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    codes = batch["codes"]
    paths = batch["paths"]

    # 1) Sample an action (we still generate, so the policy has something to optimize)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only the generated tokens
    gen_only = []
    for inp, out in zip(input_ids, generated):
        gen_only.append(out[len(inp):])

    pred_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    # 2) Compiler-based reward via feedback_learning_step
    rewards_list = []
    for i, code_path in enumerate(paths):
        if code_path is None:
            # If dataset doesn’t have a path, fall back gracefully
            print("[WARN] No code_path in batch example; assigning reward 0.0")
            r = 0.0
        else:
            # This matches the signature you showed:
            # def feedback_learning_step(model, tokenizer, code_path, feedback_log="feedback_logs.jsonl"):
            r = feedback_learning_step(model, tokenizer, code_path)
        rewards_list.append(float(r))

    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)

    # 3) Compute log-prob of the generated tokens (policy gradient style)
    padded_gen = torch.nn.utils.rnn.pad_sequence(
        gen_only, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    full_inputs = torch.cat([input_ids, padded_gen], dim=1)
    full_attn = torch.ones_like(full_inputs).to(device)

    outputs = model(input_ids=full_inputs, attention_mask=full_attn)
    logits = outputs.logits[:, :-1, :]
    target_ids = full_inputs[:, 1:]

    log_probs_all = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).view(logits.size(0), logits.size(1))

    batch_size = len(gen_only)
    gen_log_probs = []
    for i in range(batch_size):
        prompt_len = input_ids[i].ne(tokenizer.pad_token_id).sum().item()
        gen_len = gen_only[i].ne(tokenizer.pad_token_id).sum().item()
        start = max(prompt_len - 1, 0)
        end = start + gen_len
        gen_log_probs.append(log_probs_all[i, start:end].sum())

    gen_log_probs = torch.stack(gen_log_probs)

    # 4) Baseline + REINFORCE loss
    if baseline is None:
        baseline = rewards.mean().item()

    advantage = rewards - baseline
    loss = -(advantage * gen_log_probs).mean()

    new_baseline = 0.9 * baseline + 0.1 * rewards.mean().item()

    stats = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item() if len(rewards) > 1 else 0.0,
        "baseline": baseline,
        "pred_example": pred_texts[0],
    }

    return loss, new_baseline, stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(args.seed)

    print("[0] Loading dataset...")
    train, _, _ = get_dataset()

    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))

    tokenizer, model = load_model_and_tokenizer(args)
    model.train()

    loader = DataLoader(
        train,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    baseline = None
    global_step = 0

    for epoch in range(args.epochs):
        print(f"[3] RL Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            loss, baseline, stats = rl_step(model, tokenizer, batch, args, baseline)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"[step {global_step}] loss={loss.item():.4f} "
                        f"reward={stats['reward_mean']:.4f} "
                        f"baseline={stats['baseline']:.4f}"
                    )
                    print(f"   pred example: {stats['pred_example']}")

    print("[4] Saving RL model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✓ Done.")


if __name__ == "__main__":
    main()
