from datasets import concatenate_datasets
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from data_processing import LABEL_OPTIONS, get_split, process_lemon42, process_megavul, process_secvuleval
from feedback_loop import feedback_learning_step
from performance import clean_label
import argparse
import torch
import os
import random
import tempfile

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=None)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--output_dir", default="./rl_model")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--compiler_weight", type=float, default=0.7)
    p.add_argument("--report_to", default="wandb", choices=["none", "wandb"])
    p.add_argument("--wandb_entity", default="VulnRL")
    p.add_argument("--wandb_project", default="VulnRL")
    return p.parse_args()

def build_prompt(tokenizer, code):
    messages = [
        {"role": "system", "content": f"Analyze the following C++ code and classify its vulnerability. Your classification should be one of the following: {LABEL_OPTIONS}. Only respond with the classification."},
        {"role": "user", "content": code}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def write_temp_code_dirs(codes):
    paths = []
    for code in codes:
        dir_path = tempfile.mkdtemp(prefix="sample_")
        file_path = os.path.join(dir_path, "main.cpp")
        with open(file_path, "w") as f:
            f.write(code)
        paths.append(dir_path)
    return paths

def make_reward_fn(compiler_weight):
    def reward_fn(prompts, completions, **kwargs):
        codes = kwargs.get("code")
        labels = kwargs.get("label")
        compiler_cache = {}
        rewards = []

        for completion, code_text, label in zip(completions, codes, labels):
            if code_text not in compiler_cache:
                code_dir = write_temp_code_dirs([code_text])[0]
                compiler_cache[code_text] = float(feedback_learning_step(code_dir))

            compiler_reward = compiler_cache[code_text]
            pred_label = clean_label(completion)
            label = clean_label(label)
            cls_reward = 1.0 if (pred_label is not None and label is not None and pred_label == label) else 0.0
            mixed_reward = compiler_weight * compiler_reward + (1.0 - compiler_weight) * cls_reward
            rewards.append(float(mixed_reward))
        return rewards
    return reward_fn

def build_training_dataset(tokenizer, args):
    print("[0] Loading dataset...")
    datasets = [
        ("lemon42", process_lemon42),
        ("megavul", process_megavul),
        ("secvuleval", process_secvuleval),
    ]

    validation_set = []
    for name, processor in datasets:
        dataset = processor()
        _, validation, _ = get_split(dataset)
        print(f"{name} - Validation: {len(validation)}")
        validation_set.append(validation)

    dataset = concatenate_datasets(validation_set)
    
    def add_prompt(sample):
        return {
            "prompt": build_prompt(tokenizer, sample["code"]),
            "label": sample["output"],
        }

    dataset = dataset.map(add_prompt)
    dataset = dataset.select_columns(["prompt", "code", "label"])

    dataset = dataset.select(range(25))

    print(f"Combined - Validation: {len(dataset)}")
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.artifact:
        print("Downloading model artifact...")
        import wandb
        temp = tempfile.mkdtemp(prefix="model_artifact_")
        artifact = wandb.Api().artifact(args.artifact)
        artifact_dir = artifact.download(root=temp)
        args.model_name = artifact_dir
        print(f"Model artifact downloaded to {artifact_dir}")

    print(f"[1] Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = build_training_dataset(tokenizer, args)

    print(f"[2] Loading model from {args.model_name}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    peft_adapter = os.path.exists(os.path.join(args.model_name, "adapter_config.json"))
    if peft_adapter:
        print("Loading LoRA-finetuned model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config if args.load_in_4bit else None,
            device_map="auto",
        )
    else:
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config if args.load_in_4bit else None,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.load_in_4bit:
        print("[2.1] Preparing model for 4-bit training...")
        model = prepare_model_for_kbit_training(model)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        remove_unused_columns=False,
        save_strategy="epoch",
        logging_steps=10,
        report_to=args.report_to,
    )

    reward_fn = make_reward_fn(args.compiler_weight)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )

    print("[3] Starting GRPO training...")
    trainer.train()

    print("[4] Saving RL model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.report_to == "wandb":
        print("[5] Logging model to Weights & Biases...")
        import wandb
        run = wandb.run or wandb.init(project=args.wandb_project, entity=args.wandb_entity, dir=args.log_dir)
        name = os.path.basename(args.output_dir.rstrip("/"))
        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(args.output_dir)
        run.log_artifact(artifact)
        wandb.finish()

    print("✓ RL training complete.")

if __name__ == "__main__":
    main()