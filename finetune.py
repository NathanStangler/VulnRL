from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_processing import process_lemon42, process_megavul, process_secvuleval, get_split, tokenize_dataset
from datasets import concatenate_datasets
import argparse
import torch
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--output_dir", default="./finetuned_model")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--report_to", default="wandb", choices=["none", "wandb"])
    p.add_argument("--wandb_entity", default="VulnRL")
    p.add_argument("--wandb_project", default="VulnRL")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    if args.report_to == "wandb":
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "end"

    print("[1] Loading datasets...")
    datasets = [
        ("lemon42", process_lemon42),
        ("megavul", process_megavul),
        ("secvuleval", process_secvuleval),
    ]
    train_set = []
    validation_set = []
    for name, processor in datasets:
        dataset = processor()
        train, validation, _ = get_split(dataset)
        print(f"{name} - Train: {len(train)}, Validation: {len(validation)}")
        train_set.append(train)
        validation_set.append(validation)
    
    train = concatenate_datasets(train_set)
    validation = concatenate_datasets(validation_set)
    print(f"Combined - Train: {len(train)}, Validation: {len(validation)}")

    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[3] Tokenizing dataset...")
    train_dataset = tokenize_dataset(train, tokenizer)
    validation_dataset = tokenize_dataset(validation, tokenizer)

    print("[4] Loading base model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config if args.load_in_4bit else None,
        device_map="auto",
    )

    if args.load_in_4bit:
        print("[4.1] Preparing model for 4-bit training...")
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.use_lora:
        print("[4.2] Applying LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        logging_dir=args.log_dir,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=args.report_to,
        fp16=args.load_in_4bit,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    print("[5] Fine-tuning model...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.report_to == "wandb":
        print("[6] Logging model to Weights & Biases...")
        import wandb
        run = wandb.run or wandb.init(project=args.wandb_project, entity=args.wandb_entity, dir=args.log_dir)
        artifact = wandb.Artifact("finetuned_model", type="model")
        artifact.add_dir(args.output_dir)
        run.log_artifact(artifact)
        wandb.finish()

    print("[✔] Model fine-tuning complete.")

if __name__ == "__main__":
    main()