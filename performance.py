from code_chunker import build_chunks
from data_processing import CWE_DESCRIPTIONS, LABEL_OPTIONS, process_lemon42, process_megavul, process_secvuleval, get_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import argparse
import tempfile
import torch
import tqdm
import json
import os
import difflib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=None)
    p.add_argument("--model_dir", default="./finetuned_model")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--chunk_max_tokens", type=int, default=1024)
    p.add_argument("--chunk_overlap", type=int, default=128)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()

def get_predictions(prompts, tokenizer, model, max_new_tokens=64):
    messages = [
        [
            {"role": "system", "content": f"Analyze the following C++ code and classify its vulnerability. Your classification should be one of the following: {LABEL_OPTIONS}. Only respond with the classification."},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        for message in messages
    ]

    def process_texts(texts_batch):
        model_inputs = tokenizer(texts_batch, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    try:
        outputs = process_texts(texts)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM during batch inference, falling back to single-sample processing...")
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            outputs = []
            for text in texts:
                outputs.extend(process_texts([text]))
        else:
            raise

    return [clean_label(output) for output in outputs]

def predict_codes(codes, tokenizer, model, chunk_max_tokens=1024, overlap=128, max_new_tokens=64):
    all_chunk_texts = []
    chunk_counts = []

    for code in codes:
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "main.cpp"), "w") as f:
                f.write(code)
            chunks = build_chunks(directory, tokenizer.encode, max_tokens=chunk_max_tokens, overlap=overlap)
            chunk_texts = [chunk["code"] for chunk in chunks if chunk["code"]]
        chunk_counts.append(len(chunk_texts))
        all_chunk_texts.extend(chunk_texts)

    if not all_chunk_texts:
        return ["unsafe"] * len(codes)

    chunk_predictions = get_predictions(all_chunk_texts, tokenizer, model, max_new_tokens=max_new_tokens)

    predictions = []
    idx = 0
    for count in chunk_counts:
        if count == 0:
            predictions.append("unsafe")
            continue
        responses = chunk_predictions[idx : idx + count]
        most_common = Counter(responses).most_common(1)
        predictions.append(most_common[0][0] if most_common else "unsafe")
        idx += count

    return predictions

ALLOWED_LABELS = list({v.strip().lower() for v in CWE_DESCRIPTIONS.values()})

def clean_label(text):
    s = text.strip().lower()
    if s in ALLOWED_LABELS:
        return s

    for label in ALLOWED_LABELS:
        if label in s or s in label:
            return label

    for line in s.splitlines():
        l = line.strip()
        if l in ALLOWED_LABELS:
            return l
        for label in ALLOWED_LABELS:
            if label in l or l in label:
                return label

    close = difflib.get_close_matches(s, ALLOWED_LABELS, n=1, cutoff=0.6)
    if close:
        return close[0]

    if "safe" in s:
        return "safe"

    return "unsafe"

def to_binary(label):
    return "safe" if label == "safe" else "unsafe"

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    if args.artifact:
        print(f"Downloading model artifact...")
        import wandb
        temp = tempfile.mkdtemp(prefix="model_artifact_")
        artifact = wandb.Api().artifact(args.artifact)
        artifact_dir = artifact.download(root=temp)
        args.model_dir = artifact_dir
        print(f"Model artifact downloaded to {artifact_dir}")

    print("[1] Processing datasets...")
    datasets = [
        ("lemon42", process_lemon42),
        ("megavul", process_megavul),
        ("secvuleval", process_secvuleval),
    ]

    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[3] Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=quantization_config if args.load_in_4bit else None,
            device_map="auto",
        )
    except Exception as e:
        print(f"PEFT adapters not found, loading base model.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=quantization_config if args.load_in_4bit else None,
            device_map="auto",
        )
    model.eval()

    def evaluate_dataset(name, test_dataset):
        print(f"Evaluating predictions for {name}...")
        y_true = []
        y_pred = []
        for start in tqdm.tqdm(range(0, len(test_dataset), args.batch_size)):
            end = min(start + args.batch_size, len(test_dataset))
            indices = range(start, end)
            codes = [test_dataset[i]["code"] for i in indices]
            labels = [test_dataset[i]["output"] for i in indices]
            predictions = predict_codes(codes, tokenizer, model, chunk_max_tokens=args.chunk_max_tokens, overlap=args.chunk_overlap, max_new_tokens=args.max_new_tokens)
            y_true.extend(labels)
            y_pred.extend(predictions)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        y_true_binary = [to_binary(label) for label in y_true]
        y_pred_binary = [to_binary(pred) for pred in y_pred]
        binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
        binary_precision = precision_score(y_true_binary, y_pred_binary, pos_label="safe", zero_division=0)
        binary_recall = recall_score(y_true_binary, y_pred_binary, pos_label="safe", zero_division=0)
        binary_f1 = f1_score(y_true_binary, y_pred_binary, pos_label="safe", zero_division=0)

        print(f"Binary Accuracy (safe/unsafe): {binary_accuracy:.4f}")
        print(f"Binary Precision (safe): {binary_precision:.4f}")
        print(f"Binary Recall (safe): {binary_recall:.4f}")
        print(f"Binary F1 Score (safe): {binary_f1:.4f}")

        return {
            "test_samples": len(test_dataset),
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "binary_metrics": {
                "accuracy": binary_accuracy,
                "precision": binary_precision,
                "recall": binary_recall,
                "f1": binary_f1,
            }
        }

    summary = {"model": args.model_dir, "datasets": {}}

    print("[4] Evaluating datasets...")
    for name, data in datasets:
        dataset = data()
        _, _, test = get_split(dataset)
        print(f"Test size for {name}: {len(test)}")
        dataset_summary = evaluate_dataset(name, test)
        summary["datasets"][name] = dataset_summary

    with open(os.path.join(args.log_dir, "performance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {os.path.join(args.log_dir, 'performance_summary.json')}")
    print("[✔] Evaluation complete.")

if __name__ == "__main__":
    main()