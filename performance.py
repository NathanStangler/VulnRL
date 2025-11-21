from code_chunker import build_chunks
from data_processing import get_dataset, tokenize_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    p.add_argument("--model_dir", default="./finetuned_model")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--chunk_max_tokens", type=int, default=1024)
    p.add_argument("--chunk_overlap", type=int, default=128)
    return p.parse_args()

def get_prediction(prompt, tokenizer, model, max_new_tokens=64):
    messages = [
        {"role": "system", "content": "Analyze the following C++ code and classify its vulnerability. Your classification should be one of the following: Improper Input Validation, Improper Limitation of a Pathname to a Restricted Directory (“Path Traversal”), Improper Neutralization of Special Elements used in an OS Command (“OS Command Injection”), Improper Neutralization of Input During Web Page Generation (“Cross-site Scripting”), Improper Neutralization of Special Elements used in an SQL Command (“SQL Injection”), Improper Control of Generation of Code (“Code Injection”), Improper Output Neutralization for Logs, Integer Overflow or Wraparound, NULL Pointer Dereference, Deserialization of Untrusted Data, URL Redirection to Untrusted Site (“Open Redirect”), Improper Restriction of XML External Entity Reference, Out-of-bounds Write, safe. Only respond with the classification."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_label(output)

def predict_single_code(code, tokenizer, model, chunk_max_tokens=1024, overlap=128, max_new_tokens=64):
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "main.cpp"), "w") as f:
            f.write(code)
        chunks = build_chunks(directory, tokenizer.encode, max_tokens=chunk_max_tokens, overlap=overlap)
        responses = [get_prediction(chunk["code"], tokenizer, model, max_new_tokens=max_new_tokens) for chunk in chunks]
        most_common = Counter(responses).most_common(1)
        return most_common[0][0] if most_common else "unsafe"

ALLOWED_LABELS = [
    "improper input validation",
    "path traversal",
    "os command injection",
    "cross-site scripting",
    "sql injection",
    "code injection",
    "improper output neutralization for logs",
    "integer overflow or wraparound",
    "null pointer dereference",
    "deserialization of untrusted data",
    "open redirect",
    "improper restriction of xml external entity reference",
    "out-of-bounds write",
    "safe"
]

def clean_label(text):
    s = text.strip().lower()
    if s in ALLOWED_LABELS:
        return s

    for label in ALLOWED_LABELS:
        if label in s:
            return label

    for line in s.splitlines():
        l = line.strip()
        if l in ALLOWED_LABELS:
            return l
        for label in ALLOWED_LABELS:
            if label in l:
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

    print("[1] Loading test dataset...")
    _, _, test = get_dataset()
    print(f"Test: {len(test)}")

    print("[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[3] Tokenizing dataset...")
    test_dataset = tokenize_dataset(test, tokenizer)

    print("[4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    model.eval()

    print("[5] Evaluating truncated predictions...")
    y_true = []
    y_pred = []
    for sample in tqdm.tqdm(test_dataset):
        prompt = sample["code"]
        label = sample["output"].strip().lower()
        prediction = get_prediction(prompt, tokenizer, model, max_new_tokens=args.max_new_tokens).strip().lower()
        y_true.append(label)
        y_pred.append(prediction)

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

    print("[6] Evaluating chunked predictions...")
    y_true_chunk = []
    y_pred_chunk = []
    for sample in tqdm.tqdm(test_dataset):
        code = sample["code"]
        label = sample["output"].strip().lower()
        prediction = predict_single_code(code, tokenizer, model, chunk_max_tokens=args.chunk_max_tokens, overlap=args.chunk_overlap, max_new_tokens=args.max_new_tokens)
        y_true_chunk.append(label)
        y_pred_chunk.append(prediction)
        
    accuracy_chunk = accuracy_score(y_true_chunk, y_pred_chunk)
    precision_chunk = precision_score(y_true_chunk, y_pred_chunk, average="weighted", zero_division=0)
    recall_chunk = recall_score(y_true_chunk, y_pred_chunk, average="weighted", zero_division=0)
    f1_chunk = f1_score(y_true_chunk, y_pred_chunk, average="weighted", zero_division=0)

    print(f"Chunked Accuracy: {accuracy_chunk:.4f}")
    print(f"Chunked Precision: {precision_chunk:.4f}")
    print(f"Chunked Recall: {recall_chunk:.4f}")
    print(f"Chunked F1 Score: {f1_chunk:.4f}")

    y_true_chunk_binary = [to_binary(label) for label in y_true_chunk]
    y_pred_chunk_binary = [to_binary(pred) for pred in y_pred_chunk]

    binary_accuracy_chunk = accuracy_score(y_true_chunk_binary, y_pred_chunk_binary)
    binary_precision_chunk = precision_score(y_true_chunk_binary, y_pred_chunk_binary, pos_label="safe", zero_division=0)
    binary_recall_chunk = recall_score(y_true_chunk_binary, y_pred_chunk_binary, pos_label="safe", zero_division=0)
    binary_f1_chunk = f1_score(y_true_chunk_binary, y_pred_chunk_binary, pos_label="safe", zero_division=0)

    print(f"Chunked Binary Accuracy (safe/unsafe): {binary_accuracy_chunk:.4f}")
    print(f"Chunked Binary Precision (safe): {binary_precision_chunk:.4f}")
    print(f"Chunked Binary Recall (safe): {binary_recall_chunk:.4f}")
    print(f"Chunked Binary F1 Score (safe): {binary_f1_chunk:.4f}")

    summary = {
        "model": args.model_dir,
        "test_samples": len(test_dataset),
        "truncated_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "binary_truncated_metrics": {
            "accuracy": binary_accuracy,
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": binary_f1,
        },
        "chunked_metrics": {
            "accuracy": accuracy_chunk,
            "precision": precision_chunk,
            "recall": recall_chunk,
            "f1": f1_chunk,
        },
        "binary_chunked_metrics": {
            "accuracy": binary_accuracy_chunk,
            "precision": binary_precision_chunk,
            "recall": binary_recall_chunk,
            "f1": binary_f1_chunk,
        }
    }

    with open(os.path.join(args.log_dir, "performance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {os.path.join(args.log_dir, 'performance_summary.json')}")
    print("[✔] Evaluation complete.")

if __name__ == "__main__":
    main()