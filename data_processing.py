from datasets import load_dataset

def get_dataset(seed=42):
    dataset = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset", split="train")

    def format_data(sample):
        lines = sample["code"].strip().split('\n')
        sample["language"] = lines[0][3:].strip()
        sample["code"] = "\n".join(lines[1:-1])
        return sample

    dataset = dataset.map(format_data)
    dataset = dataset.filter(lambda sample: sample["language"] == "c++")
    dataset = dataset.remove_columns(["Unnamed: 0"])

    def format_instruction(sample):
        return {
            "instruction": "Analyze the following C++ code and classify its vulnerability.",
            "output": sample["label"],
            "code": sample["code"]
        }

    dataset = dataset.map(format_instruction)
    dataset = dataset.remove_columns(["label"])

    split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    train = split["train"]
    test_val = split["test"]
    val_split = test_val.train_test_split(test_size=0.5, shuffle=True, seed=seed)
    validation = val_split["train"]
    test = val_split["test"]
    return train, validation, test

def tokenize_dataset(dataset, tokenizer):
    def tokenize(sample):
        inputs = "Instruction: " + sample["instruction"] + "\n\nCode:\n" + sample["code"] + "\n\nResponse: "
        targets = sample["output"]
        tokenized = tokenizer(inputs + targets, truncation=True, max_length=1024, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, batched=False)