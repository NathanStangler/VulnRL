from datasets import load_dataset

CWE_DESCRIPTIONS = {
    "CWE-020": "improper input validation",
    "CWE-022": "improper limitation of a pathname to a restricted directory ('path traversal')",
    "CWE-078": "improper neutralization of special elements used in an os command ('os command injection')",
    "CWE-079": "improper neutralization of input during web page generation ('cross-site scripting')",
    "CWE-089": "improper neutralization of special elements used in an sql command ('sql injection')",
    "CWE-094": "improper control of generation of code ('code injection')",
    "CWE-125": "out-of-bounds read",
    "CWE-190": "integer overflow or wraparound",
    "CWE-200": "exposure of sensitive information to an unauthorized actor",
    "CWE-269": "improper privilege management",
    "CWE-287": "improper authentication",
    "CWE-306": "missing authentication for critical function",
    "CWE-352": "cross-site request forgery (csrf)",
    "CWE-400": "uncontrolled resource consumption",
    "CWE-416": "use after free",
    "CWE-434": "unrestricted upload of file with dangerous type",
    "CWE-476": "null pointer dereference",
    "CWE-502": "deserialization of untrusted data",
    "CWE-787": "out-of-bounds write",
    "CWE-798": "use of hard-coded credentials",
    "CWE-862": "missing authorization",
    "CWE-863": "incorrect authorization",
    "CWE-918": "server-side request forgery (ssrf)",
    "CWE-119": "improper restriction of operations within the bounds of a memory buffer",
    "CWE-077": "improper neutralization of special elements used in a command ('command injection')",
    "safe": "safe"
}

def clean_cwe(cwe):
    if str(cwe).strip().lower() == "safe":
        return "safe"
    s = str(cwe)
    digits = "".join(ch for ch in s if ch.isdigit())
    cwe_id = f"CWE-{digits.zfill(3)}"
    label = CWE_DESCRIPTIONS.get(cwe_id, None)
    return label if label is not None else None

def process_lemon42():
    dataset = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset", split="train")

    def format_data(sample):
        lines = sample["code"].strip().replace("\t", "").split('\n')
        sample["language"] = lines[0][3:].strip()
        sample["code"] = "\n".join(lines[1:-1])
        return sample

    dataset = dataset.map(format_data)
    dataset = dataset.filter(lambda sample: sample["language"] == "c++")
    dataset = dataset.remove_columns(["Unnamed: 0"])

    allowed_labels = set(CWE_DESCRIPTIONS.values())
    def format_instruction(sample):
        return {
            "instruction": "Analyze the following C++ code and classify its vulnerability.",
            "output": sample["label"].replace("“","'").strip().lower(),
            "code": sample["code"]
        }

    dataset = dataset.map(format_instruction)
    dataset = dataset.filter(lambda sample: sample["output"] in allowed_labels)
    return dataset.select_columns(["instruction", "output", "code"])

def process_megavul():
    dataset = load_dataset("hitoshura25/megavul", split="train")
    dataset = dataset.filter(lambda sample: sample["language"] == "C++")
    dataset = dataset.filter(lambda sample: sample["cwe_id"] != "CWE-Other")
    dataset = dataset.filter(lambda sample: sample["vulnerable_code"] is not None)

    def format_instruction(sample):
        output = clean_cwe(sample["cwe_id"])
        return {
            "instruction": "Analyze the following C++ code and classify its vulnerability.",
            "output": output if output is not None else "unknown",
            "code": sample["vulnerable_code"].replace("\t", "").strip()
        }

    dataset = dataset.map(format_instruction)
    dataset = dataset.filter(lambda sample: sample["output"] != "unknown")
    return dataset.select_columns(["instruction", "output", "code"])

def process_secvuleval():
    dataset = load_dataset("arag0rn/SecVulEval", split="train")

    def format_instruction(sample):
        cwe_id = sample["cwe_list"][0] if sample["is_vulnerable"] else "safe"
        output = clean_cwe(cwe_id)
        return {
            "instruction": "Analyze the following C++ code and classify its vulnerability.",
            "output": output if output is not None else "unknown",
            "code": sample["func_body"].replace("\t", "").strip()
        }

    dataset = dataset.map(format_instruction)
    dataset = dataset.filter(lambda sample: sample["output"] != "unknown")
    return dataset.select_columns(["instruction", "output", "code"])

def get_split(dataset, seed=42):
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