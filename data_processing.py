from datasets import load_dataset

CWE_DESCRIPTIONS = {
    "CWE-020": "Improper Input Validation",
    "CWE-022": "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')",
    "CWE-078": "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
    "CWE-079": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
    "CWE-089": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
    "CWE-094": "Improper Control of Generation of Code ('Code Injection')",
    "CWE-125": "Out-of-bounds Read",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
    "CWE-269": "Improper Privilege Management",
    "CWE-287": "Improper Authentication",
    "CWE-306": "Missing Authentication for Critical Function",
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-416": "Use After Free",
    "CWE-434": "Unrestricted Upload of File with Dangerous Type",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-502": "Deserialization of Untrusted Data",
    "CWE-787": "Out-of-bounds Write",
    "CWE-798": "Use of Hard-coded Credentials",
    "CWE-862": "Missing Authorization",
    "CWE-863": "Incorrect Authorization",
    "CWE-918": "Server-Side Request Forgery (SSRF)",
    "CWE-119": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
    "CWE-077": "Improper Neutralization of Special Elements used in a Command ('Command Injection')",
    "safe": "safe"
}

def clean_cwe(cwe):
    if cwe.lower() == "safe":
        return "safe"
    s = str(cwe)
    digits = "".join(ch for ch in s if ch.isdigit())
    cwe_id = f"CWE-{digits.zfill(3)}"
    return CWE_DESCRIPTIONS.get(cwe_id, None)

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
            "output": sample["label"].replace("“","'"),
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

    allowed_cwes = set(CWE_DESCRIPTIONS.keys())
    def format_instruction(sample):
        output = clean_cwe(sample["cwe_id"])
        return {
            "instruction": "Analyze the following code and classify its vulnerability.",
            "output": output if output is not None else "unknown",
            "code": sample["vulnerable_code"].replace("\t", "").strip()
        }

    dataset = dataset.map(format_instruction)
    dataset = dataset.filter(lambda sample: sample["output"] != "unknown")
    return dataset.select_columns(["instruction", "output", "code"])

def process_secvuleval():
    dataset = load_dataset("arag0rn/SecVulEval", split="train")

    allowed_cwes = set(CWE_DESCRIPTIONS.keys())
    def format_instruction(sample):
        cwe_id = sample["cwe_list"][0] if sample["is_vulnerable"] else "safe"
        output = clean_cwe(cwe_id)
        return {
            "instruction": "Analyze the following code and classify its vulnerability.",
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