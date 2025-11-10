from data_processing import get_dataset, tokenize_dataset
from code_chunker import build_chunks
from code_evaluator import CodeEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Example workflow
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
train, val, test = get_dataset()
tokenized_train = tokenize_dataset(train, tokenizer)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Fine-tune and evaluate
trainer = Trainer(model=model, train_dataset=tokenized_train, ...)
trainer.train()

# Evaluate compiled code feedback
evaluator = CodeEvaluator()
results = evaluator.evaluate_code("./sample_cpp/")
print(results)
