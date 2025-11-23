# api.py
from fastapi import FastAPI, UploadFile, Query
from pydantic import BaseModel
from code_evaluator import CodeEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from performance import get_prediction  # from your evaluator file
import tempfile, shutil, os

app = FastAPI(title="Vulnerability Detection API")

# List of allowed models
AVAILABLE_MODELS = {
    "phi-2": "microsoft/phi-2",
    "qwen": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "sft": "./finetuned_model",
    "rl": "./rl_model"
}

loaded_models = {}  # cache

def load_model(model_name: str):
    """Lazy-load model into cache."""
    if model_name not in loaded_models:
        print(f"[API] Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
    return loaded_models[model_name]


@app.get("/models")
def list_models():
    """Return the available models the user can choose from."""
    return {"available_models": AVAILABLE_MODELS}


@app.post("/analyze/")
async def analyze_code(
    file: UploadFile,
    model_key: str = Query("sft", description="Which model to use for vulnerability classification.")
):
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Model '{model_key}' not available."}

    # Load model on demand
    model_path = AVAILABLE_MODELS[model_key]
    tokenizer, model = load_model(model_path)
    evaluator = CodeEvaluator()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Run compiler + sanitizer analysis
        result = evaluator.evaluate_code(tmpdir)

        # 2. Read code text for ML analysis
        with open(path, "r") as f:
            code_text = f.read()

        # 3. ML vulnerability classification
        prediction = get_prediction(
            code_text,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=32
        )

        return {
            "model_used": model_key,
            "classification": prediction,
            "compiler": {
                "compile_success": result["compile"]["success"],
                "warnings": len(result["tidy"]["warnings"]),
                "asan_issue": result["run"]["address_sanitizer_issue"],
                "ubsan_issue": result["run"]["undefined_behavior_sanitizer_issue"]
            }
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
