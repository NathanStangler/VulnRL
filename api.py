from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from code_evaluator import CodeEvaluator
from performance import get_prediction
import tempfile, shutil, os

app = FastAPI(title="Vulnerability Detection API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AVAILABLE_MODELS = {
    "phi-2": "microsoft/phi-2",
    "qwen": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "sft": "./finetuned_model",
    "rl": "./rl_model"
}

loaded_models = {}


def load_model(model_name: str):
    if model_name not in loaded_models:
        print(f"[API] Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
    return loaded_models[model_name]


@app.get("/models")
def list_models():
    return {"available_models": AVAILABLE_MODELS}


@app.post("/analyze/")
async def analyze_code(
    file: UploadFile,
    model_key: str = Query("sft", description="Which model to use.")
):
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Model '{model_key}' not available."}

    tokenizer, model = load_model(AVAILABLE_MODELS[model_key])
    evaluator = CodeEvaluator()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = evaluator.evaluate_code(tmpdir)

        with open(path, "r") as f:
            code_text = f.read()

        prediction = get_prediction(code_text, tokenizer=tokenizer, model=model)

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


@app.get("/ui.html", response_class=HTMLResponse)
def serve_ui():
    path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
