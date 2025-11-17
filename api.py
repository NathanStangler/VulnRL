# api.py
from fastapi import FastAPI, UploadFile
from code_evaluator import CodeEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile, shutil, os

app = FastAPI(title="Vulnerability Detection API")

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
evaluator = CodeEvaluator()

@app.post("/analyze/")
async def analyze_code(file: UploadFile):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = evaluator.evaluate_code(tmpdir)
        return {
            "compile_success": result["compile"]["success"],
            "warnings": len(result["tidy"]["warnings"]),
            "asan_issue": result["run"]["address_sanitizer_issue"],
            "ubsan_issue": result["run"]["undefined_behavior_sanitizer_issue"]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
