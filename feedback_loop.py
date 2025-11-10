# feedback_loop.py
import json
from code_evaluator import CodeEvaluator

def analyze_feedback(result):
    """Compute reward score based on compiler + sanitizer feedback."""
    reward = 0
    if result["compile"]["success"]:
        reward += 1
    if not result["tidy"]["success"]:
        reward -= 1
    if result["run"]["address_sanitizer_issue"]:
        reward -= 2
    if result["run"]["undefined_behavior_sanitizer_issue"]:
        reward -= 2
    if not result["run"]["success"]:
        reward -= 1
    return reward

def feedback_learning_step(model, tokenizer, code_path, feedback_log="feedback_logs.jsonl"):
    evaluator = CodeEvaluator()
    result = evaluator.evaluate_code(code_path)
    reward = analyze_feedback(result)

    log_entry = {
        "path": code_path,
        "reward": reward,
        "compile_success": result["compile"]["success"],
        "tidy_warnings": len(result["tidy"]["warnings"]),
        "asan_issue": result["run"]["address_sanitizer_issue"],
        "ubsan_issue": result["run"]["undefined_behavior_sanitizer_issue"]
    }

    with open(feedback_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[Feedback] {code_path}: reward={reward}")

    return reward
