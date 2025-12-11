import json
from typing import Dict, Any
from code_evaluator import CodeEvaluator


def analyze_feedback(result: Dict[str, Any], code_path: str, feedback_log: str) -> float:
    """Shape a scalar reward from the low-level evaluation `result`.

    The high-level goals are:

    1. Strongly discourage *non-compiling* code, especially trivial mistakes like
       "initializer-string for char array is too long".
    2. Give *high positive reward* for:
         - Code that compiles, and
         - Triggers AddressSanitizer stack-buffer-overflow (our target bug).
    3. Provide small positive reward for compilable programs that clang-tidy
       flags for classic dangerous functions (strcpy/strcat/gets/etc.).
    4. Log everything so we can debug what the policy is doing.

    Parameters
    ----------
    result : dict
        Structured feedback from the toolchain. Expected keys:
          - "compile": {"success": bool, "stderr": str, ...}
          - "tidy": {"success": bool, "warnings": list, ...}
          - "run": {"success": bool, "address_sanitizer_issue": bool,
                     "undefined_behavior_sanitizer_issue": bool, ...}
    code_path : str
        Path of the code file that was evaluated (for logging only).
    feedback_log : str
        Path to a JSONL log file where each evaluation is appended.

    Returns
    -------
    float
        Scalar reward to feed into RL.
    """
    # Defensive unpacking so missing keys don't crash training.
    compile_res: Dict[str, Any] = result.get("compile", {}) or {}
    tidy_res: Dict[str, Any] = result.get("tidy", {}) or {}
    run_res: Dict[str, Any] = result.get("run", {}) or {}

    reward: float = 0.0
    debug_tags = []

    # -------------------------------
    # 1) Compilation quality
    # -------------------------------
    compile_success = bool(compile_res.get("success", False))
    compile_stderr = compile_res.get("stderr", "") or ""

    if compile_success:
        # Base reward for just compiling.
        reward += 1.0
        debug_tags.append("compile_ok")
    else:
        # Strong penalty for non-compiling programs.
        reward -= 2.0
        debug_tags.append("compile_fail")

        # Extra penalty for the boring pattern:
        #   initializer-string for char array is too long
        lower_err = compile_stderr.lower()
        if "initializer-string for" in lower_err or "is too long for" in lower_err:
            reward -= 2.0
            debug_tags.append("initializer_overflow_compile_time")

    # -------------------------------
    # 2) Static analysis (clang-tidy)
    # -------------------------------
    tidy_success = bool(tidy_res.get("success", True))

    if not tidy_success:
        # If tidy completely bails out, small penalty.
        reward -= 0.5
        debug_tags.append("tidy_fail")

    # Look for classic dangerous functions in warnings.
    dangerous_funcs = ("strcpy", "strcat", "gets", "sprintf", "scanf", "memcpy")
    tidy_warnings = tidy_res.get("warnings", []) or []

    def warning_text(w: Any) -> str:
        if isinstance(w, dict):
            return str(w.get("message", "")).lower()
        return str(w).lower()

    if compile_success and tidy_warnings:
        if any(any(df in warning_text(w) for df in dangerous_funcs) for w in tidy_warnings):
            reward += 1.5
            debug_tags.append("tidy_dangerous_func")

    # -------------------------------
    # 3) Runtime sanitizer feedback
    # -------------------------------
    asan_hit = bool(run_res.get("address_sanitizer_issue", False))
    ubsan_hit = bool(run_res.get("undefined_behavior_sanitizer_issue", False))

    # We *only* really trust sanitizer results when the program compiled.
    if compile_success and asan_hit:
        # Jackpot: program compiled and triggered a stack-buffer-overflow.
        reward += 5.0
        debug_tags.append("asan_stack_overflow")

    if compile_success and ubsan_hit:
        # Mild bonus for UB; not as strong as a clear stack overflow.
        reward += 2.0
        debug_tags.append("ubsan_issue")

    # -------------------------------
    # 4) Final shaping and logging
    # -------------------------------
    log_entry = {
        "compile": compile_res,
        "tidy": tidy_res,
        "run": run_res,
        "reward": reward,
        "tags": debug_tags,
        "path": code_path,
    }

    try:
        with open(feedback_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        # Logging failure should never crash training; just note it.
        print(f"[analyze_feedback] Logging error: {e}")

    print(f"[analyze_feedback] reward={reward:.2f} tags={debug_tags}")
    return float(reward)


def feedback_learning_step(code_path, feedback_log: str = "feedback_logs.jsonl") -> float:
    evaluator = CodeEvaluator()
    result = evaluator.evaluate_code(code_path)

    if result:
        print(f"[feedback_learning_step] Result: {result}")
    else:
        print("[feedback_learning_step] Result is empty or None")

    # New: analyze_feedback now handles logging and uses code_path + feedback_log.
    reward = analyze_feedback(result, code_path=code_path, feedback_log=feedback_log)
    print(f"[Feedback] {code_path}: reward={reward}")

    return reward
