import os
import shutil
import subprocess
import tempfile

class CodeEvaluator:
    DEFAULT_FLAGS = [
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-O0",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
    ]
    DEFAULT_TIDY_CHECKS = ("clang-analyzer-*,bugprone-*,security-*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers")

    def __init__(self, flags=None, tidy_checks=None):
        self.TOOLS = {
            "clang++": shutil.which("clang++"),
            "clang-tidy": shutil.which("clang-tidy"),
        }
        self.DEFAULT_FLAGS = list(flags) if flags is not None else list(self.DEFAULT_FLAGS)
        self.TIDY_CHECKS = tidy_checks if tidy_checks is not None else self.DEFAULT_TIDY_CHECKS

    def _run(self, cmd, cwd=None, timeout=60):
        try:
            completed = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as e:
            return -1, e.stdout, e.stderr
        except FileNotFoundError:
            return -1, "", ""

    def _get_files(self, root):
        extensions = (".C", ".cc", ".cpp", ".CPP", ".c++", ".cp", ".cxx")
        files = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(extensions):
                    files.append(os.path.join(dirpath, name))
        return files

    def _compile(self, src_path, temp_dir):
        binary_path = os.path.join(temp_dir, "compiled.out")
        files = self._get_files(src_path)
        if not files:
            return False, None, "", ""
        cmd = [self.TOOLS["clang++"]] + files + ["-o", binary_path] + self.DEFAULT_FLAGS
        retcode, out, err = self._run(cmd, cwd=src_path, timeout=300)
        return retcode == 0, binary_path, out, err

    def _run_binary(self, binary_path, args=None):
        if not os.path.exists(binary_path):
            return False, "", ""
        try:
            retcode, out, err = self._run([binary_path] + (args or []), timeout=10)
            return retcode == 0, out, err
        except subprocess.TimeoutExpired as e:
            return False, e.stdout, e.stderr

    def _run_tidy_file(self, path):
        cmd = [self.TOOLS["clang-tidy"], path, "-checks=" + self.TIDY_CHECKS, "--"] + self.DEFAULT_FLAGS
        retcode, out, err = self._run(cmd, cwd=os.path.dirname(path), timeout=120)

        warnings = []
        for line in (out + "\n" + err).splitlines():
            if ": warning:" in line or ": error:" in line:
                warnings.append(line.strip())

        return retcode == 0, out, err, warnings

    def _run_tidy(self, src_path):
        files = self._get_files(src_path)
        tidy_success = True
        tidy_out, tidy_err, tidy_warnings = [], [], []
        for file in files:
            success, out, err, warnings = self._run_tidy_file(file)
            tidy_out.append(out)
            tidy_err.append(err)
            tidy_warnings.extend([f"{file}: {warning}" for warning in warnings])
            if not success:
                tidy_success = False

        return tidy_success, "\n".join(tidy_out), "\n".join(tidy_err), tidy_warnings

    def evaluate_code(self, path, run=True, run_args=None):
        result = {}
        if not os.path.isdir(path):
            return result
        with tempfile.TemporaryDirectory(prefix="vulnrl_") as temp_dir:
            src_path = os.path.join(temp_dir, "src")
            shutil.copytree(path, src_path)
            compile_success, binary_path, compile_out, compile_err = self._compile(src_path, temp_dir)
            result["compile"] = {
                "success": compile_success,
                "stdout": compile_out,
                "stderr": compile_err
            }
            tidy_success, tidy_out, tidy_err, tidy_warnings = self._run_tidy(src_path)
            result["tidy"] = {
                "success": tidy_success,
                "stdout": tidy_out,
                "stderr": tidy_err,
                "warnings": tidy_warnings,
            }
            if run:
                if compile_success:
                    run_success, run_out, run_err = self._run_binary(binary_path, args=run_args)
                else:
                    run_success, run_out, run_err = False, "", ""
                result["run"] = {
                    "success": run_success,
                    "stdout": run_out,
                    "stderr": run_err,
                    "address_sanitizer_issue": "AddressSanitizer" in run_err,
                    "undefined_behavior_sanitizer_issue": "runtime error:" in run_err,
                }
        return result