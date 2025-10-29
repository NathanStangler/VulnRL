import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from codeEvaluator import CodeEvaluator

def test_add_evaluation():
    evaluator = CodeEvaluator()
    add = evaluator.evaluate_code(os.path.join(os.path.dirname(__file__), "add"))
    assert add["compile"]["success"], "Expected compile to succeed"
    assert add["tidy"]["success"], "Expected tidy to succeed"
    assert add["run"]["success"], "Expected run to succeed"

def test_overflow_evaluation():
    evaluator = CodeEvaluator()
    overflow = evaluator.evaluate_code(os.path.join(os.path.dirname(__file__), "overflow"))
    assert overflow["compile"]["success"], "Expected compile to succeed"
    assert overflow["tidy"]["success"], "Expected tidy to succeed"
    assert not overflow["run"]["success"], "Expected run to fail"
