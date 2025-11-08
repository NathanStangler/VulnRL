import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from codeChunker import build_chunks
from transformers import AutoTokenizer

def test_add_chunks():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    chunks = build_chunks(os.path.join(os.path.dirname(__file__), "add"), tokenizer.encode)
    assert len(chunks) == 3, f"Expected 3 chunks"
    assert chunks[0]["code"] == '#include <iostream>\n#include "add.hpp"\n\nint main() {\n    int result = add(3, 4);\n    std::cout << "3 + 4 = " << result;\n    return 0;\n}'
    assert chunks[1]["code"] == 'int add(int a, int b);'
    assert chunks[2]["code"] == 'int add(int a, int b) {\n    return a + b;\n}'

def test_overflow_chunks():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    chunks = build_chunks(os.path.join(os.path.dirname(__file__), "overflow"), tokenizer.encode)
    assert len(chunks) == 1, f"Expected 1 chunk"
    assert chunks[0]["code"] == '#include <cstring>\n#include <iostream>\n\nint main() {\n    char buffer[8];\n    strcpy(buffer, "0123456789ABCDEF");\n    std::cout << buffer;\n    return 0;\n}'

def test_overflow_chunks_small():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    chunks = build_chunks(os.path.join(os.path.dirname(__file__), "overflow"), tokenizer.encode, max_tokens=20, overlap=0)
    assert len(chunks) == 3, f"Expected 3 chunks"
    assert chunks[0]["code"] == '#include <cstring>\n#include <iostream>\n\nint main() {\n    char buffer[8];'
    assert chunks[1]["code"] == 'strcpy(buffer, "0123456789ABCDEF");'
    assert chunks[2]["code"] == 'std::cout << buffer;\n    return 0;\n}'