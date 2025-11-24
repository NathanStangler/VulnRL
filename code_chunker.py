from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, TokenTextSplitter
import os

def _get_files(root):
    extensions = (".C", ".cc", ".cpp", ".CPP", ".c++", ".cp", ".cxx",
                  ".H", ".hh", ".hpp", ".hxx", ".h++")
    return [os.path.join(dirpath, name)
            for dirpath, _, filenames in os.walk(root)
            for name in filenames if name.endswith(extensions)]

def build_chunks(root, tokenizer, max_tokens=1024, overlap=128):
    files = _get_files(root)
    docs = [doc for f in files for doc in SimpleDirectoryReader(input_files=[str(f)]).load_data()]
    token_splitter = TokenTextSplitter(tokenizer=tokenizer, chunk_size=max_tokens, chunk_overlap=overlap)
    result = []
    try:
        code_splitter = CodeSplitter(language="cpp")
        nodes = code_splitter.get_nodes_from_documents(docs)
    except Exception:
        try:
            code_splitter = CodeSplitter(language="c")
            nodes = code_splitter.get_nodes_from_documents(docs)
        except Exception:
            return {"file": "", "chunk": 0, "code": ""}
    for node in nodes:
        for i, chunk in enumerate(token_splitter.split_text(node.text)):
            result.append({
                "file": node.metadata["file_path"],
                "chunk": i,
                "code": chunk
            })
    return result