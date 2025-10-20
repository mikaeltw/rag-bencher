
import os, tempfile, textwrap
from rag_bench.config import load_config
import pytest

def write_tmp(text: str):
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
    fd.write(text)
    fd.close()
    return fd.name

def test_env_expansion_and_valid():
    os.environ["MODEL_NAME"] = "test-model"
    path = write_tmp(textwrap.dedent('''
        model:
          name: ${MODEL_NAME}
        retriever:
          k: 5
        data:
          paths: ["examples/data/sample.txt"]
    '''))
    cfg = load_config(path)
    assert cfg.model.name == "test-model"
    assert cfg.retriever.k == 5

def test_strict_validation_rejects_unknown_keys():
    bad = write_tmp(textwrap.dedent('''
        model:
          name: foo
          extra: nope
        retriever:
          k: 3
        data:
          paths: ["examples/data/sample.txt"]
    '''))
    with pytest.raises(SystemExit):
        load_config(bad)
