import os
import tempfile
import textwrap

import pytest

from rag_bencher.config import load_config

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def write_tmp(text: str) -> str:
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
    fd.write(text)
    fd.close()
    return fd.name


def test_env_expansion_and_valid() -> None:
    os.environ["MODEL_NAME"] = "test-model"
    path = write_tmp(
        textwrap.dedent(
            """
        model:
          name: ${MODEL_NAME}
        retriever:
          k: 5
        data:
          paths: ["examples/data/sample.txt"]
    """
        )
    )
    cfg = load_config(path)
    assert cfg.model.name == "test-model"
    assert cfg.retriever.k == 5


def test_strict_validation_rejects_unknown_keys() -> None:
    bad = write_tmp(
        textwrap.dedent(
            """
        model:
          name: foo
          extra: nope
        retriever:
          k: 3
        data:
          paths: ["examples/data/sample.txt"]
    """
        )
    )
    with pytest.raises(SystemExit):
        load_config(bad)
