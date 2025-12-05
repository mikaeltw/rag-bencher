from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher import config

pytestmark = pytest.mark.gpu


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        config.load_config(str(path))
