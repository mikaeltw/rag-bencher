from __future__ import annotations

from pathlib import Path

import pytest

from rag_bencher.utils import cache

pytestmark = pytest.mark.gpu


def test_cache_get_returns_none_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "D", cache_dir, raising=False)
    assert cache.cache_get("model-z", "prompt-x") is None
