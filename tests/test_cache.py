from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_bencher.utils import cache


@pytest.mark.offline
def test_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "D", cache_dir, raising=False)
    key = cache.K("model-x", "params-y")
    assert len(key) == 64

    payload = {"answer": 42}
    cache.cache_set("model-x", "params-y", payload)
    loaded = cache.cache_get("model-x", "params-y")
    assert loaded == payload
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8")) == payload


@pytest.mark.offline
def test_cache_get_handles_corrupt_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "D", cache_dir, raising=False)
    key = cache.K("model-y", "bad-params")
    f = cache_dir / f"{key}.json"
    f.write_text("{not json", encoding="utf-8")
    assert cache.cache_get("model-y", "bad-params") is None
