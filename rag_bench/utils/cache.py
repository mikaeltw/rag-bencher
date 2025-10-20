
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(".ragbench_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def _key_for(model: str, prompt: str) -> str:
    h = hashlib.sha256((model + "||" + prompt).encode("utf-8")).hexdigest()
    return h

def cache_get(model: str, prompt: str):
    k = _key_for(model, prompt)
    p = CACHE_DIR / (k + ".json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def cache_set(model: str, prompt: str, output: Any):
    k = _key_for(model, prompt)
    p = CACHE_DIR / (k + ".json")
    p.write_text(json.dumps(output), encoding="utf-8")
