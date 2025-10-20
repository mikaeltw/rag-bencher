
from __future__ import annotations
from importlib.resources import files
from pathlib import Path

def get_resource_path(relative: str) -> str:
    """Return an absolute path to a packaged resource under rag_bench/resources/.."""
    base = files("rag_bench").joinpath("resources")
    # Convert to a real filesystem path (extracts from zips if needed)
    p = base.joinpath(*relative.split("/"))
    return str(Path(p))
