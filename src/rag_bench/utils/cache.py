import hashlib
import json
from pathlib import Path
from typing import Any, Final, Optional

D: Final[Path] = Path(".ragbencher_cache")
D.mkdir(exist_ok=True, parents=True)


def K(m: str, p: str) -> str:
    """Return a SHA256 hash key from model and parameter strings."""
    return hashlib.sha256((m + "||" + p).encode()).hexdigest()


def cache_get(m: str, p: str) -> Optional[Any]:
    f = D / (K(m, p) + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text("utf-8"))
        except Exception:
            return None
    return None


def cache_set(m: str, p: str, o: Any) -> None:
    f = D / (K(m, p) + ".json")
    f.write_text(json.dumps(o), "utf-8")
