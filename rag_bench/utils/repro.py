
from __future__ import annotations
import os, random
from pathlib import Path

def set_seeds(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def make_run_id():
    import time, hashlib
    raw = f"{time.time_ns()}_{os.getpid()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]
