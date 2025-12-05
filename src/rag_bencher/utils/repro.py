import hashlib
import random
import time


def set_seeds(s: int = 42) -> None:
    random.seed(s)
    try:
        import numpy as np

        np.random.seed(s)
    except Exception:
        pass


def make_run_id() -> str:
    return hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:10]
