import os
from functools import lru_cache

# Public env knob: auto|cuda|gpu|cpu  (default: auto)
ENV_KEY = "RAG_BENCH_DEVICE"


def _normalize(mode: str | None) -> str:
    mode = (mode or "auto").lower()
    return "cuda" if mode == "gpu" else mode


@lru_cache(None)
def effective_mode() -> str:
    """Resolve the global device mode once per process."""
    return _normalize(os.getenv(ENV_KEY, "auto"))


def apply_process_wide_policy() -> str:
    """If CPU is requested, hide GPUs before torch/sentence-transformers import."""
    mode = effective_mode()
    if mode == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return mode


def wants_cpu() -> bool:
    return effective_mode() == "cpu"
