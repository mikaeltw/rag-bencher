from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .hardware import wants_cpu

if TYPE_CHECKING:
    import torch


def cuda_available() -> bool:
    """Return True when torch sees a usable CUDA device without surfacing warnings."""
    try:
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_str() -> str:
    try:
        if wants_cpu():
            return "cpu"
        return "cuda" if cuda_available() else "cpu"
    except Exception:
        return "cpu"


def to_device(x: Any) -> Any:
    """Move a torch tensor/module to the global device (no-op if torch missing)."""
    try:
        import torch  # noqa: F401

        dev = device_str()
        if hasattr(x, "to"):
            return x.to(dev)
    except Exception:
        pass
    return x


def new_tensor(data: Any, *, dtype: "torch.dtype | None" = None) -> "torch.Tensor":
    """Create a tensor on the global device."""
    import torch

    return torch.as_tensor(data, dtype=dtype, device=device_str())
