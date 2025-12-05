from importlib.metadata import PackageNotFoundError, version

# Apply device policy at import time (before anything imports torch)
try:
    from .utils.hardware import apply_process_wide_policy

    _EFFECTIVE_DEVICE = apply_process_wide_policy()
except Exception:
    _EFFECTIVE_DEVICE = "auto"

try:
    __version__ = version("rag-bencher")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Public API surface (can expand later)
__all__ = ["__version__"]

try:
    from .config import load_config as load_config  # noqa: F401
except Exception:
    globals().pop("load_config", None)
else:
    __all__ = ["__version__", "load_config"]
