from typing import TYPE_CHECKING, Any, Dict, Optional

from .hardware import wants_cpu
from .torch_utils import cuda_available

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


# Centralized factory for HuggingFaceEmbeddings
def _preferred_device() -> str:
    if wants_cpu():
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def make_hf_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
) -> "HuggingFaceEmbeddings":
    """Create a HuggingFaceEmbeddings with device already set from global policy.

    Usage everywhere:
        from rag_bencher.utils.factories import make_hf_embeddings
        embed = make_hf_embeddings()
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # local import

    mk = dict(model_kwargs or {})
    # Ensure device is enforced once here
    mk.setdefault("device", _preferred_device())
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=mk,
        encode_kwargs=encode_kwargs or {},
    )
