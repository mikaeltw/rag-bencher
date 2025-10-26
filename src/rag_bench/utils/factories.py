from typing import TYPE_CHECKING, Any, Dict, Optional

from .hardware import wants_cpu

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


# Centralized factory for HuggingFaceEmbeddings
def make_hf_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
) -> "HuggingFaceEmbeddings":
    """Create a HuggingFaceEmbeddings with device already set from global policy.

    Usage everywhere:
        from rag_bench.utils.factories import make_hf_embeddings
        embed = make_hf_embeddings()
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # local import

    mk = dict(model_kwargs or {})
    # Ensure device is enforced once here
    mk.setdefault("device", "cpu" if wants_cpu() else "cuda")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=mk,
        encode_kwargs=encode_kwargs or {},
    )
