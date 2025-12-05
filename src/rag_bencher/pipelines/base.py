from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Tuple

from langchain_core.runnables import RunnableSerializable

BuildResult = Tuple[RunnableSerializable[str, str], Callable[[], Mapping[str, Any]]]


class RagPipeline(ABC):
    @abstractmethod
    def build(self) -> BuildResult:
        """Return a runnable chain and a debug callback for metadata inspection."""
        raise NotImplementedError
