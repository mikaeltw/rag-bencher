
from abc import ABC, abstractmethod

class RagPipeline(ABC):
    @abstractmethod
    def build(self):
        """Return a Runnable chain that accepts a user question (str) and returns str."""
        raise NotImplementedError
