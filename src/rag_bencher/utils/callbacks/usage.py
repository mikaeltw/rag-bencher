from typing import Any, Dict, List

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageTracker(BaseCallbackHandler):
    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0) -> None:
        self.calls: int = 0
        self.in_tok: int = 0
        self.out_tok: int = 0
        self.cpi: float = cost_per_1k_input
        self.cpo: float = cost_per_1k_output

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kw: Any) -> None:
        self.in_tok += sum(len(p.split()) for p in prompts)

    def on_llm_end(self, response: LLMResult, **kw: Any) -> None:
        self.calls += 1
        try:
            outs: List[str] = []
            for gens in response.generations:
                for g in gens:
                    outs.append(getattr(g, "text", "") or "")
            self.out_tok += sum(len(t.split()) for t in outs)
        except Exception:
            pass

    def summary(self) -> Dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.in_tok, "output_tokens": self.out_tok}
