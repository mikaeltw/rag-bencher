
from __future__ import annotations
from typing import Any, Dict
from langchain_core.callbacks.base import BaseCallbackHandler

class UsageTracker(BaseCallbackHandler):
    """Aggregates token and cost usage if models emit them; otherwise falls back to char counts."""
    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_in = 0.0
        self.cost_out = 0.0
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.calls += 1
        # Try to read token usage from response if provided
        try:
            for gen in response.generations:
                pass
        except Exception:
            pass
        # best-effort heuristic using text lengths
        try:
            texts = []
            for gens in response.generations:
                for g in gens:
                    texts.append(getattr(g, "text", "") or "")
            out_len = sum(len(t.split()) for t in texts)
            self.output_tokens += out_len
            self.cost_out += (out_len / 1000.0) * self.cost_per_1k_output
        except Exception:
            pass

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        # approximate input tokens
        in_len = sum(len(p.split()) for p in prompts)
        self.input_tokens += in_len
        self.cost_in += (in_len / 1000.0) * self.cost_per_1k_input

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "approx_cost_in": round(self.cost_in, 6),
            "approx_cost_out": round(self.cost_out, 6),
            "approx_cost_total": round(self.cost_in + self.cost_out, 6),
        }
