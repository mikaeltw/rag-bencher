
from __future__ import annotations
from collections import Counter
from typing import List, Tuple

def _tokens(s: str) -> List[str]:
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def lexical_f1(pred: str, ref: str) -> float:
    """Token-level F1 score between prediction and reference (lexical)."""
    p = _tokens(pred)
    r = _tokens(ref)
    if not p or not r:
        return 0.0
    p_counts = Counter(p)
    r_counts = Counter(r)
    overlap = sum(min(p_counts[t], r_counts[t]) for t in set(p_counts) | set(r_counts))
    precision = overlap / max(1, sum(p_counts.values()))
    recall = overlap / max(1, sum(r_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bow_cosine(pred: str, ref: str) -> float:
    """Cosine similarity over bag-of-words counts (no external models)."""
    p = Counter(_tokens(pred))
    r = Counter(_tokens(ref))
    if not p or not r:
        return 0.0
    keys = set(p) | set(r)
    dp = sum(p[k] * r[k] for k in keys)
    p_norm = sum(v*v for v in p.values()) ** 0.5
    r_norm = sum(v*v for v in r.values()) ** 0.5
    if p_norm == 0 or r_norm == 0:
        return 0.0
    return dp / (p_norm * r_norm)

def context_recall(reference: str, retrieved_text: str) -> float:
    """Heuristic: fraction of top-N reference tokens that appear in retrieved_text."""
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return 0.0
    # take top-N frequent tokens from reference (excluding very short tokens)
    counts = Counter(t for t in ref_tokens if len(t) > 2)
    if not counts:
        return 0.0
    top = [t for t, _ in counts.most_common(10)]
    retrieved_set = set(_tokens(retrieved_text))
    hits = sum(1 for t in top if t in retrieved_set)
    return hits / len(top)
