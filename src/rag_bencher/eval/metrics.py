from collections import Counter


def _tok(s: str) -> list[str]:
    """Tokenize a string: lowercase, strip punctuation, and split on whitespace."""
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in s)
    return [t for t in normalized.split() if t]


def lexical_f1(p: str, r: str) -> float:
    P = _tok(p)
    R = _tok(r)
    if not P or not R:
        return 0.0
    Pc, Rc = Counter(P), Counter(R)
    ov = sum(min(Pc[t], Rc[t]) for t in set(Pc) | set(Rc))
    pr = ov / max(1, sum(Pc.values()))
    rc = ov / max(1, sum(Rc.values()))
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def bow_cosine(p: str, r: str) -> float:
    P = Counter(_tok(p))
    R = Counter(_tok(r))
    if not P or not R:
        return 0.0
    keys = set(P) | set(R)
    dp = sum(P[k] * R[k] for k in keys)
    import math

    return (
        0.0
        if not P or not R
        else dp / (math.sqrt(sum(v * v for v in P.values())) * math.sqrt(sum(v * v for v in R.values())))
    )


def context_recall(context: str, reference: str) -> float:
    """Simple token-level recall of reference content in the provided context.

    Returns the fraction of unique tokens from `reference` that appear in `context`.
    Range: [0.0, 1.0].
    """
    ref_tokens = set(_tok(reference))
    if not ref_tokens:
        return 0.0
    ctx_tokens = set(_tok(context))
    hits = len(ref_tokens & ctx_tokens)
    return hits / len(ref_tokens)
