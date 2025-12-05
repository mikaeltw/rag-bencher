from __future__ import annotations

import random

import pytest

from rag_bencher.utils import repro


@pytest.mark.unit
def test_set_seeds_makes_random_deterministic() -> None:
    repro.set_seeds(123)
    first = random.random()
    repro.set_seeds(123)
    second = random.random()
    assert first == second


@pytest.mark.unit
def test_make_run_id_returns_hex() -> None:
    run_id = repro.make_run_id()
    other = repro.make_run_id()
    assert len(run_id) == 10
    int(run_id, 16)  # should not raise
    assert run_id != other
