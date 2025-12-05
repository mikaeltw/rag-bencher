from __future__ import annotations

import pytest

from rag_bencher.pipelines import base

pytestmark = pytest.mark.gpu


def test_rag_pipeline_base_build_not_implemented() -> None:
    class DummyPipeline(base.RagPipeline):
        def build(self) -> base.BuildResult:
            raise AssertionError("should not call subclass build")

    dummy = DummyPipeline()
    with pytest.raises(NotImplementedError):
        base.RagPipeline.build(dummy)
