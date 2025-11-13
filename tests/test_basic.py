import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_import() -> None:
    import rag_bench

    assert isinstance(rag_bench.__version__, str)
