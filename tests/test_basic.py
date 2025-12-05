import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_import() -> None:
    import rag_bencher

    assert isinstance(rag_bencher.__version__, str)
