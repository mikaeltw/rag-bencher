import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_imports() -> None:
    import rag_bench

    assert hasattr(rag_bench, "__all__")
