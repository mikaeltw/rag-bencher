import pytest

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def test_imports() -> None:
    import rag_bencher

    assert hasattr(rag_bencher, "__all__")
