from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from rag_bencher.providers.aws import auth as aws_auth
from rag_bencher.providers.azure import auth as azure_auth
from rag_bencher.providers.gcp import auth as gcp_auth

pytestmark = [pytest.mark.unit, pytest.mark.offline]


def _force_missing(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ImportError(f"missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize(
    ("module_name", "func"),
    [
        ("langchain_aws", aws_auth.is_installed),
        ("langchain_openai", azure_auth.is_installed),
        ("langchain_google_vertexai", gcp_auth.is_installed),
    ],
)
def test_is_installed_returns_true_when_module_present(
    monkeypatch: pytest.MonkeyPatch, module_name: str, func: Callable[[], bool]
) -> None:
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace())
    assert func() is True


@pytest.mark.parametrize(
    ("module_name", "func"),
    [
        ("langchain_aws", aws_auth.is_installed),
        ("langchain_openai", azure_auth.is_installed),
        ("langchain_google_vertexai", gcp_auth.is_installed),
    ],
)
def test_is_installed_returns_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch, module_name: str, func: Callable[[], bool]
) -> None:
    _force_missing(monkeypatch, module_name)
    assert func() is False
