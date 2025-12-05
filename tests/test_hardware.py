from __future__ import annotations

import os

import pytest

from rag_bencher.utils import hardware


@pytest.mark.unit
def test_normalize_and_wants_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware.effective_mode.cache_clear()
    assert hardware._normalize(None) == "auto"
    assert hardware._normalize("GPU") == "cuda"
    assert hardware._normalize("cpu") == "cpu"
    monkeypatch.setenv(hardware.ENV_KEY, "CPU")
    hardware.effective_mode.cache_clear()
    assert hardware.wants_cpu() is True


@pytest.mark.unit
def test_effective_mode_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware.effective_mode.cache_clear()
    monkeypatch.setenv(hardware.ENV_KEY, "cuda")
    assert hardware.effective_mode() == "cuda"
    monkeypatch.setenv(hardware.ENV_KEY, "gpu")
    hardware.effective_mode.cache_clear()
    assert hardware.effective_mode() == "cuda"


@pytest.mark.unit
def test_apply_process_wide_policy_masks_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware.effective_mode.cache_clear()
    monkeypatch.setenv(hardware.ENV_KEY, "cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    mode = hardware.apply_process_wide_policy()
    assert mode == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.unit
def test_apply_process_wide_policy_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware.effective_mode.cache_clear()
    monkeypatch.setenv(hardware.ENV_KEY, "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    hardware.apply_process_wide_policy()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.unit
def test_apply_process_wide_policy_passthrough_for_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    hardware.effective_mode.cache_clear()
    monkeypatch.setenv(hardware.ENV_KEY, "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    mode = hardware.apply_process_wide_policy()
    assert mode == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
