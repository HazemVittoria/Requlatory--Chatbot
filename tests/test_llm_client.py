from __future__ import annotations

import pytest

from src.llm_client import DeterminismSettings, DeterministicLLMClient


def test_deterministic_defaults_are_forced():
    seen: dict = {}

    def _fake_request(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    client = DeterministicLLMClient(_fake_request, settings=DeterminismSettings())
    client.request(model="x", input="hello")

    assert seen["temperature"] == 0.2
    assert seen["top_p"] == 1.0
    assert seen["frequency_penalty"] == 0.0
    assert seen["presence_penalty"] == 0.0


def test_per_call_override_is_blocked():
    def _fake_request(**kwargs):
        return kwargs

    client = DeterministicLLMClient(_fake_request, settings=DeterminismSettings(), allow_param_overrides=False)

    with pytest.raises(ValueError):
        client.request(model="x", input="hello", temperature=0.9)


def test_fallback_temperature_zero_mode():
    seen: dict = {}

    def _fake_request(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    settings = DeterminismSettings(force_fallback=True)
    client = DeterministicLLMClient(_fake_request, settings=settings)
    client.request(model="x", input="hello")

    assert seen["temperature"] == 0.0
    assert seen["top_p"] == 1.0
    assert seen["frequency_penalty"] == 0.0
    assert seen["presence_penalty"] == 0.0
