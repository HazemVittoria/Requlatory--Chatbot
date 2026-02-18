from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

_LOCKED_PARAM_KEYS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DeterminismSettings:
    temperature: float = 0.2
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_temperature: float = 0.0
    force_fallback: bool = False

    @classmethod
    def from_env(cls) -> "DeterminismSettings":
        return cls(
            force_fallback=_env_truthy("LLM_FORCE_FALLBACK_TEMPERATURE_ZERO", default=False),
        )

    def as_request_params(self) -> dict[str, float]:
        return {
            "temperature": self.fallback_temperature if self.force_fallback else self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class DeterministicLLMClient:
    """
    Deterministic wrapper for all LLM calls.
    - Forces global sampling defaults.
    - Blocks per-call overrides of determinism keys unless explicitly enabled.
    """

    def __init__(
        self,
        request_fn: Callable[..., Any],
        settings: DeterminismSettings | None = None,
        allow_param_overrides: bool = False,
    ):
        self._request_fn = request_fn
        self._settings = settings or DeterminismSettings.from_env()
        self._allow_param_overrides = bool(allow_param_overrides)

    @property
    def settings(self) -> DeterminismSettings:
        return self._settings

    def request(self, **kwargs: Any) -> Any:
        if not self._allow_param_overrides:
            bad = [k for k in _LOCKED_PARAM_KEYS if k in kwargs]
            if bad:
                keys = ", ".join(sorted(bad))
                raise ValueError(
                    f"Per-call override blocked for deterministic keys: {keys}. "
                    "Set deterministic params only in DeterminismSettings."
                )

        params = dict(kwargs)
        params.update(self._settings.as_request_params())
        return self._request_fn(**params)

