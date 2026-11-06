"""Evaluation-only waiting through model endpoint outages without discarding the rollout."""

from __future__ import annotations

import asyncio
import logging
import random
import time

import openai
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class ModelRetryConfig(BaseModel):
    """Retry model calls that fail with these HTTP statuses, backing off exponentially.

    Training leaves this unset: there the rollout, not the HTTP call, is the retry unit.
    """

    model_config = ConfigDict(extra="forbid")

    status_codes: tuple[int, ...] = Field(default=(500, 502, 503, 504), min_length=1)
    retry_connection_errors: bool = True
    initial_backoff_seconds: float = Field(default=30.0, gt=0)
    max_backoff_seconds: float = Field(default=120.0, gt=0)
    max_total_wait_seconds: float = Field(default=1200.0, gt=0)
    # A replica that just restarted falls over again if every waiting rollout resends its
    # long-context request at once, so rollouts released by the gate resume over this window.
    release_spread_seconds: float = Field(default=120.0, ge=0)

    @model_validator(mode="after")
    def ordered_backoff(self) -> ModelRetryConfig:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        return self


class EndpointGate:
    """Shared by one harness's rollouts: while any rollout waits out an outage, the others
    hold their next model call instead of hammering the endpoint and burning their budgets."""

    def __init__(self) -> None:
        self._healthy = asyncio.Event()
        self._healthy.set()

    def mark_down(self) -> None:
        self._healthy.clear()

    def mark_up(self) -> None:
        self._healthy.set()

    async def wait_up(self, timeout: float) -> None:
        await asyncio.wait_for(self._healthy.wait(), timeout)

    @property
    def is_up(self) -> bool:
        return self._healthy.is_set()


class RetryingModel:
    """Wraps a Feather model; the agent loop, sandbox and history stay alive while it waits."""

    def __init__(self, model, config: ModelRetryConfig, gate: EndpointGate) -> None:
        self.model = model
        self.config = config
        self.gate = gate
        self.waited_seconds = 0.0
        self.retries = 0

    @property
    def prompt_token_budget(self):
        return self.model.prompt_token_budget

    async def complete(self, messages, **kwargs):
        backoff = self.config.initial_backoff_seconds
        holding_gate = False
        while True:
            # Wait out an outage another rollout is probing; never wait on a gate we closed.
            if not holding_gate and not self.gate.is_up:
                await self._wait(self._remaining(), gate=True)
                spread = random.uniform(0, self.config.release_spread_seconds)
                await self._wait(min(spread, max(self._remaining(), 0.0)), gate=False)
            try:
                call = await self.model.complete(messages, **kwargs)
            except Exception as exc:
                if not self._retryable(exc):
                    raise
                remaining = self._remaining()
                if remaining <= 0:
                    self.gate.mark_up()
                    raise
                self.gate.mark_down()
                holding_gate = True
                self.retries += 1
                # Jitter keeps rollouts that failed together from retrying in lockstep.
                delay = min(backoff * random.uniform(0.5, 1.0), remaining)
                logger.warning(
                    "model endpoint unavailable (%s); retry %d in %.0fs (waited %.0fs of %.0fs)",
                    exc,
                    self.retries,
                    delay,
                    self.waited_seconds,
                    self.config.max_total_wait_seconds,
                )
                try:
                    await self._wait(delay, gate=False)
                except BaseException:
                    self.gate.mark_up()
                    raise
                backoff = min(backoff * 2, self.config.max_backoff_seconds)
                continue
            self.gate.mark_up()
            return call

    def _retryable(self, exc: Exception) -> bool:
        if isinstance(exc, openai.APITimeoutError):
            # A slow generation is not an outage; retrying would double its cost.
            return False
        if isinstance(exc, openai.APIConnectionError):
            return self.config.retry_connection_errors
        return (
            isinstance(exc, openai.APIStatusError)
            and exc.status_code in self.config.status_codes
        )

    def _remaining(self) -> float:
        return self.config.max_total_wait_seconds - self.waited_seconds

    async def _wait(self, seconds: float, *, gate: bool) -> None:
        start = time.monotonic()
        try:
            if gate:
                try:
                    await self.gate.wait_up(max(seconds, 0.0))
                except TimeoutError:
                    pass
            else:
                await asyncio.sleep(seconds)
        finally:
            self.waited_seconds += time.monotonic() - start
