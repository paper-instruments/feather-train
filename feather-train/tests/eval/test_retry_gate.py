import asyncio

import httpx
import pytest
from openai import APIConnectionError

from feather_train.model_retry import EndpointGate, ModelRetryConfig, RetryingModel


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_probe_releases_waiting_rollouts(cancel: bool) -> None:
    probe_started = asyncio.Event()
    finish_probe = asyncio.Event()

    class FailingModel:
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, messages: tuple) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
            probe_started.set()
            await finish_probe.wait()
            raise ValueError("invalid response")

    gate = EndpointGate()
    model = RetryingModel(
        FailingModel(),
        ModelRetryConfig(initial_backoff_seconds=0.001, release_spread_seconds=0),
        gate,
    )
    probe = asyncio.create_task(model.complete(()))
    try:
        await asyncio.wait_for(probe_started.wait(), timeout=1)
        assert not gate.is_up
        if cancel:
            probe.cancel()
        else:
            finish_probe.set()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await probe
        await gate.wait_up(timeout=0.1)
    finally:
        probe.cancel()
        await asyncio.gather(probe, return_exceptions=True)
