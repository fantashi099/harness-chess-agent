"""Runtime disconnections must not hide confirmed sandbox termination."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import ServerDisconnectedError
import pytest

from assignment.env import Environment


@pytest.mark.parametrize("close_error", [None, ServerDisconnectedError("gone"), TimeoutError()])
def test_runtime_disconnect_is_nonfatal_only_after_termination(close_error):
    terminate = AsyncMock()
    sandbox = SimpleNamespace(terminate=SimpleNamespace(aio=terminate))
    env = object.__new__(Environment)
    env.deployment = SimpleNamespace(_sandbox=sandbox, stop=AsyncMock(side_effect=close_error))
    if close_error is None:
        env.stop()
    else:
        with pytest.warns(RuntimeWarning, match="sandbox termination succeeded"):
            env.stop()
    terminate.assert_awaited_once()
    assert env.deployment._sandbox is None
    assert env.deployment._runtime is None


def test_failed_termination_still_propagates():
    terminate = AsyncMock(side_effect=RuntimeError("termination failed"))
    env = object.__new__(Environment)
    env.deployment = SimpleNamespace(
        _sandbox=SimpleNamespace(terminate=SimpleNamespace(aio=terminate)),
        stop=AsyncMock(side_effect=ServerDisconnectedError("gone")))
    with pytest.raises(RuntimeError, match="termination failed"):
        env.stop()
    terminate.assert_awaited_once()


def test_programming_errors_are_not_suppressed():
    terminate = AsyncMock()
    env = object.__new__(Environment)
    env.deployment = SimpleNamespace(
        _sandbox=SimpleNamespace(terminate=SimpleNamespace(aio=terminate)),
        stop=AsyncMock(side_effect=ValueError("invalid shutdown state")))
    with pytest.raises(ValueError, match="invalid shutdown state"):
        env.stop()
    terminate.assert_awaited_once()
