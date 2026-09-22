"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        fen = args.get("fen")
        move = args.get("move")
        if not isinstance(fen, str):
            raise ValueError("Argument 'fen' must be a string")
        if move and not isinstance(move, str):
            raise ValueError("Argument 'move' must be a UCI string, e.g. 'e2e4'.")
        payload = {"fen": fen}
        if move:
            payload["move"] = move
        return json.dumps(_request_state(client, "POST", "/api/simulate", json=payload))
    except httpx.HTTPError as exc:
        return f"<chess_error>Transport failure: {exc}</chess_error>"
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        move = args.get("move")
        if not isinstance(move, str):
            raise ValueError("Argument 'move' must be a UCI string, e.g. 'e2e4'.")
        return json.dumps(_request_state(client, "POST", "/api/move", json={"move": move}))
    except httpx.HTTPError as exc:
        return f"<chess_error>Transport failure: {exc}</chess_error>"
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    def format_error_code(message):
        return f"<chess_error>{message}</chess_error>"

    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        code = args.get("code")
        if not isinstance(code, str):
            raise ValueError("Argument 'code' must be a string")
        encoded = base64.b64encode(code.encode()).decode() # base64 alphabet shell-safe
        result = env.execute(f"python /opt/assignment/sandbox_python.py {port} {encoded}")
        status = result["returncode"]
        if status != 0:
            detail = result.get("exception_info") or result.get("output") or "sandbox command failed"
            return format_error_code(detail)
        return result["output"]
    except Exception as error:
        return format_error_code(str(error))


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("Arguments must be a JSON object.")
        name = args.get("name")
        if not isinstance(name, str):
            raise ValueError("Argument 'name' must be a string, the skill's name.")
        skill = skills.get(name)
        if skill is None:
            raise ValueError(
                f"unknown skill '{name}'. Available: {', '.join(skills)}"
            )
        return skill["content"]
    except json.JSONDecodeError as exc: # malformed args
        return f"<chess_error>Invalid JSON arguments: {exc}</chess_error>"
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"



def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
