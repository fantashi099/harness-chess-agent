"""Run the chess server in a Modal sandbox and play moves against it.

This is the part 2 surface. It does not know anything about fixing the bug: it
takes an optional patch (the fix produced in part 1), applies it to the
testbed, serves the app, and exposes the game over HTTP.
"""

from __future__ import annotations

import base64
import json
import shlex
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import httpx

from assignment.agent.chess_tools import CHESS_PORT
from assignment.task import Task
from assignment.env import Environment
from assignment.chess_server import add_opponent_arguments, opponent_kwargs
from assignment.utils.image import build_testbed_image

TESTBED = "/testbed"
SERVER_LOG = "/tmp/chess-server.log"
DEFAULT_TASK = Path(__file__).resolve().parents[2] / "tasks" / "chess-terminal-move"

class IllegalMove(Exception):
    """The server rejected a move as illegal or malformed."""

# The agent's own tool implementations, so a model-written snippet can run in
# here against the local server instead of in the agent process.
SANDBOX_FILES = (
    Path(__file__).with_name("chess_server.py"),
    Path(__file__).parent / "agent" / "chess_tools.py",
    Path(__file__).with_name("sandbox_python.py"),
    Path(__file__).parent / "agent" / "chess_policy.py",
    Path(__file__).with_name("fixed_chess_server.py"),
    Path(__file__).with_name("stockfish_game.py"),
)

def _with_assignment_files(image, search_policy=False, harness_policy=False):
    """Copy the files the sandbox runs into /opt/assignment."""

    for source in SANDBOX_FILES:
        image = image.add_local_file(
            str(source),
            f"/opt/assignment/{source.name}",
            copy=True,  # SWE-ReX adds its runtime build layer afterwards.
        )
    if search_policy:
        source=Path(__file__).parent/"agent/chess_policy_search.py"
        image=image.add_local_file(str(source),f"/opt/assignment/{source.name}",copy=True)
    if harness_policy:
        source = Path(__file__).parent / "agent/chess_policy_harness.py"
        image = image.add_local_file(str(source), f"/opt/assignment/{source.name}", copy=True)
    return image

class ChessSandbox(Environment):
    """A chess server hosted in an isolated Modal sandbox.

    Built from the same testbed image the evaluation harness uses, so the code
    under test and the code being played are identical. Without a patch the
    sandbox runs the unfixed engine, which still plays normally right up until
    a game-ending move.
    """

    def __init__(
        self,
        task: str | Path | Task | None = None,
        patch: str | Path | None = None,
        port: int = CHESS_PORT,
        strict: bool = True,
        startup_timeout: float = 300,
        runtime_timeout: float = 600,
        deployment_timeout: float = 1800,
        server_timeout: float = 30,
        chess_policy: str | None = None,
        max_nodes: int = 6000,
        search_depth: int = 4,
        opponent: str = "minimax",
        stockfish_path: str | None = None,
        stockfish_time: float = 0.2,
        stockfish_elo: int | None = None,
    ):
        """Launch the sandbox and block until the server answers.

        Args:
            task: The task whose testbed to serve, as a directory or a loaded
                `Task`. Defaults to the chess task.
            patch: A unified diff applied to the testbed before the server
                starts, normally the fix from part 1.
            port: Port to serve on, forwarded through an encrypted tunnel.
            strict: Refuse to build when the local checkout does not match the
                task's base commit.
            startup_timeout: Seconds to wait for the SWE-ReX runtime.
            runtime_timeout: Seconds a single command may run.
            deployment_timeout: Seconds the sandbox may stay alive.
            server_timeout: Seconds to wait for `/health` to succeed.
            opponent: Black's engine, either the default minimax or Stockfish.
            stockfish_path: Executable path inside Modal, not on the laptop.
            stockfish_time: Seconds per Black move (maximum 10).
            stockfish_elo: Optional strength limit supported by the engine.
        """
        if chess_policy not in {None, "search", "harness"}:
            raise ValueError("Unsupported chess policy")
        self.chess_policy = chess_policy
        self.max_nodes = max_nodes
        self.search_depth=search_depth
        if type(search_depth) is not int or not 1<=search_depth<=12:
            raise ValueError("Search depth must be 1..12")
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if opponent not in {"minimax", "stockfish"}:
            raise ValueError("opponent must be 'minimax' or 'stockfish'.")
        if opponent == "stockfish" and not 0 < stockfish_time <= 10:
            raise ValueError("Stockfish move time must be between 0 and 10 seconds.")
        self.opponent = opponent
        self.stockfish_path, self.stockfish_time, self.stockfish_elo = (
            stockfish_path, stockfish_time, stockfish_elo)
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid port: {port}")

        self.task = task if isinstance(task, Task) else Task.load(task or DEFAULT_TASK)
        self.port = port
        self.server_url = ""
        self._client: httpx.Client | None = None

        image = build_testbed_image(self.task, strict=strict)
        if opponent == "stockfish":
            image = image.apt_install("stockfish")
        super().__init__(
            image=(_with_assignment_files(image,search_policy=True,harness_policy=chess_policy=="harness") if chess_policy in {"search", "harness"}
                   else _with_assignment_files(image)),
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            modal_sandbox_kwargs={"encrypted_ports": [port]},
        )

        try:
            if patch is not None:
                self._apply_patch(patch)
            self.server_url = self.tunnel_url(port).rstrip("/")
            self._start_server(server_timeout)
            self._client = httpx.Client(base_url=self.server_url, timeout=30)
        except Exception:
            self.stop()
            raise

    def _apply_patch(self, patch: str | Path) -> None:
        """Apply the part 1 fix to the testbed before serving it."""
        # Accept either a path to a diff or the diff itself; a diff always has
        # newlines, a path never usefully does.
        if isinstance(patch, Path):
            text = patch.read_text()
        else:
            text = patch if "\n" in patch else Path(patch).read_text()
        encoded = base64.b64encode(text.encode()).decode()

        written = self.execute(f"echo {encoded} | base64 -d > /tmp/fix.diff")
        if written["returncode"] != 0:
            raise RuntimeError(f"Could not upload the patch: {written['stderr'].strip()}")

        applied = self.execute("git apply -v /tmp/fix.diff", cwd=TESTBED)
        if applied["returncode"] != 0:
            raise RuntimeError(f"Patch did not apply:\n{applied['stderr'].strip()}")

    def _start_server(self, timeout: float) -> None:
        server_script = "fixed_chess_server.py" if self.chess_policy else "chess_server.py"
        opponent_args = []
        if self.opponent == "stockfish":
            opponent_args = ["--opponent", "stockfish", "--stockfish-time", str(self.stockfish_time)]
            if self.stockfish_path is not None:
                opponent_args += ["--stockfish-path", self.stockfish_path]
            if self.stockfish_elo is not None:
                opponent_args += ["--stockfish-elo", str(self.stockfish_elo)]
        if self.chess_policy in {"search", "harness"}:
            opponent_args += ["--policy",self.chess_policy,"--search-depth",str(self.search_depth),"--max-nodes",str(self.max_nodes)]
        command = (
            f"nohup python /opt/assignment/{server_script} "
            f"--host 0.0.0.0 --port {self.port} "
            f"{shlex.join(opponent_args)} "
            f"> {SERVER_LOG} 2>&1 < /dev/null & echo $!"
        )
        result = self.execute(command, cwd=TESTBED)
        if result["returncode"] != 0:
            raise RuntimeError(f"Could not start the chess server: {result['stderr'].strip()}")

        deadline = time.monotonic() + timeout
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{self.server_url}/health", timeout=3) as response:
                    payload = json.load(response)
                if response.status == 200 and payload == {"status": "ok"}:
                    return
                last_error = f"unexpected health response: {payload!r}"
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(0.25)

        logs = self.execute(f"tail -n 80 {SERVER_LOG}")
        log_text = (logs["stdout"] or logs["stderr"]).strip()
        raise RuntimeError(
            f"Chess server did not become ready within {timeout:g}s ({last_error}).\n{log_text}"
        )

    # -- Playing ------------------------------------------------------------
    # A chess agent drives the game through these. They speak the same HTTP API
    # the browser frontend uses.

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("The sandbox is not running.")
        return self._client

    def state(self) -> dict:
        """Return the current board state, legal moves, history, and result."""
        response = self.client.get("/api/state")
        response.raise_for_status()
        return response.json()

    def play(self, move: str) -> dict:
        """Play one move as White and return the resulting state.

        Args:
            move: A move in UCI notation, such as `e2e4`.

        Returns:
            The new state, including `human_move` and the opponent's
            `engine_move`. `engine_move` is None once the game has ended.

        Raises:
            IllegalMove: The move was rejected. An agent should treat this as a
                recoverable mistake and choose another move.
        """
        response = self.client.post("/api/move", json={"move": move})
        if response.status_code == 400:
            raise IllegalMove(response.json().get("detail", f"{move} was rejected"))
        response.raise_for_status()
        return response.json()

    def reset(self) -> dict:
        """Start a new game and return the opening state."""
        response = self.client.post("/api/reset")
        response.raise_for_status()
        return response.json()

    def stop(self, timeout: float = 10) -> None:
        """Close the HTTP client, then tear the sandbox down."""
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop(timeout=timeout)

def main() -> None:
    """Launch a chess sandbox and keep it alive until the user exits."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve the chess app from a Modal sandbox")
    parser.add_argument("--patch", type=Path, help="Fix to apply before serving, e.g. the part 1 output")
    parser.add_argument("--task", type=Path, help="Task directory to serve the testbed of")
    add_opponent_arguments(parser)
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=1800,
        help="maximum lifetime of the Modal sandbox in seconds",
    )
    args = parser.parse_args()

    with ChessSandbox(
        task=args.task,
        patch=args.patch,
        deployment_timeout=args.sandbox_timeout,
        **opponent_kwargs(args),
    ) as sandbox:
        print(f"Chess sandbox ready: {sandbox.server_url}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping chess sandbox…", flush=True)

if __name__ == "__main__":
    main()
