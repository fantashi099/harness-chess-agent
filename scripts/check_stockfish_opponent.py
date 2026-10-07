"""Free local API checks, optionally including a real Stockfish executable."""

import argparse
import shlex
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chess
import chess.engine
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chess_app/src"))

from assignment.chess_server import create_app
from assignment.stockfish_game import StockfishGame
from assignment.chess_sandbox import ChessSandbox


def check():
    engine = Mock(id={"name": "Stockfish test"})
    normal_reply = lambda board, limit, **kwargs: SimpleNamespace(move=next(iter(board.legal_moves)))
    engine.play.side_effect = normal_reply
    with patch("chess.engine.SimpleEngine.popen_uci", return_value=engine):
        with TestClient(create_app(opponent="stockfish", stockfish_elo=1500)) as client:
            engine.configure.assert_called_once_with(
                {"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": 1500})
            assert client.post("/api/move", json={"move": "e2e5"}).status_code == 400
            response = client.post("/api/move", json={"move": "e2e4"})
            response.raise_for_status()
            state = response.json()
            assert len(state["history"]) == 2 and state["turn"] == "white"
            assert state["opponent"]["mode"] == "stockfish"
            assert state["opponent"]["elo"] == 1500
            assert engine.play.call_count == 1
            assert engine.play.call_args.args[0].move_stack == [chess.Move.from_uci("e2e4")]

            # Engine errors and invalid replies must roll back the entire turn.
            for failure in (chess.engine.EngineTerminatedError("dead"), TimeoutError(),
                            SimpleNamespace(move=None), SimpleNamespace(move=chess.Move.from_uci("a1a8"))):
                before = client.get("/api/state").json()
                engine.play.side_effect = failure if isinstance(failure, Exception) else None
                engine.play.return_value = failure
                assert client.post("/api/move", json={"move": before["legal_moves"][0]}).status_code == 503
                assert client.get("/api/state").json() == before
            engine.play.side_effect = normal_reply
            old_game = engine.play.call_args.kwargs["game"]
            assert client.post("/api/reset").json()["history"] == []
            assert client.post("/api/move", json={"move": "d2d4"}).status_code == 200
            assert engine.play.call_args.kwargs["game"] is not old_game
            client.app.state.chess_game._board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
            calls = engine.play.call_count
            terminal = client.post("/api/move", json={"move": "f7g7"}).json()
            assert terminal["game_over"] and terminal["engine_move"] is None
            assert engine.play.call_count == calls
        engine.quit.assert_called_once()
        engine.close.assert_called_once()
    with TestClient(create_app()) as client:
        assert "opponent" not in client.get("/api/state").json()
        assert client.post("/api/move", json={"move": "e2e4"}).json()["engine_move"] == "e7e5"
    for value in (0, -1, 11, float("nan"), float("inf")):
        try:
            StockfishGame(move_time=value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid move time accepted")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=FileNotFoundError()):
        try:
            with TestClient(create_app(opponent="stockfish", stockfish_path="missing")):
                raise AssertionError("Missing engine silently accepted")
        except RuntimeError as exc:
            assert "Cannot start Stockfish" in str(exc)
    for mode in ("minimax", "stockfish"):
        image = Mock()
        with patch("assignment.chess_sandbox.build_testbed_image", return_value=image), \
             patch("assignment.chess_sandbox._with_assignment_files", side_effect=lambda img: img), \
             patch("assignment.chess_sandbox.Environment.__init__", return_value=None), \
             patch.object(ChessSandbox, "tunnel_url", return_value="http://test"), \
             patch.object(ChessSandbox, "_start_server"):
            sandbox = ChessSandbox(opponent=mode, stockfish_path="/engine dir/stockfish;literal", stockfish_elo=1500)
            assert image.apt_install.call_count == int(mode == "stockfish")
            sandbox.client.close()
        sandbox.execute = Mock(return_value={"returncode": 0})
        health = Mock(status=200)
        health.read.return_value = b'{"status": "ok"}'
        with patch("assignment.chess_sandbox.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = health
            sandbox._start_server(timeout=1)
        command = shlex.split(sandbox.execute.call_args.args[0])
        assert ("--opponent" in command) == (mode == "stockfish")
        if mode == "stockfish":
            assert command[command.index("--stockfish-path") + 1] == sandbox.stockfish_path
            assert command[command.index("--stockfish-elo") + 1] == "1500"
    print("Opponent checks passed: API, rollback, reset, terminal moves, sandbox and minimax default.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish-path", help="also verify this real executable")
    args = parser.parse_args()
    check()
    if args.stockfish_path:
        app = create_app(opponent="stockfish", stockfish_path=args.stockfish_path)
        with TestClient(app) as client:
            for move in ("e2e4", "g1f3"):
                result = client.post("/api/move", json={"move": move})
                result.raise_for_status()
                assert result.json()["turn"] == "white"
            print(f"Real engine passed: {result.json()['opponent']}")
        assert app.state.chess_game.engine is None
