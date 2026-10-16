"""Optional CPU opponent; imported only when Stockfish is selected."""

from __future__ import annotations

from shutil import which

import chess
import chess.engine
from chess_app.game import ChessGame, InvalidMoveError


class OpponentError(RuntimeError):
    """A failed engine reply; the human move has been rolled back."""


class StockfishGame(ChessGame):
    def __init__(self, path: str | None = None, move_time: float = 0.2,
                 elo: int | None = None):
        if not 0 < move_time <= 10:
            raise ValueError("Stockfish move time must be between 0 and 10 seconds.")
        super().__init__()
        self.path = path or which("stockfish") or "/usr/games/stockfish"
        self.move_time, self.elo = move_time, elo
        self.engine: chess.engine.SimpleEngine | None = None
        self._engine_game = object()

    def start(self) -> None:
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=5)
            options = {"Threads": 1, "Hash": 64, "UCI_LimitStrength": self.elo is not None,
                       "Skill Level": 20}
            if self.elo is not None:
                options.update(UCI_LimitStrength=True, UCI_Elo=self.elo)
            self.engine.configure(options)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Cannot start Stockfish at {self.path!r}: {exc}. "
                "Install Stockfish on the server host or pass --stockfish-path."
            ) from exc

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except (chess.engine.EngineError, TimeoutError):
                pass
            finally:
                self.engine.close()
                self.engine = None

    def reset(self) -> dict:
        with self._lock:
            self._board.reset()
            self._history.clear()
            self._last_move = None
            self._engine_game = object()
            return self._state_unlocked()

    def _choose_engine_move(self) -> chess.Move:
        if self.engine is None:
            raise OpponentError("Stockfish is not running.")
        try:
            reply = self.engine.play(
                self._board.copy(stack=True), chess.engine.Limit(time=self.move_time),
                game=self._engine_game,
            ).move
        except (chess.engine.EngineError, TimeoutError) as exc:
            raise OpponentError(f"Stockfish failed to reply: {exc}") from exc
        if reply is None or reply not in self._board.legal_moves:
            raise OpponentError("Stockfish did not return a legal move.")
        return reply

    def play_human_move(self, move_text: str) -> dict:
        # Keep this transaction independent of the pinned app's terminal-move bug.
        with self._lock:
            if self._board.outcome(claim_draw=True) is not None:
                raise InvalidMoveError("The game is over. Start a new game to continue.")
            if self._board.turn != chess.WHITE:
                raise InvalidMoveError("It is not White's turn.")
            move = self._parse_move(move_text)
            previous_board = self._board.copy(stack=True)
            previous_last_move, history_length = self._last_move, len(self._history)
            try:
                self._record_and_push(move)
                human_state = self._state_unlocked()
                reply = None
                if not human_state["game_over"]:
                    reply = self._choose_engine_move()
                    self._record_and_push(reply)
                return dict(self._state_unlocked(), human_move=move.uci(),
                            human_state=human_state,
                            engine_move=reply.uci() if reply else None)
            except Exception:
                self._board = previous_board
                del self._history[history_length:]
                self._last_move = previous_last_move
                raise

    def _state_unlocked(self) -> dict:
        return dict(super()._state_unlocked(), opponent={
            "mode": "stockfish", "name": self.engine.id.get("name") if self.engine else None,
            "move_time": self.move_time, "threads": 1, "hash_mib": 64, "elo": self.elo,
        })
