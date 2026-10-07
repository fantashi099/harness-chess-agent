"""Provided Part 3 server extension; leaves the pinned Part 1 testbed untouched.

Run this file inside the chess testbed, or locally with chess_app/src on
PYTHONPATH. Simulation reconstructs a position from FEN, never the live game.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import chess
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fen: str = Field(min_length=1, max_length=200)
    move: str | None = Field(default=None, min_length=4, max_length=5)

def simulate_position(fen: str, move: str | None = None) -> dict:
    """Inspect a position or apply exactly one legal ply, for either color.

    FEN preserves turn, castling, en passant, and move counters, but not a
    repetition history. Repetition-based draws cannot be inferred here.
    """
    try:
        if len(fen.split()) != 6:
            raise ValueError("Expected all six FEN fields.")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("The FEN describes an invalid chess position.")
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc

    # Unlike the live game, no repetition history or automatic bot is involved.
    outcome = board.outcome(claim_draw=True)
    applied_move = None
    if move is not None:
        if outcome is not None:
            raise ValueError("Cannot move in a terminal position.")
        try:
            candidate = chess.Move.from_uci(move)
        except ValueError as exc:
            raise ValueError("Move must use UCI notation, including a promotion suffix.") from exc
        if candidate not in board.legal_moves:
            raise ValueError(f"{move} is not legal in the supplied position.")
        board.push(candidate)
        applied_move = candidate.uci()
        outcome = board.outcome(claim_draw=True)

    winner = None if outcome is None or outcome.winner is None else (
        "white" if outcome.winner else "black"
    )
    turn = "white" if board.turn else "black"
    return {
        "fen": board.fen(en_passant="fen"),
        "squares": {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
        "turn": turn,
        "legal_moves": sorted(m.uci() for m in board.legal_moves) if outcome is None else [],
        "in_check": board.is_check(),
        "game_over": outcome is not None,
        "winner": winner,
        "result": outcome.result() if outcome else "*",
        "termination": outcome.termination.name.lower() if outcome else None,
        "status": (f"{winner.title()} wins" if winner else "Draw") if outcome else f"{turn.title()} to move",
        "applied_move": applied_move,
        "simulation": True,
        "repetition_history_available": False,
    }

def add_opponent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--opponent", choices=["minimax", "stockfish"], default="minimax")
    parser.add_argument("--stockfish-path", help="engine executable on the server host")
    parser.add_argument("--stockfish-time", type=float, default=0.2,
                        help="seconds per Black move, greater than 0 and at most 10")
    parser.add_argument("--stockfish-elo", type=int, help="limit strength to the engine's supported Elo range")


def opponent_kwargs(args: argparse.Namespace) -> dict:
    return {key: getattr(args, key) for key in
            ("opponent", "stockfish_path", "stockfish_time", "stockfish_elo")}


def create_app(game=None, *, opponent="minimax", stockfish_path=None,
               stockfish_time=0.2, stockfish_elo=None) -> FastAPI:
    from chess_app.server import create_app as create_chess_app

    if opponent not in {"minimax", "stockfish"}:
        raise ValueError("opponent must be 'minimax' or 'stockfish'.")
    if game is not None and opponent != "minimax":
        raise ValueError("Pass either a game or an opponent mode.")
    if opponent == "stockfish":
        if __package__:
            from assignment.stockfish_game import StockfishGame, OpponentError
        else:
            from stockfish_game import StockfishGame, OpponentError
        game = StockfishGame(stockfish_path, stockfish_time, stockfish_elo)

    app = create_chess_app(game)
    if opponent == "stockfish":
        @asynccontextmanager
        async def lifespan(app):
            game.start()
            try:
                yield
            finally:
                game.close()

        app.router.lifespan_context = lifespan

        @app.exception_handler(OpponentError)
        async def opponent_error(request, exc):
            return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.post("/api/simulate")
    def simulate(request: SimulationRequest) -> dict:
        try:
            return simulate_position(request.fen, request.move)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess server with stateless simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    add_opponent_arguments(parser)
    args = parser.parse_args()
    uvicorn.run(create_app(**opponent_kwargs(args)), host=args.host, port=args.port)
