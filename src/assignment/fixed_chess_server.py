"""Opt-in chess server with atomic, idempotent fixed-policy selection/commit.

The pinned chess app and default opponent are unchanged. Its mutation methods
share a reentrant lock with this endpoint, including browser move/reset calls.
"""

from __future__ import annotations

import argparse
from threading import RLock
from time import monotonic
from uuid import uuid4

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from assignment.agent.chess_policy import position_id, select_move
    from assignment.chess_server import create_app as create_base_app, simulate_position, add_opponent_arguments, opponent_kwargs
else:  # Files copied into /opt/assignment inside Modal.
    from chess_policy import position_id, select_move
    from chess_server import create_app as create_base_app, simulate_position, add_opponent_arguments, opponent_kwargs


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_position_id: str = Field(pattern=r"^[0-9a-f]{64}$")


def create_app(game=None, **opponent_options):
    app = create_base_app(game, **opponent_options)
    game = app.state.chess_game
    # RLock allows state()/play_human_move() to use the same lock while this
    # endpoint owns the entire selection/commit transaction.
    game._lock = RLock()
    game_id = uuid4().hex
    original_state, original_reset = game.state, game.reset

    def state_with_game_id():
        with game._lock:
            return dict(original_state(), policy_game_id=game_id)

    def reset_with_game_id():
        nonlocal game_id
        with game._lock:
            game_id = uuid4().hex
            return dict(original_reset(), policy_game_id=game_id)

    # Existing GET /state and POST /reset routes call these instance methods.
    game.state, game.reset = state_with_game_id, reset_with_game_id
    decisions: dict[str, dict] = {}

    @app.post("/api/select-and-play")
    def select_and_play(request: PolicyRequest) -> dict:
        with game._lock:
            current = game.state()
            root_id = position_id(current)
            if request.expected_position_id in decisions:
                # A retry after a lost HTTP response must not play another move.
                return {"decision": decisions[request.expected_position_id],
                        "state": current, "replayed": True}
            if root_id != request.expected_position_id:
                raise HTTPException(status_code=409, detail="Stale position; refresh state.")
            if current["game_over"]:
                return {"decision": None, "state": current, "replayed": False}
            recent = [row["uci"] for row in current["history"]
                      if row.get("color") == "white"][-4:]
            started = monotonic()
            decision = select_move(current["fen"], simulate_position, recent)
            search_seconds = monotonic() - started
            chosen = decision["chosen_move"]
            if chosen is None or chosen not in current["legal_moves"]:
                raise HTTPException(status_code=400, detail="Policy did not return a legal move.")
            # Guard against any accidental mutation by the search implementation.
            if position_id(game.state()) != root_id:
                raise HTTPException(status_code=409, detail="Position changed during search.")
            state = game.play_human_move(chosen)
            state["policy_game_id"] = game_id
            decision.update(
                position_id=root_id, committed_move=chosen,
                recent_moves_available=True, post_fen=state["fen"],
                post_position_id=position_id(state), engine_move=state.get("engine_move"),
                search_seconds=search_seconds, transaction_seconds=monotonic() - started,
            )
            decisions[root_id] = decision
            return {"decision": decision, "state": state, "replayed": False}

    @app.get("/api/policy-decisions")
    def policy_decisions() -> list[dict]:
        with game._lock:
            return list(decisions.values())

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    add_opponent_arguments(parser)
    args = parser.parse_args()
    uvicorn.run(create_app(**opponent_kwargs(args)), host=args.host, port=args.port)
