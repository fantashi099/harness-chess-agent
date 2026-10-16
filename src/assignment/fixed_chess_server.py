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
    from assignment.agent.chess_policy import position_id
    from assignment.chess_server import create_app as create_base_app, simulate_position, add_opponent_arguments, opponent_kwargs
else:  # Files copied into /opt/assignment inside Modal.
    from chess_policy import position_id
    from chess_server import create_app as create_base_app, simulate_position, add_opponent_arguments, opponent_kwargs


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_position_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class HarnessRequest(PolicyRequest):
    action: dict


def create_app(game=None, *, policy="harness", max_nodes=6000, search_depth=4, **opponent_options):
    if policy not in {"search", "harness"} or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("Invalid policy or node budget")
    if policy in {"search", "harness"}:
        if type(search_depth) is not int or not 1<=search_depth<=12:
            raise ValueError("Search depth must be 1..12")
        if __package__:
            from assignment.agent import chess_policy_search as search_policy
        else:
            import chess_policy_search as search_policy
    if policy == "harness":
        if __package__:
            from assignment.agent import chess_policy_harness as harness
        else:
            import chess_policy_harness as harness
    analyses = {}
    working = {}
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
            working.clear()
            analyses.clear()
            decisions.clear()
            return dict(original_reset(), policy_game_id=game_id)

    # Existing GET /state and POST /reset routes call these instance methods.
    game.state, game.reset = state_with_game_id, reset_with_game_id
    decisions: dict[str, dict] = {}

    def commit_decision(decision, current, root_id, started):
        decision.setdefault("search_seconds", monotonic() - started)
        chosen = decision["chosen_move"]
        if chosen is None or chosen not in current["legal_moves"]:
            raise HTTPException(status_code=400, detail="Policy did not return a legal move.")
        if position_id(game.state()) != root_id:
            raise HTTPException(status_code=409, detail="Position changed during analysis.")
        state = game.play_human_move(chosen)
        state["policy_game_id"] = game_id
        decision.update(position_id=root_id, committed_move=chosen,
            recent_moves_available=True, post_fen=state["fen"], post_position_id=position_id(state),
            engine_move=state.get("engine_move"), transaction_seconds=monotonic() - started)
        decisions[root_id] = decision
        if policy == "harness":
            working["previous"] = dict(decision["llm_action"], pv=decision["pv"],
                actual_black_reply=decision["engine_move"], cutoff_reached=decision["cutoff_reached"])
        return {"decision": decision, "state": state, "replayed": False}

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
            if policy == "harness":
                raise HTTPException(status_code=400, detail="harness requires an explicit LLM harness commit")
            started = monotonic()
            decision = search_policy.select_move(game._board, game_id, max_nodes, search_depth,
                                                 evaluation="development-king-safety-v2")
            return commit_decision(decision, current, root_id, started)

    def harness_session(request):
        current = game.state()
        root_id = position_id(current)
        if policy != "harness":
            raise HTTPException(status_code=400, detail="Harness endpoints require harness")
        if request.expected_position_id in decisions:
            return current, request.expected_position_id, None
        if root_id != request.expected_position_id:
            raise HTTPException(status_code=409, detail="Stale harness snapshot")
        if current["game_over"]:
            return current, root_id, None
        if root_id not in analyses:
            analyses[root_id] = harness.prepare(game._board, root_id, max_nodes, search_depth,
                                                working.get("previous"), current.get("opponent"))
        return current, root_id, analyses[root_id]

    @app.post("/api/harness-context")
    def harness_context(request: PolicyRequest):
        with game._lock:
            current, root_id, session = harness_session(request)
            if session is None:
                return {"decision": decisions.get(root_id), "state": current, "replayed": True}
            return {"context": session["context"], "feedback": harness.feedback(session)}

    @app.post("/api/harness-analyze")
    def harness_analyze(request: HarnessRequest):
        with game._lock:
            _, _, session = harness_session(request)
            if session is None:
                raise HTTPException(status_code=409, detail="Position already committed or terminal")
            try:
                harness.analyze(game._board, game_id, session, request.action)
                return harness.feedback(session)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harness-commit")
    def harness_commit(request: HarnessRequest):
        with game._lock:
            current, root_id, session = harness_session(request)
            if session is None:
                return {"decision": decisions.get(root_id), "state": current, "replayed": True}
            try:
                decision = harness.decision_from(session, request.action, game._board, game_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return commit_decision(decision, current, root_id, monotonic())

    @app.get("/api/policy-config")
    def policy_config():
        if policy in {"search", "harness"}:
            return {"policy":policy,"max_nodes":max_nodes,"search_config":{"max_depth":search_depth},
                    "search_version":search_policy.SEARCH_VERSION,
                    "evaluation":harness.EVALUATION if policy == "harness" else "development-king-safety-v2",
                    "planner":"llm_harness" if policy == "harness" else "iterative_alpha_beta",
                    "max_llm_calls": harness.MAX_LLM_CALLS if policy == "harness" else 0,
                    "max_analyses": harness.MAX_ANALYSES if policy == "harness" else 0,
                    "verification_reserve": max(1, max_nodes//3) if policy == "harness" else 0,
                    "comparison_reserve": max(1, max_nodes//6) if policy == "harness" else 0,
                    "score_drop_threshold": harness.SCORE_DROP_THRESHOLD if policy == "harness" else None,
                    "tactical_probe_max_nodes": harness.TACTICAL_PROBE_NODES if policy == "harness" else 0,
                    "learning_during_evaluation":False}

    @app.get("/api/policy-decisions")
    def policy_decisions() -> list[dict]:
        with game._lock:
            return list(decisions.values())

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=["search", "harness"], default="harness")
    parser.add_argument("--max-nodes", type=int, default=6000)
    parser.add_argument("--search-depth",type=int,default=4)
    add_opponent_arguments(parser)
    args = parser.parse_args()
    uvicorn.run(create_app(policy=args.policy, max_nodes=args.max_nodes, search_depth=args.search_depth, **opponent_kwargs(args)), host=args.host, port=args.port)
