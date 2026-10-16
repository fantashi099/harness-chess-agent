"""Opt-in chess server with atomic, idempotent fixed-policy selection/commit.

The pinned chess app and default opponent are unchanged. Its mutation methods
share a reentrant lock with this endpoint, including browser move/reset calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


class VerificationRequest(PolicyRequest):
    plan: dict


class HarnessRequest(PolicyRequest):
    action: dict


def create_app(game=None, *, policy="v2", memory_path=None, max_nodes=6000, planner="rules", search_depth=4, **opponent_options):
    if policy not in {"v2", "v3", "v4", "v5", "v6"} or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("Invalid policy or node budget")
    memory_text, catalog = "", {}
    if policy in {"v3", "v4"}:
        if __package__:
            from assignment.agent.chess_policy_v3 import select_move as select_v3, identity
        else:
            from chess_policy_v3 import select_move as select_v3, identity
        if memory_path is None:
            raise ValueError("v3 requires an explicit frozen memory path")
        memory_text = Path(memory_path).read_text()
        catalog = json.loads(memory_text)
        if catalog.get("learning_during_evaluation") is not False:
            raise ValueError("v3 requires frozen memory")
    if planner not in {"rules", "llm"}:
        raise ValueError("Invalid planner")
    if policy == "v4":
        if __package__:
            from assignment.agent import chess_policy_planned as planned
        else:
            import chess_policy_planned as planned
    if policy in {"v5", "v6"}:
        if type(search_depth) is not int or not 1<=search_depth<=12:
            raise ValueError("Search depth must be 1..12")
        if __package__:
            from assignment.agent import chess_policy_search as search_policy
        else:
            import chess_policy_search as search_policy
    if policy == "v6":
        if __package__:
            from assignment.agent import chess_policy_harness as harness
        else:
            import chess_policy_harness as harness
    analyses = {}
    memory_hash = hashlib.sha256(memory_text.encode()).hexdigest() if memory_text else None
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
            if policy in {"v3", "v4", "v5", "v6"}:
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
        if policy == "v4":
            working["previous"] = dict(decision["plan"] or {}, pv=decision["pv"],
                cutoff_reached=decision["cutoff_reached"], primary_score=decision["primary_score"])
        elif policy == "v3":
            working["previous"] = dict(decision["plan"], pv=decision["pv"], cutoff_reached=decision["cutoff_reached"])
        elif policy == "v6":
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
            if policy == "v6":
                raise HTTPException(status_code=400, detail="v6 requires an explicit LLM harness commit")
            recent = [row["uci"] for row in current["history"]
                      if row.get("color") == "white"][-4:]
            started = monotonic()
            if policy == "v5":
                decision=search_policy.select_move(game._board,game_id,max_nodes,search_depth)
            elif policy == "v4":
                session = analyses.get(root_id)
                if session is None:
                    raise HTTPException(status_code=400, detail="Prepare a plan before committing")
                decision = planned.decision_from(session, planner)
                decision.update(memory_sha256=memory_hash, executor_position_id=identity(game._board, game_id))
            elif policy == "v3":
                snapshot = game._board.copy(stack=True)
                decision = select_v3(snapshot, game_id, catalog, max_nodes, previous=working.get("previous"))
                if decision["plan"]["position_id"] != identity(game._board, game_id):
                    raise HTTPException(status_code=409, detail="Stale executor plan")
                decision.update(memory_sha256=memory_hash, executor_position_id=decision["plan"]["position_id"])
            else:
                decision = select_move(current["fen"], simulate_position, recent)
            return commit_decision(decision, current, root_id, started)

    def harness_session(request):
        current = game.state()
        root_id = position_id(current)
        if policy != "v6":
            raise HTTPException(status_code=400, detail="Harness endpoints require v6")
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

    @app.post("/api/planning-context")
    def planning_context(request: PolicyRequest):
        with game._lock:
            current = game.state()
            if policy != "v4":
                raise HTTPException(status_code=400, detail="Planning endpoints require v4")
            root_id = position_id(current)
            if request.expected_position_id in decisions:
                return {"decision": decisions[request.expected_position_id], "state": current, "replayed": True}
            if root_id != request.expected_position_id:
                raise HTTPException(status_code=409, detail="Stale planning snapshot")
            if current["game_over"]:
                return {"decision": None, "state": current, "replayed": True}
            if root_id not in analyses:
                analyses[root_id] = planned.prepare(game._board, game_id, catalog, root_id, max_nodes, working.get("previous"))
            session = analyses[root_id]
            return {"context": session["context"], "attempt_count": len(session["attempts"]),
                    "feedback": planned.feedback_summary(session["attempts"][-1]) if session["attempts"] else None}

    @app.post("/api/verify-plan")
    def verify_plan(request: VerificationRequest):
        with game._lock:
            root_id = position_id(game.state())
            if policy != "v4" or root_id != request.expected_position_id:
                raise HTTPException(status_code=409, detail="Stale or unsupported plan")
            if root_id not in analyses:
                raise HTTPException(status_code=400, detail="Prepare planning context first")
            try:
                return planned.execute_plan(game._board, game_id, analyses[root_id], request.plan)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/policy-config")
    def policy_config():
        if policy in {"v5", "v6"}:
            return {"policy":policy,"max_nodes":max_nodes,"search_config":{"max_depth":search_depth},
                    "search_version":search_policy.SEARCH_VERSION,
                    "evaluation":harness.EVALUATION if policy == "v6" else "unchanged-v2-static",
                    "planner":"llm_harness" if policy == "v6" else "iterative_alpha_beta",
                    "max_llm_calls": harness.MAX_LLM_CALLS if policy == "v6" else 0,
                    "max_analyses": harness.MAX_ANALYSES if policy == "v6" else 0,
                    "verification_reserve": max(1, max_nodes//3) if policy == "v6" else 0,
                    "comparison_reserve": max(1, max_nodes//6) if policy == "v6" else 0,
                    "score_drop_threshold": harness.SCORE_DROP_THRESHOLD if policy == "v6" else None,
                    "tactical_probe_max_nodes": harness.TACTICAL_PROBE_NODES if policy == "v6" else 0,
                    "learning_during_evaluation":False}
        return {"policy": policy, "max_nodes": max_nodes, "memory_sha256": memory_hash,
                "memory_text": memory_text, "planner": planner if policy == "v4" else "bounded_rule_planner", "variant": "R"}

    @app.get("/api/policy-decisions")
    def policy_decisions() -> list[dict]:
        with game._lock:
            return list(decisions.values())

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=["v2", "v3", "v4", "v5", "v6"], default="v2")
    parser.add_argument("--planner", choices=["rules", "llm"], default="rules")
    parser.add_argument("--memory-path", type=Path)
    parser.add_argument("--max-nodes", type=int, default=6000)
    parser.add_argument("--search-depth",type=int,default=4)
    add_opponent_arguments(parser)
    args = parser.parse_args()
    uvicorn.run(create_app(policy=args.policy, memory_path=args.memory_path, max_nodes=args.max_nodes, planner=args.planner, search_depth=args.search_depth, **opponent_kwargs(args)), host=args.host, port=args.port)
