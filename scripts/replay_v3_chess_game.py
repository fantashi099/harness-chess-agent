"""Validate v3/v4/v5/v6 sources, exact White decisions and game consistency.

Timed Stockfish is checked for legal moves/configuration, never exact replies.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import chess
import chess.pgn

from assignment.agent.chess_policy import position_id
from assignment.agent.chess_policy_v3 import POLICY_VERSION, identity, select_move

REPO = Path(__file__).resolve().parents[1]


def verify(result_path, patch):
    state = json.loads(result_path.read_text())
    manifest = json.loads((result_path.parent / "manifest.json").read_text())
    decisions = [json.loads(line) for line in (result_path.parent / "decisions.jsonl").read_text().splitlines() if line]
    from assignment.agent import chess_policy_planned as planned
    from assignment.agent import chess_policy_search as search_policy
    from assignment.agent import chess_policy_harness as harness
    is_planned = state["policy_version"] == planned.POLICY_VERSION
    is_search=state["policy_version"]==search_policy.POLICY_VERSION
    is_harness=state["policy_version"] in {harness.POLICY_VERSION, "select-move-v6-llm-harness-1",
                                         "select-move-v6-llm-harness-2", "select-move-v6-llm-harness-3",
                                         "select-move-v6-llm-harness-4", "select-move-v6-llm-harness-5",
                                         "select-move-v6-llm-harness-6", "select-move-v6-llm-harness-7",
                                         "select-move-v6-llm-harness-8", "select-move-v6-llm-harness-9", "select-move-v6-llm-harness-10", "select-move-v6-llm-harness-11", "select-move-v6-llm-harness-12"}
    if not (is_harness or state["policy_version"] in {POLICY_VERSION, planned.POLICY_VERSION,search_policy.POLICY_VERSION}) or manifest["policy_version"] != state["policy_version"]:
        raise ValueError("Unsupported chess policy version")
    for relative, digest in manifest["source_sha256"].items():
        source_path = REPO / relative
        if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
            archive = result_path.parent.parent / "source-snapshot" / relative
            if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed without matching snapshot: {relative}")
            if (is_search or is_harness) and relative=="src/assignment/agent/chess_policy_search.py":
                spec=importlib.util.spec_from_file_location("assignment.agent._recorded_search",archive)
                search_policy=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(search_policy)
            if is_harness and relative=="src/assignment/agent/chess_policy_harness.py":
                spec=importlib.util.spec_from_file_location("assignment.agent._recorded_harness",archive)
                harness=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(harness)
            if relative in {"src/assignment/agent/chess_policy.py", "src/assignment/agent/chess_policy_v3.py",
                            "src/assignment/agent/chess_policy_planned.py", "experiments/chess/verify_exchange.py"}:
                raise ValueError(f"Executor changed; recompute with archived version: {relative}")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Patch changed since run")
    if is_harness:
        if harness.POLICY_VERSION != state["policy_version"]:
            raise ValueError("Missing matching archived harness version")
        if getattr(harness, "REVISION", 0) >= 3:
            limits = {"max_llm_calls_per_turn": harness.MAX_LLM_CALLS,
                      "max_analyses_per_turn": harness.MAX_ANALYSES,
                      "verification_reserve_nodes": max(1, manifest["max_nodes"]//3),
                      "comparison_reserve_nodes": max(1, manifest["max_nodes"]//6),
                      "score_drop_threshold": harness.SCORE_DROP_THRESHOLD}
            if getattr(harness, "REVISION", 0) >= 6:
                limits["tactical_probe_max_nodes"] = harness.TACTICAL_PROBE_NODES
            if any(manifest[key] != value for key, value in limits.items()):
                raise ValueError("Harness limits/threshold differ")
        harness.search = search_policy
        harness_records = [json.loads(line) for line in (result_path.parent/"harness-trace.jsonl").read_text().splitlines() if line]
        if manifest["decision_authority"] != "llm" or manifest["planner"] != "llm_harness":
            raise ValueError("Unexpected v6 decision authority")
        responses = [r["response"] for r in harness_records if r["response"] is not None]
        usage = {key:sum((r.get("usage") or {}).get(key,0) for r in responses)
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if manifest["llm_response_count"] != len(responses) or manifest["llm_token_usage"] != usage:
            raise ValueError("LLM response/token totals differ")
    if is_search or is_harness:
        if manifest["teacher_used"] or manifest["memory_used"] or manifest["learning_during_evaluation"]:
            raise ValueError("Unexpected teacher/memory mutation")
        catalog={}
    else:
        memory_bytes = (result_path.parent / "memory-snapshot.json").read_bytes()
        if hashlib.sha256(memory_bytes).hexdigest() != manifest["memory_sha256"]:
            raise ValueError("Memory snapshot/hash differ")
        catalog = json.loads(memory_bytes)
        if catalog["learning_during_evaluation"] is not False or manifest["learning_during_evaluation"] is not False:
            raise ValueError("Memory was not frozen")
    opponent = manifest["opponent"]
    if opponent != state.get("opponent", {"mode": "minimax", "depth": 2}) or opponent["mode"] not in {"stockfish", "minimax"}:
        raise ValueError("Opponent configuration differs")
    if opponent["mode"] == "stockfish":
        if opponent["threads"] != 1 or opponent["hash_mib"] != 64 or not 0 < opponent["move_time"] <= 10 or not opponent["name"]:
            raise ValueError("Invalid recorded Stockfish configuration")
    with tempfile.TemporaryDirectory(prefix="v3-replay-") as directory:
        clone = Path(directory) / "app"
        subprocess.run(["git", "clone", "--quiet", "--shared", str(REPO / "chess_app"), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "apply", str(patch.resolve())], check=True, capture_output=True)
        revision = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
        if revision != manifest["chess_app_revision"]:
            raise ValueError("Chess app revision differs")
        source = str(clone / "src")
        sys.path.insert(0, source)
        try:
            from chess_app.game import ChessGame
            bot = ChessGame(engine_depth=2)
            board = chess.Board(manifest["initial_fen"])
            history, index, previous = [], 0, None
            for ply, row in enumerate(state["history"], 1):
                move = chess.Move.from_uci(row["uci"])
                if board.outcome(claim_draw=True) is not None or move not in board.legal_moves or row["ply"] != ply:
                    raise ValueError(f"Illegal or post-terminal move at ply {ply}")
                if row["color"] != ("white" if board.turn else "black") or row["san"] != board.san(move):
                    raise ValueError(f"SAN/color mismatch at ply {ply}")
                if board.turn:
                    decision = decisions[index]
                    index += 1
                    root = {"fen": board.fen(), "history": history, "policy_game_id": state["policy_game_id"]}
                    executor_id=(search_policy.identity(board,state["policy_game_id"]) if is_search or is_harness else identity(board,state["policy_game_id"]))
                    if decision["position_id"] != position_id(root) or decision["executor_position_id"] != executor_id:
                        raise ValueError(f"Stale game/history/plan at ply {ply}")
                    if is_harness:
                        records = [r for r in harness_records if r["position_id"] == decision["position_id"]]
                        if records != decision["harness_records"] or len(records)>harness.MAX_LLM_CALLS:
                            raise ValueError("Missing LLM evidence or call allowance exceeded")
                        accepted = []
                        for record in records:
                            request = record["request"]
                            data = json.loads(request[-1]["content"])
                            if data["context"]["position_id"] != decision["position_id"] or data["context"]["fen"] != board.fen():
                                raise ValueError("LLM request used stale state")
                            if getattr(harness, "REVISION", 0) >= 12 and data["context"]["board_facts"] != harness.board_facts(board):
                                raise ValueError("LLM request board_facts differ")
                            if hasattr(harness, "EVALUATION") and data["context"]["opponent"] != (opponent if opponent["mode"] == "stockfish" else None):
                                raise ValueError("LLM request used a different opponent")
                            if hashlib.sha256(request[0]["content"].encode()).hexdigest() != manifest["harness_prompt_sha256"]:
                                raise ValueError("Harness prompt differs")
                            if record["status"] == "accepted":
                                response = record["response"]
                                choice = response["choices"][0]
                                if choice["finish_reason"] != "stop" or not (response.get("usage") or {}).get("total_tokens"):
                                    raise ValueError("Accepted action lacks complete provider response/usage")
                                action = json.loads(choice["message"]["content"])
                                if action != record["action"]:
                                    raise ValueError("LLM response and executed action differ")
                                accepted.append(action)
                        prepare_kwargs = {"opponent": opponent} if hasattr(harness, "EVALUATION") else {}
                        session = harness.prepare(board, position_id(root), manifest["max_nodes"], manifest["search_depth"], previous, **prepare_kwargs)
                        for recorded in decision["analysis_trace"]:
                            if recorded["action"] not in accepted:
                                raise ValueError("Analysis was not requested by the LLM")
                            recomputed = harness.analyze(board, state["policy_game_id"], session, recorded["action"])
                            for key in ("action_id", "action", "evidence", "remaining_nodes", *(("safety",) if hasattr(harness, "EVALUATION") else ())):
                                if recorded[key] != recomputed[key]:
                                    raise ValueError(f"Harness analysis differs for {key} at ply {ply}")
                            if getattr(harness, "REVISION", 0) >= 12 and recorded["move_facts"] != recomputed["move_facts"]:
                                raise ValueError("Harness move_facts differ")
                            if getattr(harness, "REVISION", 0) >= 3:
                                for key in ("verification", "effective_max_depth"):
                                    if recorded[key] != recomputed[key]:
                                        raise ValueError(f"Verification configuration differs at ply {ply}")
                        if decision["llm_action"] not in accepted:
                            raise ValueError("Commit was not explicitly chosen by the LLM")
                        expected = harness.decision_from(session, decision["llm_action"], board, state["policy_game_id"])
                        for key in ("llm_action", "score_bound", "first_search_suggestion", "llm_changed_first_suggestion"):
                            if decision[key] != expected[key]:
                                raise ValueError(f"LLM selection metadata differs at ply {ply}")
                        if hasattr(harness, "EVALUATION"):
                            if decision["safety"] != expected["safety"] or decision["verification_nodes"] != expected["verification_nodes"]:
                                raise ValueError("Commit safety/verification differs")
                            if decision["nodes_used"] > manifest["max_nodes"] or manifest["evaluation"] != harness.EVALUATION:
                                raise ValueError("Harness budget/evaluator differs")
                        if getattr(harness, "REVISION", 0) >= 3:
                            for key in ("score_source_action_id", "verification_action_id", "replan_events", "continuation_checks",
                                        "verification_count", "comparison_count", "replanned"):
                                if decision[key] != expected[key]:
                                    raise ValueError(f"Harness evidence/plan differs for {key} at ply {ply}")
                        if getattr(harness, "REVISION", 0) >= 10:
                            if decision["risk_acceptance"] != expected["risk_acceptance"]:
                                raise ValueError("Risk acceptance metadata differs")
                        if getattr(harness, "REVISION", 0) >= 6:
                            for key in ("candidate_comparisons", "proven_refutations", "unavoidable_mate"):
                                if decision[key] != expected[key]:
                                    raise ValueError(f"Tactical/comparison evidence differs for {key} at ply {ply}")
                    elif is_search:
                        expected=search_policy.select_move(board,state["policy_game_id"],manifest["max_nodes"],**manifest["search_config"])
                        for key in ("search_version","evaluation","iterations","alpha_beta_prunes","top_candidates"):
                            if decision[key]!=expected[key]:raise ValueError(f"Search trace mismatch for {key} at ply {ply}")
                        if "quiescence" in expected and decision["quiescence"]!=expected["quiescence"]:
                            raise ValueError(f"Quiescence trace mismatch at ply {ply}")
                        if decision["nodes_used"]>manifest["max_nodes"]:raise ValueError("Node cap exceeded")
                    elif is_planned:
                        session = planned.prepare(board, state["policy_game_id"], catalog, position_id(root), manifest["max_nodes"], previous)
                        for recorded in decision["execution_attempts"]:
                            recomputed = planned.execute_plan(board, state["policy_game_id"], session, recorded["plan"])
                            if json.dumps(recorded, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
                                raise ValueError(f"Task/evidence recompute differs at ply {ply}")
                        expected = planned.decision_from(session, manifest["planner"])
                        if manifest["planner"] == "llm":
                            accepted = [r["plan"] for r in decision["planner_records"] if r["status"] == "accepted"]
                            if any(r["plan"] not in accepted for r in decision["execution_attempts"]):
                                raise ValueError("Executed plan has no accepted LLM record")
                    else:
                        expected = select_move(board, state["policy_game_id"], catalog, manifest["max_nodes"], previous=previous)
                    for key in ("policy_version", "root_fen", "chosen_move", "primary_score", "pv", "completed_depth", "nodes_used", "cutoff_reached", "reason"):
                        if decision[key] != expected[key]:
                            raise ValueError(f"v3 recompute mismatch for {key} at ply {ply}")
                    for key in (() if is_planned or is_search or is_harness else ("profile", "workflow_id", "position_id", "keep/revise", "budget")):
                        if decision["plan"][key] != expected["plan"][key]:
                            raise ValueError(f"Planner mismatch for {key} at ply {ply}")
                    if decision["chosen_move"] != row["uci"] or decision["committed_move"] != row["uci"]:
                        raise ValueError(f"Decision/commit mismatch at ply {ply}")
                    if not is_search and not is_harness and decision["memory_sha256"] != manifest["memory_sha256"]:
                        raise ValueError("Decision memory differs")
                    previous = dict(decision.get("plan") or {}, pv=decision["pv"], cutoff_reached=decision["cutoff_reached"])
                    if is_planned:
                        previous["primary_score"] = decision["primary_score"]
                    if is_harness:
                        previous = dict(decision["llm_action"], pv=decision["pv"],
                            actual_black_reply=decision["engine_move"], cutoff_reached=decision["cutoff_reached"])
                elif opponent["mode"] == "minimax":
                    bot._board = board.copy(stack=True)
                    if bot._choose_engine_move() != move:
                        raise ValueError(f"Minimax reply differs at ply {ply}")
                board.push(move)
                history.append(row)
                if board.turn or board.outcome(claim_draw=True) is not None:
                    decision = decisions[index-1]
                    post = {"fen": board.fen(), "history": history, "policy_game_id": state["policy_game_id"]}
                    if decision["post_fen"] != board.fen() or decision["post_position_id"] != position_id(post):
                        raise ValueError(f"Post-state differs at ply {ply}")
                    if decision["engine_move"] != (row["uci"] if row["color"] == "black" else None):
                        raise ValueError("Recorded Black reply differs")
            outcome = board.outcome(claim_draw=True)
            if board.fen() != state["fen"] or bool(outcome) != state["game_over"]:
                raise ValueError("Final state/outcome differs")
            if not board.turn and outcome is None:
                raise ValueError("Incomplete White/Black transaction")
            if index != len(decisions) or index != manifest["decision_count"]:
                raise ValueError("Missing/extra decisions")
            if outcome:
                status = "White wins" if outcome.winner is True else "Black wins" if outcome.winner is False else "Draw"
                if state["status"] != status or state["replay_result"] != outcome.result():
                    raise ValueError("Winner/result differs")
            if state["completed"] != bool(outcome and not state["artifact_errors"]):
                raise ValueError("Completed flag differs")
            with (result_path.parent / "game.pgn").open() as handle:
                pgn = chess.pgn.read_game(handle)
            if pgn is None or pgn.errors or [m.uci() for m in pgn.mainline_moves()] != [r["uci"] for r in history]:
                raise ValueError("PGN and history differ")
            if is_harness and getattr(harness, "REVISION", 0) >= 3:
                from assignment.chess_artifacts import harness_metrics
                if any(manifest[key] != value for key, value in harness_metrics(decisions, harness_records).items()):
                    raise ValueError("Harness metrics differ")
            return {"validated": True, "policy_version": state["policy_version"], "plies": len(history),
                    "decision_count": index, "result": outcome.result() if outcome else "*", "completed": state["completed"],
                    "all_white_moves_follow_policy": True, "memory_hash_verified": None if is_search or is_harness else True,
                    "all_commits_match_llm_response": True if is_harness else None,
                    "opponent_validation": "exact_replay" if opponent["mode"] == "minimax" else "legal_moves_and_recorded_config_only"}
        finally:
            sys.path.remove(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--patch", type=Path, default=REPO / "artifacts/fix.patch")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.result, args.patch)
    except Exception as exc:
        report = {"validated": False, "error": str(exc)}
    output = args.report or args.result.parent / "replay-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
