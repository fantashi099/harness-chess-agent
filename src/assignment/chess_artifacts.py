"""Artifacts for search and harness runs, including incomplete runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import chess
import chess.pgn



def _revision(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def search_sources(repo):
    from assignment.chess_sandbox import SANDBOX_FILES
    return (*SANDBOX_FILES,*(repo/p for p in ("src/assignment/agent/chess_policy_search.py",
        "src/assignment/agent/chess_agent.py","src/assignment/chess_sandbox.py","src/assignment/chess_artifacts.py",
        "src/assignment/cli.py","scripts/replay_chess_game.py","experiments/chess/evaluate_search_policy.py")))


def harness_sources(repo):
    return (*search_sources(repo), *(repo/p for p in ("src/assignment/agent/chess_policy_harness.py",
        "src/assignment/agent/chess_harness.py", "src/assignment/agent/base.py", "src/assignment/agent/chess_tools.py",
        "scripts/record_chess_game.py")))


def harness_metrics(decisions, records):
    return {
        "verification_analyses": sum(d["verification_count"] for d in decisions),
        "comparison_turns": sum(d["comparison_count"] > 0 for d in decisions),
        "replanned_turns": sum(d["replanned"] for d in decisions),
        "repair_turns": len({r["position_id"] for r in records if r["status"] == "rejected"}),
    }


def record_saved_game(result_path: Path) -> None:
    """Render saved history without letting a video failure hide the game result."""
    result_path = result_path.resolve()
    output = result_path.parent / "game_output.mp4"
    repo = Path(__file__).resolve().parents[2]
    try:
        with tempfile.TemporaryDirectory(dir=result_path.parent, prefix=".game-record-") as directory:
            temporary_output = Path(directory) / output.name
            subprocess.run(
                ["uv", "run", "python", str(repo / "scripts/record_chess_game.py"),
                 "--result", str(result_path), "--out", str(temporary_output)],
                cwd=repo, check=True, capture_output=True, text=True,
            )
            temporary_output.replace(output)
        print(f"Recording: {output}", flush=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"Warning: could not record {result_path}; saved result retained.\n{detail}",
              file=sys.stderr, flush=True)


def save_chess_run(agent, result_path: Path, patch_path: Path, stop_reason: str) -> None:
    """Called while the sandbox is alive; recover decisions after a lost response."""
    if agent.chess_policy not in {"search", "harness"}:
        raise ValueError("Artifacts require search or harness mode")
    artifact_errors = []
    try:
        response = agent.chess_client.get("/api/state")
        response.raise_for_status()
        agent.last_state = response.json()
    except Exception as exc:
        artifact_errors.append(f"Final state refresh failed: {exc}")
    try:
        response = agent.chess_client.get("/api/policy-decisions")
        response.raise_for_status()
        decisions = response.json()
        if not isinstance(decisions, list):
            raise ValueError("Expected a decision list.")
        agent.chess_decisions = decisions
    except Exception as exc:
        artifact_errors.append(f"Decision recovery failed: {exc}")
    state = agent.last_state
    search_config=None
    is_harness = agent.chess_policy == "harness"
    if agent.chess_policy in {"search", "harness"}:
        if is_harness:
            from assignment.agent.chess_policy_harness import POLICY_VERSION as version
        else:
            from assignment.agent.chess_policy_search import POLICY_VERSION as version
        try:
            response=agent.chess_client.get("/api/policy-config")
            response.raise_for_status()
            search_config=response.json()
        except Exception as exc:
            artifact_errors.append(f"Search configuration recovery failed: {exc}")
    if is_harness:
        for decision in agent.chess_decisions:
            decision["harness_records"] = [r for r in agent.planner_records if r["position_id"] == decision["position_id"]]
    opponent = state.get("opponent", {"mode": "minimax", "depth": 2})
    # Incomplete is explicit even when a stale last_state looked terminal.
    completed = bool(state.get("game_over")) and not artifact_errors
    result = dict(state, stop_reason=stop_reason, completed=completed,
                  policy_version=version, artifact_errors=artifact_errors)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (result_path.parent / "decisions.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in agent.chess_decisions)
    )
    if is_harness:
        (result_path.parent / "harness-trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in agent.planner_records))
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = version
    game.headers["Black"] = (opponent.get("name") or "Stockfish") if opponent["mode"] == "stockfish" else "Provided minimax depth 2"
    node = game
    try:
        for row in state["history"]:
            move = chess.Move.from_uci(row["uci"])
            if board.outcome(claim_draw=True) is not None or move not in board.legal_moves:
                raise ValueError(f"Invalid history at ply {row['ply']}.")
            node = node.add_variation(move)
            board.push(move)
        if board.fen() != state["fen"]:
            raise ValueError("History and final FEN differ.")
        outcome = board.outcome(claim_draw=True)
        if bool(outcome) != bool(state["game_over"]):
            raise ValueError("History and game_over differ.")
        if outcome:
            result["replay_result"] = outcome.result()
            result["termination"] = outcome.termination.name.lower()
            game.headers["Result"] = outcome.result()
        else:
            game.headers["Result"] = "*"
        (result_path.parent / "game.pgn").write_text(str(game) + "\n")
    except Exception as exc:
        result["completed"] = False
        artifact_errors.append(f"History replay failed: {exc}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[2]
    policy_path = Path(__file__).parent / "agent" / "chess_policy.py"
    prompt_path = Path(__file__).with_name("prompts.py")
    manifest = {
        "policy": agent.chess_policy, "policy_version": version, "requested_model": agent.model,
        "reported_models": sorted({r.get("model", "") for r in agent.api_responses}),
        "repo_revision": _revision(repo),
        "chess_app_revision": _revision(repo / "chess_app"),
        "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "skills_sha256": {name: hashlib.sha256(s["content"].encode()).hexdigest()
                          for name, s in agent.skills.items()},
        "opponent": opponent, "opponent_depth": opponent.get("depth"),
        "initial_fen": chess.STARTING_FEN,
        "search_depth": 2, "step_limit": agent.step_limit,
        "compact_threshold_tokens": agent.compact_threshold_tokens,
        "teacher_used": False, "stop_reason": stop_reason,
        "decision_count": len(agent.chess_decisions),
        "simulator_backend": "in_process_simulate_position",
        "note": "Repository files may have uncommitted changes; source hashes describe this run.",
    }
    if agent.chess_policy in {"search", "harness"}:
        sources=harness_sources(repo) if is_harness else search_sources(repo)
        manifest.update(source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            simulator_backend="immutable_board_stack",planner="iterative_alpha_beta",memory_used=False,
            learning_during_evaluation=False,action_driver=getattr(agent,"action_driver","react_llm"))
        if search_config:
            manifest.update(max_nodes=search_config["max_nodes"],
                search_config=dict(search_config["search_config"], evaluation=search_config["evaluation"]),
                search_depth=search_config["search_config"]["max_depth"],search_version=search_config["search_version"],
                evaluation=search_config["evaluation"])
        if is_harness:
            from assignment.agent.chess_harness import HARNESS_PROMPT
            records = agent.planner_records
            manifest.update(planner="llm_harness", decision_authority="llm", llm_model=agent.model,
                max_llm_calls_per_turn=search_config["max_llm_calls"] if search_config else None,
                max_analyses_per_turn=search_config["max_analyses"] if search_config else None,
                verification_reserve_nodes=search_config["verification_reserve"] if search_config else None,
                comparison_reserve_nodes=search_config["comparison_reserve"] if search_config else None,
                tactical_probe_max_nodes=search_config["tactical_probe_max_nodes"] if search_config else None,
                score_drop_threshold=search_config["score_drop_threshold"] if search_config else None,
                harness_prompt_sha256=hashlib.sha256(HARNESS_PROMPT.encode()).hexdigest(),
                llm_response_count=sum(r["response"] is not None for r in records),
                llm_reported_models=sorted({r["response"]["model"] for r in records if r["response"] is not None}),
                llm_seconds=sum(r["seconds"] for r in records),
                llm_token_usage={key: sum(((r.get("response") or {}).get("usage") or {}).get(key, 0) for r in records)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                changed_first_suggestion=sum(d["llm_changed_first_suggestion"] for d in agent.chess_decisions),
                **harness_metrics(agent.chess_decisions, records))
    (result_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    record_saved_game(result_path)
