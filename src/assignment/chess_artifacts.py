"""Artifacts for opt-in fixed chess runs, including incomplete runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import chess
import chess.pgn

from assignment.agent.chess_policy import POLICY_VERSION


def _revision(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def save_fixed_run(agent, result_path: Path, patch_path: Path, stop_reason: str) -> None:
    """Called while the sandbox is alive; recover decisions after a lost response."""
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
    opponent = state.get("opponent", {"mode": "minimax", "depth": 2})
    # Incomplete is explicit even when a stale last_state looked terminal.
    completed = bool(state.get("game_over")) and not artifact_errors
    result = dict(state, stop_reason=stop_reason, completed=completed,
                  policy_version=POLICY_VERSION, artifact_errors=artifact_errors)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (result_path.parent / "decisions.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in agent.chess_decisions)
    )
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = POLICY_VERSION
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
        "policy_version": POLICY_VERSION, "requested_model": agent.model,
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
    (result_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
