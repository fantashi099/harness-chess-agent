"""Verify policy/history; reproduce minimax replies, check Stockfish reply legality."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import chess

from assignment.agent.chess_policy import POLICY_VERSION, position_id, select_move
from assignment.chess_server import simulate_position

REPO = Path(__file__).resolve().parents[1]


def verify(state, decisions, patch):
    if state.get("policy_version") != POLICY_VERSION:
        raise ValueError("Result was produced by a different policy version.")
    manifest_path = state.pop("_manifest_path")
    manifest = json.loads(manifest_path.read_text())
    opponent_mode = manifest.get("opponent", {"mode": "minimax"})["mode"]
    if opponent_mode not in {"minimax", "stockfish"}:
        raise ValueError(f"Unknown opponent mode: {opponent_mode}")
    if state.get("opponent", {"mode": "minimax"})["mode"] != opponent_mode:
        raise ValueError("Result and manifest opponents differ.")
    helper = REPO / "src/assignment/agent/chess_policy.py"
    if manifest["helper_sha256"] != hashlib.sha256(helper.read_bytes()).hexdigest():
        raise ValueError("Helper changed since this run; replay that version instead.")
    if manifest["patch_sha256"] != hashlib.sha256(patch.read_bytes()).hexdigest():
        raise ValueError("Patch differs from the one used in this run.")
    with tempfile.TemporaryDirectory(prefix="chess-replay-") as directory:
        clone = Path(directory) / "app"
        subprocess.run(["git", "clone", "--quiet", "--shared",
                        str(REPO / "chess_app"), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "apply", str(patch.resolve())],
                       check=True, capture_output=True)
        revision = subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
        if revision != manifest["chess_app_revision"]:
            raise ValueError("Chess app revision differs from run manifest.")
        sys.path.insert(0, str(clone / "src"))
        from chess_app.game import ChessGame

        bot = ChessGame(engine_depth=2)
        board = chess.Board()
        decision_index = 0
        previous_history = []
        previous_white = []
        for index, row in enumerate(state["history"], start=1):
            if board.outcome(claim_draw=True) is not None:
                raise ValueError(f"History continued after terminal at ply {index}.")
            move = chess.Move.from_uci(row["uci"])
            if move not in board.legal_moves or row["ply"] != index:
                raise ValueError(f"Illegal/inconsistent move at ply {index}.")
            expected_color = "white" if board.turn else "black"
            if row["color"] != expected_color or row["san"] != board.san(move):
                raise ValueError(f"Color/SAN mismatch at ply {index}.")
            if board.turn == chess.WHITE:
                decision = decisions[decision_index]
                decision_index += 1
                root = {"fen": board.fen(), "history": previous_history,
                        "policy_game_id": state["policy_game_id"]}
                if decision["position_id"] != position_id(root):
                    raise ValueError(f"Wrong decision root/game/history at ply {index}.")
                expected = select_move(board.fen(), simulate_position, previous_white[-4:])
                for key in ("policy_version", "root_fen", "chosen_move", "primary_score",
                            "secondary", "pv", "completed_depth", "simulator_calls"):
                    if decision.get(key) != expected.get(key):
                        raise ValueError(f"Policy mismatch for {key} at ply {index}.")
                if decision["chosen_move"] != move.uci() or decision["committed_move"] != move.uci():
                    raise ValueError(f"Move did not follow policy at ply {index}.")
                previous_white.append(move.uci())
            elif opponent_mode == "minimax":
                bot._board = board.copy(stack=True)
                if bot._choose_engine_move() != move:
                    raise ValueError(f"Opponent mismatch at ply {index}.")
            board.push(move)
            previous_history.append(row)
            # A completed turn is either Black's reply or a terminal White move.
            if not board.turn or index % 2 == 0:
                if index % 2 == 0 or board.outcome(claim_draw=True) is not None:
                    decision = decisions[decision_index - 1]
                    if decision["post_fen"] != board.fen():
                        raise ValueError(f"Post-move FEN mismatch at ply {index}.")
                    expected_reply = move.uci() if index % 2 == 0 else None
                    if decision["engine_move"] != expected_reply:
                        raise ValueError(f"Logged Black reply mismatch at ply {index}.")
        if decision_index != len(decisions):
            raise ValueError("Extra/missing decision records.")
        if board.fen() != state["fen"]:
            raise ValueError("Final FEN does not match history.")
        outcome = board.outcome(claim_draw=True)
        if bool(outcome) != bool(state["game_over"]):
            raise ValueError("Final outcome and game_over differ.")
        if outcome:
            expected_status = ("White wins" if outcome.winner == chess.WHITE
                               else "Black wins" if outcome.winner == chess.BLACK else "Draw")
            if state["status"] != expected_status:
                raise ValueError("Reported winner/status differs from replay.")
        return {
            "validated": True, "plies": board.ply(), "decision_count": decision_index,
            "result": outcome.result() if outcome else "*",
            "termination": outcome.termination.name.lower() if outcome else None,
            "all_white_moves_follow_policy": True,
            "all_black_moves_match_bot": True if opponent_mode == "minimax" else None,
            "opponent_validation": "exact_replay" if opponent_mode == "minimax" else "legal_moves_only",
            "game_over": outcome is not None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--patch", type=Path, default=REPO / "artifacts/fix.patch")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.result.parent / "replay-report.json"
    decision_path = args.decisions or args.result.parent / "decisions.jsonl"
    try:
        state = json.loads(args.result.read_text())
        state["_manifest_path"] = args.result.parent / "manifest.json"
        decisions = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
        report = verify(state, decisions, args.patch)
    except Exception as exc:
        report = {"validated": False, "error": str(exc)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
