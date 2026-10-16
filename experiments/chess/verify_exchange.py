"""Offline continuation checks and bounded quiescence using unchanged v2 evaluation.

Run self-checks and regenerate evidence from the repository root:
    .venv/bin/python experiments/chess/verify_exchange.py

verify_exchange(snapshot, candidate_or_leaf=None, budget=None) accepts a FEN or
chess.Board. None treats the snapshot as a search leaf; a UCI string applies
one legal candidate first. Board inputs retain their supplied move stack.
This is a leaf verifier, not a full candidate comparison or a live executor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess

try:
    from assignment.agent.chess_policy import POLICY_VERSION, VALUES, evaluate
except ModuleNotFoundError:  # Flat, explicitly packaged Modal files.
    from chess_policy import POLICY_VERSION, VALUES, evaluate

VERSION = "exchange-verifier-1"
SCOPE = "Captures and promotions; all legal check evasions; stand-pat only outside check. Quiet non-evasion moves are excluded."


def _copy_board(snapshot: str | chess.Board) -> chess.Board:
    if isinstance(snapshot, chess.Board):
        board = snapshot.copy(stack=True)
    elif isinstance(snapshot, str) and len(snapshot.split()) == 6:
        board = chess.Board(snapshot)
    else:
        raise ValueError("Snapshot must be a chess.Board or a six-field FEN.")
    if not board.is_valid():
        raise ValueError("Snapshot describes an invalid position.")
    return board


def _state(board: chess.Board) -> dict:
    outcome = board.outcome(claim_draw=True)
    winner = None
    if outcome is not None and outcome.winner is not None:
        winner = "white" if outcome.winner else "black"
    return {
        "game_over": outcome is not None,
        "winner": winner,
        "squares": {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
        "turn": "white" if board.turn else "black",
        "in_check": board.is_check(),
    }


def _push(board: chess.Board, uci: str) -> str:
    if not isinstance(uci, str):
        raise ValueError("Moves must be UCI strings.")
    if board.outcome(claim_draw=True) is not None:
        raise ValueError("Cannot move in a terminal position.")
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {uci}")
    san = board.san(move)
    board.push(move)
    return san


def _moves(board: chess.Board) -> list[chess.Move]:
    checked = board.is_check()
    return sorted(
        (m for m in board.legal_moves if checked or board.is_capture(m) or m.promotion),
        key=lambda m: m.uci(),
    )


def verify_continuation(snapshot: str | chess.Board, continuation: list[str]) -> dict:
    """Check a supplied line only; make no claim about alternative replies."""
    if not isinstance(continuation, (list, tuple)):
        raise ValueError("Continuation must be a list or tuple of UCI strings.")
    board = _copy_board(snapshot)
    root_fen = board.fen(en_passant="fen")
    steps = []
    for ply, uci in enumerate(continuation, 1):
        san = _push(board, uci)
        steps.append({
            "relative_ply": ply, "uci": uci, "san": san, "legal_before_push": True,
            "fen_after": board.fen(en_passant="fen"),
            "material_white_minus_black": sum(
                VALUES[p.symbol().upper()] * (1 if p.color else -1)
                for p in board.piece_map().values()
            ),
            "v2_static_score": evaluate(_state(board)),
        })
    return {"root_fen": root_fen, "continuation": list(continuation), "steps": steps,
            "score": evaluate(_state(board)), "scope": "Supplied continuation only."}


def verify_exchange(
    snapshot: str | chess.Board,
    candidate_or_leaf: str | None = None,
    budget: dict[str, int] | None = None,
) -> dict:
    """Return White's score, or None if any required branch hits the budget.

    Nodes count visited quiescence positions, including its starting position.
    max_qdepth counts additional plies after the optional candidate. Completion
    means completion within SCOPE, not proof of safety against all legal moves.
    """
    if budget is not None and not isinstance(budget, dict):
        raise ValueError("Budget must be a dictionary.")
    limits = {"max_nodes": 1000, "max_qdepth": 8}
    if budget is not None:
        if budget.keys() - limits.keys():
            raise ValueError("Only max_nodes and max_qdepth are supported.")
        limits.update(budget)
    if type(limits["max_nodes"]) is not int or limits["max_nodes"] < 1:
        raise ValueError("max_nodes must be a positive integer.")
    if type(limits["max_qdepth"]) is not int or not 0 <= limits["max_qdepth"] <= 64:
        raise ValueError("max_qdepth must be an integer from 0 to 64.")

    board = _copy_board(snapshot)
    root_fen = board.fen(en_passant="fen")
    input_history = [m.uci() for m in board.move_stack]
    prefix = []
    if candidate_or_leaf is not None:
        _push(board, candidate_or_leaf)
        prefix.append(candidate_or_leaf)
    analysis_fen = board.fen(en_passant="fen")
    terminal_root = board.outcome(claim_draw=True) is not None
    nodes = root_moves_completed = 0
    root_moves = []
    cutoff_reason = None

    # ponytail: enumerate every scoped branch; add alpha-beta if node cost matters.
    def search(depth: int) -> tuple[int | None, list[str]]:
        nonlocal nodes, cutoff_reason, root_moves_completed
        nodes += 1
        state = _state(board)
        if state["game_over"]:
            return evaluate(state), []
        moves = _moves(board)
        if depth == 0:
            root_moves.extend(m.uci() for m in moves)
        if not moves:
            return evaluate(state), []
        if depth >= limits["max_qdepth"]:
            cutoff_reason = "max_qdepth"
            return None, []
        maximizing = board.turn == chess.WHITE
        best = None if state["in_check"] else evaluate(state)
        best_line = []
        for move in moves:
            if nodes >= limits["max_nodes"]:
                cutoff_reason = "max_nodes"
                return None, []
            board.push(move)
            try:
                score, line = search(depth + 1)
            finally:
                board.pop()
            if score is None:
                return None, [move.uci()] + line
            if depth == 0:
                root_moves_completed += 1
            if best is None or (score > best if maximizing else score < best):
                best, best_line = score, [move.uci()] + line
        return best, best_line

    score, line = search(0)
    return {
        "helper_version": VERSION, "evaluation_version": POLICY_VERSION,
        "root_fen": root_fen, "analysis_fen": analysis_fen,
        "input_history_uci": input_history,
        "history_mode": "provided_move_stack" if isinstance(snapshot, chess.Board) else "fen_only",
        "budget": limits, "scope": SCOPE,
        "score": score, "continuation": prefix + line, "nodes": nodes,
        "stop_reason": cutoff_reason or ("terminal" if terminal_root else "quiescence_complete"),
        "cutoff_reached": cutoff_reason is not None,
        "continuation_role": "cutoff_path" if cutoff_reason else "representative_branch",
        "root_moves_considered": root_moves, "root_moves_completed": root_moves_completed,
    }


def demo(output: Path) -> None:
    """Runnable assertions and evidence for the Step 3 checkpoint."""
    repo = Path(__file__).resolve().parents[2]
    batch = repo / "artifacts/assignment2/stockfish_20261007_191056"
    cases = json.loads((batch / "failure-cases.json").read_text())["cases"]
    case = next(c for c in cases if c["id"] == "game_4_white_6")
    lesson = json.loads((repo / "learning/step-02-lesson.json").read_text())
    board = chess.Board()
    result = json.loads((batch / "game_4/result.json").read_text())
    for entry in result["history"][:10]:
        _push(board, entry["uci"])
    assert board.fen(en_passant="fen") == case["root_fen"]
    original = (board.fen(en_passant="fen"), list(board.move_stack))
    sequence = verify_continuation(board, ["g2g3", "g5d2", "d1d2"])
    assert [s["material_white_minus_black"] for s in sequence["steps"]] == [0, -100, 800]
    assert sequence["steps"] == lesson["evidence"]["verified_continuation"]
    final = chess.Board(sequence["steps"][-1]["fen_after"])
    assert not final.pieces(chess.QUEEN, chess.BLACK)
    assert (board.fen(en_passant="fen"), board.move_stack) == original
    records = [{"id": "step_02_supplied_recapture", "source_case": case["id"],
                "result": sequence, "passed": True}]

    leaf = board.copy(stack=True)
    for uci in ("g2g3", "g5d2"):
        _push(leaf, uci)
    unchanged = (leaf.fen(en_passant="fen"), list(leaf.move_stack))
    recapture = verify_exchange(leaf, "d1d2")
    assert not recapture["cutoff_reached"] and recapture["score"] == 812
    assert recapture["continuation"] == ["d1d2"]
    records.append({"id": "verified_recapture_candidate", "source_case": case["id"],
                    "result": recapture, "passed": True,
                    "limitation": "This verifies Qxd2 in the supplied branch, not the value of g3 against every Black reply."})
    result = verify_exchange(leaf, budget={"max_nodes": 10000, "max_qdepth": 12})
    assert result["cutoff_reached"] and result["score"] is None
    assert result["stop_reason"] == "max_qdepth"
    assert result["root_moves_considered"] == sorted(m.uci() for m in leaf.legal_moves)
    assert "d1d2" in result["root_moves_considered"]
    assert (leaf.fen(en_passant="fen"), leaf.move_stack) == unchanged
    records.append({"id": "quiescence_recapture_leaf", "source_case": case["id"],
                    "static_score_before": evaluate(_state(leaf)), "result": result, "passed": True,
                    "limitation": "The Kf1 evasion leads to a long tactical branch that reaches the depth budget. The whole leaf remains unresolved despite verifying the recapture candidate separately."})

    checked = chess.Board("q3r3/7k/8/8/8/8/8/Q3K3 w - - 0 1")
    assert checked.is_valid() and checked.is_check()
    result = verify_exchange(checked, budget={"max_nodes": 10000, "max_qdepth": 12})
    assert not result["cutoff_reached"] and result["continuation"]
    assert result["score"] < evaluate(_state(checked))
    assert result["root_moves_completed"] == len(list(checked.legal_moves))
    records.append({"id": "check_evasions_without_stand_pat", "provenance": "Hand-constructed legal position: every evasion permits loss of the White queen.",
                    "static_score_before": evaluate(_state(checked)), "result": result, "passed": True})

    for key, limit in (("max_nodes", 1), ("max_qdepth", 0)):
        result = verify_exchange(leaf, budget={key: limit})
        assert result["cutoff_reached"] and result["stop_reason"] == key
        assert result["score"] is None and result["nodes"] <= result["budget"]["max_nodes"]
        assert not result["root_moves_completed"]
        records.append({"id": f"budget_{key}", "source_case": case["id"], "result": result, "passed": True})

    terminal = verify_exchange("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", budget={"max_nodes": 1, "max_qdepth": 0})
    assert terminal["score"] == 100000 and terminal["stop_reason"] == "terminal"
    assert not terminal["cutoff_reached"]
    records.append({"id": "terminal_before_heuristic_and_depth_limit", "provenance": "Hand-constructed checkmate.", "result": terminal, "passed": True})

    for fen, white in (("7k/P7/8/8/8/8/8/7K w - - 0 1", True),
                       ("7k/8/8/8/8/8/p7/7K b - - 0 1", False)):
        result = verify_exchange(fen)
        assert not result["cutoff_reached"] and result["continuation"]
        assert result["score"] >= 900 if white else result["score"] <= -900
        assert all(len(m) == 5 for m in result["root_moves_considered"])
        records.append({"id": "white_max_promotion" if white else "black_min_promotion",
                        "provenance": "Hand-constructed legal non-capturing promotion.", "result": result, "passed": True})

    repeated = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        move = chess.Move.from_uci(uci)
        assert move in repeated.legal_moves
        repeated.push(move)
    with_history = verify_exchange(repeated)
    fen_only = verify_exchange(repeated.fen(en_passant="fen"))
    assert with_history["stop_reason"] == "terminal" and with_history["score"] == 0
    assert fen_only["stop_reason"] == "quiescence_complete"
    assert len(repeated.move_stack) == 8
    records.append({"id": "repetition_requires_supplied_history", "provenance": "Full legal knight-cycle prefix from the initial board.",
                    "with_history": with_history, "fen_only": fen_only, "passed": True})

    fork_case = next(c for c in cases if c["id"] == "game_5_white_7")
    fork = _copy_board(fork_case["root_fen"])
    _push(fork, "d1g4")
    quiet = chess.Move.from_uci("f7f5")
    assert quiet in fork.legal_moves and not fork.is_check()
    assert quiet not in _moves(fork)
    before = fork.fen(en_passant="fen")
    _push(fork, quiet.uci())
    targets = sorted(chess.square_name(s) for s in fork.attacks(chess.F5)
                     if fork.piece_at(s) is not None and fork.piece_at(s).color == chess.WHITE)
    assert targets == ["e4", "g4"]
    records.append({"id": "quiet_fork_outside_quiescence_scope", "source_case": fork_case["id"],
                    "leaf_fen": before, "legal_reply": "f7f5", "included_in_search_scope": False,
                    "attacked_white_targets": targets, "passed": True,
                    "limitation": "Completing capture/promotion quiescence cannot certify safety against this excluded quiet pawn fork."})

    # Exercise candidate application, snapshot isolation, and trust-boundary validation.
    candidate = verify_exchange(board, "g2g3", {"max_nodes": 1})
    assert candidate["analysis_fen"] == sequence["steps"][0]["fen_after"]
    assert candidate["continuation"][0] == "g2g3" and candidate["cutoff_reached"]
    assert (board.fen(en_passant="fen"), board.move_stack) == original
    for snapshot, move, budget in ((chess.STARTING_FEN, "e2e5", None),
                                   ("8/8/8/8/8/8/8/8 w - - 0 1", None, None),
                                   (chess.STARTING_FEN, None, {"max_nodes": 0}),
                                   (chess.STARTING_FEN, None, {"max_qdepth": True})):
        try:
            verify_exchange(snapshot, move, budget)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted.")

    report = {"helper_version": VERSION, "evaluation_version": POLICY_VERSION,
              "reproduce_command": ".venv/bin/python experiments/chess/verify_exchange.py",
              "scope": SCOPE, "candidate_safety_proven": False,
              "passed": True, "checks": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(records)} evidence checks plus candidate/isolation/input assertions; wrote {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "learning/step-03-exchange-checks.json")
    demo(parser.parse_args().output)
