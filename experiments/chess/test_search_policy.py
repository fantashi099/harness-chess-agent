"""Search correctness, interrupted iterations, history and trusted v5 integration."""
from pathlib import Path
import json

import chess
from fastapi.testclient import TestClient

from assignment.agent.chess_policy import evaluate, position_id
from assignment.agent.chess_policy_search import MATE, select_move, static_score, positional_score
from assignment.agent.chess_agent import ChessAgent
from assignment.chess_artifacts import save_chess_run
from assignment.fixed_chess_server import create_app
from experiments.chess.test_chess_transactions import patched_app_source, REGRESSIONS

REPO = Path(__file__).resolve().parents[2]


def test_v6_evaluator_values_king_cover_mobility_and_pawn_structure():
    shielded = chess.Board("k7/8/8/3q4/8/8/5PPP/6K1 w - - 0 1")
    exposed = chess.Board("k7/8/8/3q4/8/8/PPP5/6K1 w - - 0 1")
    assert positional_score(shielded) > positional_score(exposed)
    center = chess.Board("6k1/8/8/8/3N4/8/8/6K1 w - - 0 1")
    edge = chess.Board("6k1/8/8/8/8/8/8/N5K1 w - - 0 1")
    assert positional_score(center)-static_score(center) > positional_score(edge)-static_score(edge)
    connected = chess.Board("6k1/8/8/8/8/3P4/2P5/6K1 w - - 0 1")
    doubled = chess.Board("6k1/8/8/8/8/3P4/3P4/6K1 w - - 0 1")
    assert positional_score(connected) > positional_score(doubled)
    for board in (shielded, exposed, center, edge, connected, doubled):
        assert positional_score(board.mirror()) == -positional_score(board)
    # v5 keeps its original evaluator and opening shortcut; v6 must actually search.
    opening = chess.Board()
    assert select_move(opening, "test")["reason"] == "opening_preference"
    result = select_move(opening, "test", 1000, 1, evaluation="king-safety-mobility-pawns-v1")
    assert result["search_performed"] and result["completed_depth"] == 1


def exhaustive(board, depth, ply=0):
    end = board.outcome(claim_draw=True)
    if end:
        return 0 if end.winner is None else MATE-ply if end.winner else -MATE+ply
    if not depth:
        return static_score(board)
    scores = []
    for move in board.legal_moves:
        board.push(move)
        scores.append(exhaustive(board, depth-1, ply+1))
        board.pop()
    return (max if board.turn else min)(scores)


def test_static_and_alpha_beta_agree_with_exhaustive():
    for fen, *_ in REGRESSIONS:
        board = chess.Board(fen)
        state = {"squares": {chess.square_name(s): p.symbol() for s,p in board.piece_map().items()},
                 "game_over": False, "in_check": board.is_check(), "turn": "white"}
        assert static_score(board) == evaluate(state)
    board = chess.Board("6k1/7p/8/8/8/8/P7/1K6 w - - 0 1")
    for depth in (1, 2, 3):
        expected = exhaustive(board, depth)
        result = select_move(board, "test", 100000, depth)
        assert result["completed_depth"] == depth and not result["cutoff_reached"]
        assert result["primary_score"] == expected
        board.push_uci(result["chosen_move"])
        assert exhaustive(board, depth-1, 1) == expected
        board.pop()


def test_budget_retains_completed_iteration_without_mutation():
    board = chess.Board("6k1/7p/8/8/8/8/P7/1K6 w - - 0 1")
    before = board.fen(), list(board.move_stack)
    completed = select_move(board, "test", 100000, 2)
    interrupted = select_move(board, "test", completed["nodes_used"]+1, 4)
    assert interrupted["cutoff_reached"] and interrupted["completed_depth"] == 2
    assert interrupted["iterations"][-1]["completed"] is False
    for key in ("chosen_move", "primary_score", "pv"):
        assert interrupted[key] == completed[key]
    assert interrupted["nodes_used"] == completed["nodes_used"]+1
    fallback = select_move(board, "test", 1, 4)
    assert fallback["primary_score"] is None and fallback["completed_depth"] == 0
    assert chess.Move.from_uci(fallback["chosen_move"]) in board.legal_moves
    assert before == (board.fen(), board.move_stack)


def test_history_draw_and_shortest_mate():
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8")*2:
        board.push_uci(move)
    result = select_move(board, "draw")
    assert result["reason"] == "terminal_position" and result["primary_score"] == 0
    assert chess.Board(board.fen()).outcome(claim_draw=True) is None
    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = select_move(mate, "mate")
    assert result["primary_score"] == MATE-1 and result["completed_depth"] == 1
    mate.push_uci(result["chosen_move"])
    assert mate.is_checkmate()


def test_quiescence_rejects_hanging_queen_capture_at_shallow_horizon():
    board = chess.Board("3r3k/8/8/3p4/4Q3/8/8/7K w - - 0 1")
    result = select_move(board, "exchange", 6000, 1)
    assert result["completed_depth"] == 1 and result["chosen_move"] != "e4d5"
    assert result["quiescence"]["nodes"] > 0
    assert result["nodes_used"] <= 6000


def test_search_transaction_and_recoverable_artifacts(monkeypatch, tmp_path, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    with TestClient(create_app(policy="search", max_nodes=100, search_depth=3)) as client:
        agent = ChessAgent(object(), model="test-model", http_client=client,
                           auto_stop_environment=False, chess_policy="search")
        root = position_id(agent.last_state)
        observations = agent.execute_tool_calls([{"id":"turn", "function":{"name":"select_and_play","arguments":"{}"}}])
        assert "<chess_error>" not in observations[0]["content"]
        repeated = client.post("/api/select-and-play", json={"expected_position_id":root}).json()
        assert repeated["replayed"] and len(repeated["state"]["history"]) == 2
        recordings = []
        def record(path):
            assert path.is_file() and (path.parent/"manifest.json").is_file()
            recordings.append(path)
        monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", record)
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        assert recordings == [tmp_path/"result.json"]
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["search_config"] == {"max_depth":3, "evaluation":"development-king-safety-v2"} and manifest["max_nodes"] == 100
        assert not manifest["memory_used"] and not manifest["teacher_used"]
        from scripts.replay_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["validated"]
        client.post("/api/reset").raise_for_status()
        assert client.post("/api/select-and-play", json={"expected_position_id":root}).status_code == 409
        assert client.get("/api/policy-decisions").json() == []
