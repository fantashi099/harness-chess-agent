"""Development evaluation and legal move grounding regressions."""
import chess
import pytest

from assignment.agent.chess_policy_search import development_score, positional_score, select_move
from assignment.agent.chess_policy_harness import EVALUATION, board_facts, move_facts

OPENING = "rnbqkbnr/1ppp1ppp/p3p3/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"


def test_actual_game2_compares_development_at_equal_depth():
    board = chess.Board(OPENING)
    scores = {}
    for evaluator in ("king-safety-mobility-pawns-v1", EVALUATION):
        scores[evaluator] = {}
        for move in ("d1h5", "b1c3"):
            row = select_move(board, "regression", 2000, 3, [move], evaluation=evaluator)
            assert row["completed_depth"] == 3 and row["nodes_used"] <= 2000
            scores[evaluator][move] = row["primary_score"]
    assert scores["king-safety-mobility-pawns-v1"]["d1h5"] > scores["king-safety-mobility-pawns-v1"]["b1c3"]
    assert scores[EVALUATION]["b1c3"] > scores[EVALUATION]["d1h5"]
    assert board.fen() == OPENING and not board.move_stack


def test_development_is_symmetric_phase_weighted_and_preserves_tactics():
    board = chess.Board(OPENING)
    for move in ("b1c3", "d1h5", "e1e2"):
        after = board.copy()
        after.push_uci(move)
        assert development_score(after.mirror()) == -development_score(after)
    lost_rights = board.copy()
    lost_rights.castling_rights &= ~chess.BB_RANK_1
    assert development_score(board) > development_score(lost_rights)
    bare_kings = chess.Board("7k/8/8/3K4/8/8/8/8 w - - 0 1")
    assert development_score(bare_kings) == positional_score(bare_kings)
    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    row = select_move(mate, "mate", 1000, 2, evaluation=EVALUATION)
    mate.push_uci(row["chosen_move"])
    assert mate.is_checkmate() and row["primary_score"] == 99999
    # v5's evaluator and original opening shortcut are still the defaults.
    assert select_move(chess.Board(), "baseline")["reason"] == "opening_preference"


def test_blocked_rook_cannot_be_claimed_as_defender_and_captures_are_legal():
    board = chess.Board("r2qk2r/1Q1b1ppp/p4n2/2bPp3/2Bp4/N7/PPPP1PPP/R1B1K2R w KQkq - 1 12")
    before = board.fen()
    facts = move_facts(board, "c4a6")
    assert "a1" not in facts["geometric_defenders_after"]  # a3 knight blocks the rook.
    board.push_uci("c4a6")
    assert {r["move"] for r in facts["opponent_captures_after"]} == {
        m.uci() for m in board.legal_moves if board.is_capture(m)}
    for row in facts["opponent_captures_after"]:
        assert row["san"] == board.san(chess.Move.from_uci(row["move"]))
    board.pop()
    assert board.fen() == before and not board.move_stack


def test_castling_rights_are_distinct_from_current_legal_castling():
    facts = board_facts(chess.Board())
    assert facts["white_castling_rights"] == {"kingside": True, "queenside": True}
    assert facts["legal_white_castling"] == []
    assert set(facts["legal_development_moves"]) == {"b1a3", "b1c3", "g1f3", "g1h3"}
    assert facts["undeveloped_white_minors"] == ["b1", "c1", "f1", "g1"]
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert set(board_facts(board)["legal_white_castling"]) == {"e1g1", "e1c1"}
