"""Tactical regressions from v6-safe2400-6r5YnAUJ; no provider or engine calls."""
import chess
import pytest

from assignment.agent.chess_policy_harness import (
    analyze, check_continuation, decision_from, feedback, prepare,
)
from experiments.chess.test_agent_harness import analysis, commit


@pytest.mark.parametrize("fen,move,cap,score", [
    ("r1b1k2r/1ppqbp1p/2np1np1/1p4B1/4P3/2Q2N2/PPP2PPP/RN3RK1 w kq - 0 11", "g5f6", 2000, -202),
    ("8/2p3kp/4b3/1PbpP1p1/8/5PP1/2qN1N1P/3R3K w - - 1 32", "d2e4", 1000, -825),
])
@pytest.mark.parametrize("evaluator", ["king-safety-mobility-pawns-v1", "development-king-safety-v2"])
def test_shallow_request_exposes_tactical_loss_and_requires_replan(monkeypatch, evaluator, fen, move, cap, score):
    monkeypatch.setattr("assignment.agent.chess_policy_harness.EVALUATION", evaluator)
    if evaluator == "development-king-safety-v2" and move == "g5f6":
        score = -165
    board = chess.Board(fen)
    session = prepare(board, "root", 6000, 4)
    analyze(board, "test", session, analysis(nodes=3000, depth=2))
    verified = analyze(board, "test", session, analysis([move], nodes=cap, depth=2))
    assert verified["effective_max_depth"] == 4
    assert verified["evidence"]["completed_depth"] == 3
    assert verified["evidence"]["primary_score"] == score
    report = feedback(session)
    assert report["replan_required"] and report["replan_events"][0]["drop"] >= 150
    with pytest.raises(ValueError, match="Replan required after score drop"):
        decision_from(session, commit(move), board, "test")
    assert board.fen() == fen and not board.move_stack and session["nodes_used"] <= 6000


def test_score_drop_needs_a_comparable_alternative_but_llm_keeps_final_choice():
    board = chess.Board("8/2p1bk1p/3pbp1B/6p1/1P2P3/5P2/2qN1NPP/3R2K1 w - - 0 27")
    session = prepare(board, "root", 6000, 4)
    analyze(board, "test", session, analysis(nodes=2000, depth=2))
    analyze(board, "test", session, analysis(["b4b5"], nodes=1500, depth=2))
    choice = commit("b4b5", ["b4b5", "f7g6", "h6g5"])
    with pytest.raises(ValueError, match="Replan required"):
        decision_from(session, choice, board, "test")
    analyze(board, "test", session, analysis(["g1h1"], nodes=80, depth=2))
    assert feedback(session)["replan_required"]
    analyze(board, "test", session, analysis(["g1h1"], nodes=2000, depth=2))
    assert not feedback(session)["replan_required"]
    decision = decision_from(session, choice, board, "test")
    assert decision["chosen_move"] == "b4b5" and decision["replanned"]
    assert decision["replan_events"][0]["compared_moves"] == ["g1h1"]
    assert decision["comparison_count"] == 1 and session["nodes_used"] <= 6000


@pytest.mark.parametrize("evaluator,shallow_score", [("king-safety-mobility-pawns-v1", -1799),
                                                    ("development-king-safety-v2", -1781)])
def test_deeper_mate_evidence_survives_shallow_verification_and_sole_move_is_merged(monkeypatch, evaluator, shallow_score):
    monkeypatch.setattr("assignment.agent.chess_policy_harness.EVALUATION", evaluator)
    board = chess.Board("8/6kp/1Pp1b3/4P1P1/4p1P1/5q2/5b2/7K w - - 3 41")
    assert board.legal_moves.count() == 1
    session = prepare(board, "root", 6000, 4)
    broad = analyze(board, "test", session, analysis(nodes=4000, depth=4))
    assert broad["verification"] and broad["evidence"]["primary_score"] == -99996
    merged = decision_from(session, commit("h1h2", ["h1h2"]), board, "test")
    assert merged["verification_count"] == 1 and not merged["replanned"]
    assert merged["unavoidable_mate"] and merged["proven_refutations"]["h1h2"]["mate_in"] == 2
    session.get("search_cache", {}).clear()  # Force a genuinely shallow retry to test evidence retention.
    shallow = analyze(board, "test", session, analysis(["h1h2"], nodes=200, depth=2))
    assert shallow["evidence"]["completed_depth"] == 2 and shallow["evidence"]["primary_score"] == shallow_score
    decision = decision_from(session, commit("h1h2", ["h1h2"]), board, "test")
    assert decision["completed_depth"] == 4 and decision["primary_score"] == -99996
    assert decision["score_source_action_id"] == broad["action_id"]
    assert decision["verification_action_id"] == shallow["action_id"]


def test_continuation_cannot_invent_a_recapture_and_material_is_accounted():
    board = chess.Board("8/2p1b2p/3pbpkB/1P4p1/4P3/5P2/2qN1NPP/3R2K1 w - - 1 28")
    before = board.fen()
    with pytest.raises(ValueError, match="Illegal continuation at ply 3: f4g5"):
        check_continuation(board, commit("h6g5", ["h6g5", "f6g5", "f4g5"]))
    report = check_continuation(board, commit("h6g5", ["h6g5", "f6g5", "f3f4"]))
    assert [row["change_from_root"] for row in report["plies"]] == [100, -230, -230]
    assert board.fen() == before and not board.move_stack
    with pytest.raises(ValueError, match="at least White move"):
        check_continuation(board, commit("h6g5", ["h6g5"]))


def test_continuation_stops_at_native_terminal_position():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    assert check_continuation(board, commit("f7g7", ["f7g7"]))["terminal_result"] == "1-0"
    with pytest.raises(ValueError, match="after terminal"):
        check_continuation(board, commit("f7g7", ["f7g7", "h8h7"]))
