"""r10 retry turn 24: invalid future plans must not block a verified LLM move."""
import chess
import pytest

from assignment.agent.chess_policy_harness import analyze, decision_from, feedback, prepare, validate_action, check_continuation
from experiments.chess.test_agent_harness import analysis, commit

FEN = "5rk1/2Q2pb1/6pp/1p1p1b2/3P4/P1P3B1/3qrPPP/4RK1R w - - 0 24"


@pytest.fixture(scope="module")
def verified():
    board = chess.Board(FEN)
    session = prepare(board, "regression", 6000, 4)
    analyze(board, "regression", session, analysis(nodes=3000, depth=4))
    session.get("search_cache", {}).clear()  # Preserve the original shallow cutoff; cache has separate tests.
    analyze(board, "regression", session, analysis(["e1e2"], nodes=2000, depth=4))
    assert feedback(session)["commit_ready_moves"] == ["e1e2"]
    return board, session


@pytest.mark.parametrize("line", [
    ["e1e2", "b5c4", "c7e7"],
    ["e1e2", "d2e2", "c7e7", "f5e2"],
    ["e1e2", "b5c4", "d4c5", "d5c4"],
    ["e1e2"],
    None,
])
def test_actual_failed_continuations_and_omission_can_commit(verified, line):
    board, session = verified
    action = commit("e1e2", line or ["e1e2"])
    if line is None:
        del action["continuation"]
    decision = decision_from(session, action, board, "regression")
    assert decision["chosen_move"] == decision["llm_action"]["move"] == "e1e2"
    assert decision["llm_action"] == action and decision["nodes_used"] <= 6000
    checks = decision["continuation_checks"]
    assert checks["plies"][0]["uci"] == "e1e2"
    assert checks["valid"] == (line is None or len(line) == 1)
    assert board.fen() == FEN


def test_optional_plan_does_not_bypass_current_move_or_schema_checks(verified):
    board, session = verified
    with pytest.raises(ValueError, match="legal UCI"):
        decision_from(session, dict(commit("e1e2"), move="a1a8"), board, "regression")
    with pytest.raises(ValueError, match="starting with"):
        decision_from(session, commit("e1e2", ["f1e2"]), board, "regression")
    with pytest.raises(ValueError, match="unexpected:.*extra"):
        validate_action(dict(commit("e1e2"), extra=True), session["context"])
    with pytest.raises(ValueError, match="Action must"):
        validate_action({"action": []}, session["context"])


def test_invalid_suffix_after_mate_is_a_warning_not_an_executed_move():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    checks = check_continuation(board, commit("f7g7", ["f7g7", "h8h7"]), strict=False)
    assert not checks["valid"] and "after terminal" in checks["error"]
    assert checks["terminal_result"] == "1-0" and len(checks["plies"]) == 1
