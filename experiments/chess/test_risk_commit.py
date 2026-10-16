"""Budget-exhausted LLM commits acknowledge risk without claiming unavoidable mate."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import chess
import chess.pgn
import pytest
from fastapi.testclient import TestClient

from assignment.agent import chess_policy_harness as harness
from assignment.agent.chess_agent import ChessAgent
from assignment.chess_artifacts import save_chess_run
from assignment.fixed_chess_server import create_app
from experiments.chess.test_agent_harness import analysis, commit, fake_client
from experiments.chess.test_chess_transactions import patched_app_source

CASES = [
    ("2k4r/pp1r1p2/4b1p1/q3P2B/2Pp2PQ/1p6/P4P1P/K1R4R w - - 0 25",
     "a2a4", "a1b2", ["a2a4", "a5a4", "a1b1", "a4a2"]),
    ("4r1k1/5p1p/6p1/2P5/1P6/2bpqP2/7P/1R1K3R w - - 1 47",
     "b1b2", "c5c6", ["b1b2", "e3e1", "h1e1", "e8e1"]),
]
REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fen,first,second,line", CASES)
def test_exhaustion_keeps_llm_authority_and_explicit_risk(fen, first, second, line):
    board = chess.Board(fen)
    session = harness.prepare(board, "root", 6000, 4)
    # Keep the recorded a4 suggestion at depth 1; deeper forcing order now suggests a3.
    broad_depth = 1 if first == "a2a4" else 4
    harness.analyze(board, "test", session, analysis(nodes=3000, depth=broad_depth))
    harness.analyze(board, "test", session, analysis([first], nodes=2000, depth=4))
    action = dict(commit(first, line), accept_risk=True)
    with pytest.raises(ValueError, match="exhausted"):
        harness.decision_from(session, action, board, "test")
    harness.analyze(board, "test", session, analysis([second], nodes=1000, depth=4))
    if session["max_nodes"]-session["nodes_used"] >= 1000:
        harness.analyze(board, "test", session, analysis([first], nodes=1000, depth=4))
    report = harness.feedback(session)
    assert (session["nodes_used"] == session["max_nodes"] or len(session["analyses"]) == harness.MAX_ANALYSES)
    assert not report["commit_ready_moves"]
    assert report["risk_commit_moves"] == sorted([first, second])
    assert not report["unavoidable_mate"]
    with pytest.raises(ValueError, match="Unsafe"):
        harness.decision_from(session, commit(first, line), board, "test")
    decision = harness.decision_from(session, action, board, "test")
    assert decision["chosen_move"] == decision["llm_action"]["move"] == first
    assert decision["nodes_used"] <= 6000 and not decision["unavoidable_mate"]
    assert decision["risk_acceptance"]["known_refutation"] == report["proven_refutations"][first]
    assert decision["risk_acceptance"]["reason"] == "analysis_budget_exhausted"
    assert decision["reason"] == "explicit_llm_commit" and board.fen() == fen
    with pytest.raises(ValueError, match="legal UCI"):
        harness.decision_from(session, dict(action, move="a1a8"), board, "test")


def test_risk_field_is_strict_boolean():
    session = harness.prepare(chess.Board(), "test", 100, 1)
    with pytest.raises(ValueError, match="boolean"):
        harness.validate_action(dict(commit("e2e4"), accept_risk="true"), session["context"])


def test_unrefuted_candidates_take_precedence_without_selecting_a_move(monkeypatch):
    session = {"analyses": [{"safety": {"move": move}, "evidence": {"completed_depth": 1}}
                            for move in ("a2a4", "a1b2")], "nodes_used": 10, "max_nodes": 10}
    monkeypatch.setattr(harness, "exact_evidence", lambda _: {"a2a4": None, "a1b2": None})
    monkeypatch.setattr(harness, "proven_refutations", lambda _: {"a1b2": {"mate_in": 1}})
    assert harness.risk_commit_moves(session, []) == ["a2a4"]
    assert harness.risk_commit_moves(session, ["a2a4"]) == []
    session["nodes_used"] = 9
    assert harness.risk_commit_moves(session, []) == []


@pytest.mark.parametrize("fen,first,second,line", CASES)
def test_fake_llm_reaches_risk_commit_and_saved_replay_validates(monkeypatch, patched_app_source,
                                                               tmp_path, fen, first, second, line):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    engine = Mock()
    engine.id = {"name": "Stockfish 17.1"}
    engine.play.return_value = SimpleNamespace(move=chess.Move.from_uci(line[1]))
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", Mock(return_value=engine))
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4,
                               opponent="stockfish", stockfish_elo=2000)) as client:
        client.app.state.chess_game._board = chess.Board(fen)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        broad_depth = 1 if first == "a2a4" else 4
        agent.client = fake_client([analysis(nodes=3000, depth=broad_depth),
            analysis([first], nodes=2000, depth=4), analysis([second], nodes=1000, depth=4),
            analysis([first], nodes=1000, depth=4), dict(commit(first, line), accept_risk=True)], requests)
        observation = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in observation[0]["content"] and len(requests) == 5
        feedback = json.loads(requests[-1]["messages"][-1]["content"])["feedback"]
        assert feedback["allowed_actions"] == ["commit"] and first in feedback["risk_commit_moves"]
        assert not feedback["commit_ready_moves"] and not feedback["unavoidable_mate"]
        assert agent.chess_decisions[0]["risk_acceptance"]
        monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        # This fixture starts midgame; production full-game exports start at the initial board.
        manifest_path = tmp_path/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["initial_fen"] = fen
        manifest_path.write_text(json.dumps(manifest))
        game = chess.pgn.Game()
        game.setup(chess.Board(fen))
        node = game
        for row in agent.last_state["history"]:
            node = node.add_variation(chess.Move.from_uci(row["uci"]))
        (tmp_path/"game.pgn").write_text(str(game)+"\n")
        from scripts.replay_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["all_commits_match_llm_response"]
