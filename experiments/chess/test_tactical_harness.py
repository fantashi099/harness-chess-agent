"""Regression for 43.g3 in v6-r5-2400-AkIa9jox, without paid services."""
import json
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_policy_harness import (
    TACTICAL_PROBE_NODES, analyze, checking_mate_refutation, decision_from, feedback, mate_safety, prepare,
)
from assignment.chess_artifacts import save_fixed_run
from assignment.fixed_chess_server import create_app
from experiments.chess.test_agent_harness import analysis, commit, fake_client
from experiments.chess.test_fixed_policy import patched_app_source

FEN = "8/5bk1/6r1/p1P4p/P4Q2/2P5/4q1PP/5N1K w - - 9 43"
REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("chosen", ["h5g4", "h5g5"])
def test_r8_turn104_closes_missing_proof_and_llm_finishes_game(monkeypatch, patched_app_source, chosen):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    from unittest.mock import Mock
    from types import SimpleNamespace
    fen = "8/5k2/8/7K/8/2q5/8/4q3 w - - 4 104"
    reply, mate = (("c3g3", "g3g6") if chosen == "h5g4" else ("c3e5", "e1h4"))
    defense = "g4h5" if chosen == "h5g4" else "g5h6"
    engine = Mock(id={"name": "Stockfish test"})
    engine.play.side_effect = [SimpleNamespace(move=chess.Move.from_uci(m)) for m in [reply, mate]]
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", Mock(return_value=engine))
    with TestClient(create_app(policy="v6", max_nodes=6000, search_depth=4,
                               opponent="stockfish", stockfish_elo=2000)) as client:
        client.app.state.chess_game._board = chess.Board(fen)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, fixed_chess_policy="v6")
        requests = []
        agent.client = fake_client([
            analysis(["h5g4", "h5g5", "h5h6"], nodes=3000, depth=4),
            analysis(["h5g5"], nodes=2000, depth=4), analysis(["h5g4"], nodes=1000, depth=4),
            commit(chosen, [chosen, reply, defense, mate]),
            analysis(nodes=1000, depth=4), analysis([defense], nodes=1000, depth=4),
            commit(defense, [defense, mate]),
        ], requests)
        call = [{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}]
        output = agent.execute_tool_calls(call)
        assert "<chess_error>" not in output[0]["content"] and len(requests) == 4
        report = json.loads(requests[3]["messages"][-1]["content"])["feedback"]
        assert report["remaining_nodes"] >= 0 and report["unavoidable_mate"]
        assert report["commit_ready_moves"] == ["h5g4", "h5g5"]
        assert set(report["proven_refutations"]) == {"h5g4", "h5g5", "h5h6"}
        assert report["proven_refutations"]["h5h6"]["black_reply"] == "e1h4"
        decision = agent.chess_decisions[0]
        last = decision["analysis_trace"][-1]
        extra = last["safety"]["remaining_root_safety"]
        assert extra["complete"] and 0 < extra["nodes"] <= TACTICAL_PROBE_NODES
        assert report["proven_refutations"]["h5h6"]["action_id"] == last["action_id"]
        assert decision["nodes_used"] <= 6000 and decision["chosen_move"] == decision["llm_action"]["move"] == chosen
        for trace in decision["analysis_trace"]:
            assert trace["evidence"]["nodes_used"] + (trace["safety"] or {}).get("nodes", 0) <= trace["action"]["max_nodes"]
        assert not agent.finished
        assert "<chess_error>" not in agent.execute_tool_calls(call)[0]["content"]
        assert agent.finished and agent.last_state["status"] == "Black wins"
        assert client.app.state.chess_game._board.is_checkmate()


@pytest.mark.parametrize("probe_cap", [1, 60])
def test_incomplete_remaining_root_probe_cannot_claim_unavoidable_mate(monkeypatch, probe_cap):
    from assignment.agent import chess_policy_harness as harness
    original = harness.mate_safety
    def limited(board, move, cap):
        return original(board, move, min(cap, probe_cap) if move == "h5h6" else cap)
    monkeypatch.setattr(harness, "mate_safety", limited)
    board = chess.Board("8/5k2/8/7K/8/2q5/8/4q3 w - - 4 104")
    before = board.fen(), list(board.move_stack)
    session = prepare(board, "root", 6000, 4)
    for moves, nodes in [(["h5g4", "h5g5", "h5h6"], 3000), (["h5g5"], 2000), (["h5g4"], 1000)]:
        trace = analyze(board, "test", session, analysis(moves, nodes=nodes, depth=4))
    assert not trace["safety"]["remaining_root_safety"]["complete"]
    report = feedback(session)
    assert not report["unavoidable_mate"] and not report["commit_ready_moves"]
    assert "h5h6" not in report["proven_refutations"]
    with pytest.raises(ValueError, match="mate in 2"):
        decision_from(session, commit("h5g4", ["h5g4", "c3g3", "g4h5"]), board, "test")
    assert session["nodes_used"] <= 6000 and before == (board.fen(), board.move_stack)


def test_refutation_certificate_covers_every_native_defense_without_mutation():
    board = chess.Board(FEN)
    before = board.fen(), list(board.move_stack)
    immediate = mate_safety(board, "g2g3", 1000)
    assert immediate["complete"] and immediate["mating_reply"] is None
    proof = checking_mate_refutation(board, "g2g3", immediate["checking_replies"], TACTICAL_PROBE_NODES)
    assert proof["status"] == "refuted" and proof["black_reply"] == "f7d5"
    assert 0 < proof["nodes"] <= TACTICAL_PROBE_NODES
    board.push_uci("g2g3")
    board.push_uci(proof["black_reply"])
    assert board.is_check()
    assert {m.uci() for m in board.legal_moves} == {r["white_defense"] for r in proof["defenses"]}
    for row in proof["defenses"]:
        board.push_uci(row["white_defense"])
        assert chess.Move.from_uci(row["black_mate"]) in board.legal_moves
        board.push_uci(row["black_mate"])
        assert board.is_checkmate()
        board.pop()
        board.pop()
    board.pop()
    board.pop()
    assert before == (board.fen(), board.move_stack)


@pytest.mark.parametrize("cap", [0, 1, 20, 292])
def test_probe_cutoff_is_unknown_with_no_partial_certificate(cap):
    board = chess.Board(FEN)
    immediate = mate_safety(board, "g2g3", 1000)
    proof = checking_mate_refutation(board, "g2g3", immediate["checking_replies"], cap)
    assert proof["status"] == "unknown" and proof["nodes"] <= cap
    assert proof["black_reply"] is None and not proof["defenses"]
    assert board.fen() == FEN and not board.move_stack


def test_no_checking_reply_is_only_not_found_in_the_stated_scope():
    board = chess.Board(FEN)
    immediate = mate_safety(board, "f4d4", 1000)
    proof = checking_mate_refutation(board, "f4d4", immediate["checking_replies"], 0)
    assert proof["status"] == "not_found" and proof["nodes"] == 0
    assert "quiet replies and longer mates not covered" in proof["scope"]
    board.push_uci("f4d4")
    assert board.is_check() and chess.Move.from_uci("f7d5") not in board.legal_moves


def test_refutation_survives_a_later_shallower_search():
    board = chess.Board(FEN)
    session = prepare(board, "root", 6000, 4)
    analyze(board, "test", session, analysis(nodes=2000, depth=2))
    first = analyze(board, "test", session, analysis(["g2g3"], nodes=1000, depth=4))
    assert first["safety"]["tactical_refutation"]["status"] == "refuted"
    later = analyze(board, "test", session, analysis(["g2g3"], nodes=300, depth=2))
    assert later["evidence"]["completed_depth"] == 2
    report = feedback(session)
    assert report["proven_refutations"]["g2g3"]["action_id"] == first["action_id"]
    assert not report["unavoidable_mate"] and "g2g3" not in report["commit_ready_moves"]
    with pytest.raises(ValueError, match="f7d5 mate in 2"):
        decision_from(session, commit("g2g3", ["g2g3", "f7d5", "h1g1"]), board, "test")
    assert session["nodes_used"] <= 6000 and board.fen() == FEN


def test_unknown_probe_is_reported_without_claiming_safety_or_overriding_llm(monkeypatch):
    monkeypatch.setattr("assignment.agent.chess_policy_harness.TACTICAL_PROBE_NODES", 1)
    board = chess.Board(FEN)
    session = prepare(board, "root", 6000, 4)
    analyze(board, "test", session, analysis(nodes=2000, depth=2))
    trace = analyze(board, "test", session, analysis(["g2g3"], nodes=1000, depth=4))
    assert trace["safety"]["tactical_refutation"]["status"] == "unknown"
    report = feedback(session)
    assert not report["proven_refutations"] and not report["unavoidable_mate"]
    assert "g2g3" in report["commit_ready_moves"]
    decision = decision_from(session, commit("g2g3", ["g2g3", "f7d5", "h1g1"]), board, "test")
    assert decision["chosen_move"] == "g2g3" and decision["safety"]["tactical_refutation"]["status"] == "unknown"
    assert decision["nodes_used"] <= 6000 and board.fen() == FEN


@pytest.mark.parametrize("evaluator,g3_score", [("king-safety-mobility-pawns-v1", -236),
                                               ("development-king-safety-v2", -199)])
def test_llm_reads_comparison_and_replans_after_proven_refutation(monkeypatch, patched_app_source, evaluator, g3_score):
    monkeypatch.setattr("assignment.agent.chess_policy_harness.EVALUATION", evaluator)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    with TestClient(create_app(policy="v6", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = chess.Board(FEN)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, fixed_chess_policy="v6")
        requests = []
        agent.client = fake_client([
            analysis(nodes=3000, depth=4), analysis(["f4d4"], nodes=2000, depth=4),
            analysis(["g2g3"], nodes=1000, depth=4), commit("g2g3", ["g2g3", "f7d5", "h1g1"]),
            commit("f4d4", ["f4d4", "g6f6", "f1g3"]),
        ], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"] and len(requests) == 5
        report = json.loads(requests[3]["messages"][-1]["content"])["feedback"]
        assert not report["replan_events"] and not report["replan_required"]
        pair = report["candidate_comparisons"][0]
        assert pair["moves"] == ["f4d4", "g2g3"] and pair["depth"] == 2 and pair["depth_limited"]
        assert pair["evidence"]["f4d4"]["score"] == -194
        assert pair["evidence"]["g2g3"]["score"] == g3_score
        assert report["strongest_exact_evidence"]["f4d4"]["depth"] == 4
        assert report["commit_ready_moves"] == ["f4d4"]
        repair = json.loads(requests[4]["messages"][-1]["content"])["feedback"]
        assert "f7d5 mate in 2" in repair["error"]
        assert agent.planner_records[3]["status"] == "rejected"
        decision = agent.chess_decisions[0]
        assert decision["chosen_move"] == decision["llm_action"]["move"] == "f4d4"
        assert decision["candidate_comparisons"] == report["candidate_comparisons"]
        assert len(agent.last_state["history"]) == 2
        assert decision["nodes_used"] <= 6000
        for trace in decision["analysis_trace"]:
            safety = trace["safety"] or {}
            assert trace["evidence"]["nodes_used"] + safety.get("nodes", 0) <= trace["action"]["max_nodes"]
            assert safety.get("tactical_refutation", {}).get("nodes", 0) <= TACTICAL_PROBE_NODES


def test_candidate_comparisons_are_replayed_and_tampering_is_detected(monkeypatch, tmp_path, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    with TestClient(create_app(policy="v6", max_nodes=6000, search_depth=4)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client, auto_stop_environment=False, fixed_chess_policy="v6")
        agent.client = fake_client([analysis(nodes=1000, depth=2), analysis(["e2e4"], nodes=1000, depth=4),
                                   analysis(["d2d4"], nodes=1000, depth=4), commit("d2d4")], [])
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"]
        save_fixed_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["tactical_probe_max_nodes"] == TACTICAL_PROBE_NODES
        from scripts.replay_v3_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["validated"]
        path = tmp_path/"decisions.jsonl"
        original = path.read_text()
        for key in ("candidate_comparisons", "proven_refutations", "unavoidable_mate"):
            decision = json.loads(original)
            if key == "candidate_comparisons":
                pair = decision[key][0]
                pair["evidence"][pair["moves"][0]]["score"] += 1
            elif key == "proven_refutations":
                decision[key] = {"d2d4": {"mate_in": 2}}
            else:
                decision[key] = True
            path.write_text(json.dumps(decision)+"\n")
            with pytest.raises(ValueError, match=key):
                verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
