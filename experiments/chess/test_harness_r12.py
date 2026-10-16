"""Regressions from v6-r11-2000-WA2i35an; no provider/Modal calls."""
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_harness import action_contract
from assignment.agent.chess_policy_harness import (
    TACTICAL_PROBE_NODES, analyze, board_facts, checking_chain_refutation,
    decision_from, feedback, mate_safety, move_facts, prepare, validate_action,
)
from assignment.chess_artifacts import save_chess_run
from experiments.chess.test_agent_harness import analysis, commit, fake_client
from experiments.chess.test_chess_transactions import patched_app_source
from assignment.fixed_chess_server import create_app

REPO = Path(__file__).resolve().parents[2]
MATE_FEN = "N2Q4/p4pk1/4p3/2Kb4/3Pn3/1PP5/q6P/3R2NR w - - 4 30"


def verify_tree(board, tree):
    reply = tree.get("black_mate", tree.get("black_reply"))
    move = chess.Move.from_uci(reply)
    assert move in board.legal_moves and board.gives_check(move)
    board.push(move)
    try:
        if "black_mate" in tree:
            assert board.is_checkmate()
        else:
            assert {m.uci() for m in board.legal_moves} == {d["white_defense"] for d in tree["defenses"]}
            for row in tree["defenses"]:
                board.push_uci(row["white_defense"])
                try:
                    verify_tree(board, row)
                finally:
                    board.pop()
    finally:
        board.pop()


def test_game3_turn30_mate_in_three_certificate_blocks_normal_commit():
    board = chess.Board(MATE_FEN)
    session = prepare(board, "regression", 6000, 4)
    analyze(board, "regression", session, analysis(nodes=3000, depth=4))
    trace = analyze(board, "regression", session, analysis(["c5b4"], nodes=2000, depth=4))
    safety = trace["safety"]
    assert safety["complete"] and not safety["mating_reply"]
    assert safety["tactical_refutation"]["status"] == "not_found"
    proof = safety["checking_chain_refutation"]
    assert proof["status"] == "refuted" and proof["mate_in"] == 3 and proof["black_reply"] == "a2b3"
    assert safety["tactical_refutation"]["nodes"] + proof["nodes"] <= TACTICAL_PROBE_NODES
    board.push_uci("c5b4")
    verify_tree(board, proof)
    board.pop()
    report = feedback(session)
    assert report["proven_refutations"]["c5b4"]["mate_in"] == 3
    assert "c5b4" not in report["commit_ready_moves"] and not report["unavoidable_mate"]
    with pytest.raises(ValueError, match="a2b3 mate in 3"):
        decision_from(session, commit("c5b4", ["c5b4"]), board, "regression")
    assert board.fen() == MATE_FEN and not board.move_stack and session["nodes_used"] <= 6000


@pytest.mark.parametrize("cap", [0, 1, 10, 32])
def test_checking_chain_cutoff_never_publishes_partial_proof(cap):
    board = chess.Board(MATE_FEN)
    immediate = mate_safety(board, "c5b4", 1000)
    proof = checking_chain_refutation(board, "c5b4", immediate["checking_replies"], cap)
    assert proof["status"] == "unknown" and proof["nodes"] <= cap
    assert proof["black_reply"] is None and proof["mate_in"] is None and not proof["defenses"]
    assert board.fen() == MATE_FEN and not board.move_stack


def test_chain_requires_checks_and_preserves_draw_claims():
    board = chess.Board(MATE_FEN)
    proof = checking_chain_refutation(board, "c5b4", [], 0)
    assert proof["status"] == "not_found" and proof["nodes"] == 0
    drawn = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
    proof = checking_chain_refutation(drawn, "a1a2", ["h8g7"], 30)
    assert proof["status"] == "not_found" and not proof["defenses"]


def test_chain_certificate_covers_multiple_white_defenses():
    board = chess.Board("8/5bk1/6r1/p1P4p/P4Q2/2P5/4q1PP/5N1K w - - 9 43")
    immediate = mate_safety(board, "g2g3", 1000)
    proof = checking_chain_refutation(board, "g2g3", immediate["checking_replies"], 500)
    assert proof["status"] == "refuted"
    assert len(proof["defenses"]) > 1
    board.push_uci("g2g3")
    verify_tree(board, proof)
    board.pop()
    assert not board.move_stack


def test_board_grounding_corrects_actual_piece_color_and_defense_errors():
    board = chess.Board("8/1B1b4/3k3p/pp5P/3K4/P1P5/8/8 w - - 0 58")
    facts = board_facts(board)
    assert facts["pieces"]["b7"] == "B" and facts["pieces"]["d7"] == "b"
    assert facts["bishop_square_colors"]["b7"] == facts["bishop_square_colors"]["d7"]
    board = chess.Board("2kr3r/pp1q1ppp/2p5/8/1b6/2N5/PPbP1PPP/RKB3NR w - - 0 15")
    assert move_facts(board, "b1c2")["captured_piece"] == "b"
    board = chess.Board("r1b1k1nr/pppp2bp/2n2qp1/8/4PB2/8/PPP2PPP/RN1QKBNR w KQkq - 0 7")
    facts = move_facts(board, "b1c3")
    assert "b2" not in facts["attacks_after"] and "d5" in facts["attacks_after"]
    assert board.piece_at(chess.B1).symbol() == "N"


def test_en_passant_grounding_identifies_the_actual_capture_square():
    board = chess.Board()
    for uci in "e2e4 a7a6 e4e5 d7d5".split():
        board.push_uci(uci)
    facts = move_facts(board, "e5d6")
    assert facts["captured_piece"] == "p" and facts["captured_square"] == "d5"
    assert board.peek().uci() == "d7d5"


def test_initial_contract_example_has_complete_schema_and_correct_phase():
    session = prepare(chess.Board(), "root", 6000, 4)
    context = session["context"]
    initial = dict(feedback(session), allowed_actions=["analyze"])
    contract = action_contract(context, initial)
    assert contract["phase"] == "broad_search" and contract["example"]["candidate_moves"] == []
    assert validate_action(contract["example"], context) == contract["example"]
    assert contract["example"]["max_nodes"] == 3000 and contract["example"]["rationale"]


def test_game4_turn14_contract_surfaces_limited_depth_comparison():
    board = chess.Board("2kr3r/pp1q1ppp/2p5/5b2/1b2Q3/2N5/PP1P1PPP/RKB3NR w - - 1 14")
    session = prepare(board, "comparison", 6000, 4)
    for moves, nodes in [([], 3000), (["b2b3"], 2000), (["e4c2"], 1000)]:
        analyze(board, "comparison", session, analysis(moves, nodes=nodes, depth=4))
    report = dict(feedback(session), allowed_actions=["commit"])
    contract = action_contract(session["context"], report)
    assert contract["phase"] == "commit" and "comparison_caution" in contract
    assert report["candidate_comparisons"][0]["depth_limited"]
    assert report["analyses"][-1]["move_facts"]["piece"] == "Q"
    assert session["nodes_used"] <= 6000


@pytest.mark.parametrize("archived_recorder", [False, True])
def test_contract_grounding_commit_and_replay_provenance(monkeypatch, tmp_path, patched_app_source, archived_recorder):
    tmp_path = tmp_path / "game_1"
    tmp_path.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([analysis(nodes=3000, depth=4),
            analysis(["d2d4"], nodes=2000, depth=4), commit("d2d4")], requests)
        agent.client.base_url = SimpleNamespace(host="openrouter.ai")
        result = agent.execute_tool_calls([{"id": "turn", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in result[0]["content"]
        contexts = [json.loads(r["messages"][-1]["content"]) for r in requests]
        assert all(r["response_format"]["type"] == "json_schema" for r in requests)
        assert all(r["response_format"]["json_schema"]["strict"] for r in requests)
        assert all(r["extra_body"] == {"provider": {"require_parameters": True}} for r in requests)
        assert all(r["reasoning_effort"] == "none" for r in requests)
        assert contexts[0]["feedback"]["action_contract"]["phase"] == "broad_search"
        assert contexts[1]["feedback"]["action_contract"]["phase"] == "verify_or_replan"
        assert contexts[2]["feedback"]["analyses"][-1]["move_facts"]["piece"] == "P"
        assert agent.chess_decisions[0]["chosen_move"] == "d2d4"
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
    from scripts.replay_chess_game import verify
    if archived_recorder:
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        recorder = (REPO / "scripts/record_chess_game.py").read_bytes()
        del manifest["source_sha256"]["scripts/record_chess_game.py"]
        manifest["source_sha256"]["record_game.py"] = hashlib.sha256(recorder).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        archive = tmp_path.parent / "source-snapshot/record_game.py"
        archive.parent.mkdir()
        archive.write_bytes(recorder)
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["validated"]
        archive.write_bytes(b"tampered recorder")
        with pytest.raises(ValueError, match="Source changed without matching snapshot"):
            verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
        archive.write_bytes(recorder)
    assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["validated"]
    manifest_path = tmp_path / "manifest.json"
    original_manifest = manifest_path.read_text()
    manifest = json.loads(original_manifest)
    assert manifest["reasoning_effort"] == "none"
    assert manifest["response_format"] == "json_schema"
    manifest["reasoning_effort"] = "low"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reasoning effort"):
        verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
    manifest = json.loads(original_manifest)
    manifest["response_format"] = "json_object"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="response format"):
        verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
    manifest_path.write_text(original_manifest)
    path = tmp_path/"decisions.jsonl"
    original = path.read_text()
    decision = json.loads(original)
    decision["analysis_trace"][-1]["move_facts"]["piece"] = "Q"
    path.write_text(json.dumps(decision)+"\n")
    with pytest.raises(ValueError, match="move_facts"):
        verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
    path.write_text(original)
    path = tmp_path/"harness-trace.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    data = json.loads(records[0]["request"][-1]["content"])
    data["context"]["board_facts"]["pieces"]["e1"] = "Q"
    records[0]["request"][-1]["content"] = json.dumps(data)
    path.write_text("".join(json.dumps(row)+"\n" for row in records))
    decision = json.loads(original)
    decision["harness_records"] = records
    (tmp_path/"decisions.jsonl").write_text(json.dumps(decision)+"\n")
    with pytest.raises(ValueError, match="board_facts"):
        verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
