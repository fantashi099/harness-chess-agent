"""LLM decision authority, readonly search, caps, retries and auditable artifacts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import chess
from fastapi.testclient import TestClient

from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_policy import position_id
from assignment.agent.chess_policy_harness import MAX_LLM_CALLS, mate_safety, validate_action, prepare, analyze, decision_from
from assignment.chess_artifacts import save_chess_run
from assignment.fixed_chess_server import create_app
from experiments.chess.test_chess_transactions import patched_app_source

REPO = Path(__file__).resolve().parents[2]


def analysis(candidates=None, nodes=100, depth=1):
    return {"action": "analyze", "objective": "Develop safely", "rationale": "Compare a central pawn opening",
            "candidate_moves": candidates or [], "max_nodes": nodes, "max_depth": depth}


def commit(move, continuation=None):
    if continuation is None:
        continuation = [move]
        board = chess.Board()
        if chess.Move.from_uci(move) in board.legal_moves:
            board.push_uci(move)
            for _ in range(2):
                if board.outcome(claim_draw=True):
                    break
                reply = next(iter(board.legal_moves))
                continuation.append(reply.uci())
                board.push(reply)
    return {"action": "commit", "objective": "Develop safely", "rationale": "Choose the analyzed queen-pawn opening",
            "move": move, "continuation": continuation}


def fake_client(actions, requests, *, finish_reasons=None):
    def create(**kwargs):
        requests.append(kwargs)
        action = actions[len(requests)-1]
        finish_reason = finish_reasons[len(requests)-1] if finish_reasons else "stop"
        raw = {"id": f"fake-{len(requests)}", "model": "fake", "usage": {
            "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps(action)}}]}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
            message=SimpleNamespace(content=json.dumps(action)))], model_dump=lambda **_: raw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **_: client
    return client


@pytest.mark.parametrize("candidates", [["e1d1", "e1e2", "e1f1"], ["e1f1", "e1e2", "e1d1"]])
def test_r7_game2_full_legal_scope_is_broad_search(monkeypatch, patched_app_source, candidates):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    board = chess.Board()
    for move in "e2e4 b8c6 d2d4 c6b8 d1h5 c7c6 c1f4 d8b6 f1c4 e7e6 f4b8 b6b2 c4e6 f8b4 c2c3 b4c3 b1c3 b2c3".split():
        board.push_uci(move)
    assert board.fen() == "rBb1k1nr/pp1p1ppp/2p1B3/7Q/3PP3/2q5/P4PPP/R3K1NR w KQkq - 0 10"
    initial = analysis(candidates, nodes=3000, depth=4)
    baseline = analyze(board, "offline", prepare(board, "offline", 6000, 4), analysis(nodes=3000, depth=4))
    for bad in [candidates[:2], [candidates[0]]*2, ["e1g1"]]:
        session = prepare(board, "offline", 6000, 4)
        with pytest.raises(ValueError):
            analyze(board, "offline", session, analysis(bad, nodes=3000, depth=4))
        assert session["nodes_used"] == 0 and not session["analyses"]
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = board
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([initial, analysis(["e1f1"], nodes=1000, depth=4),
                                    analysis(["e1d1"], nodes=1000, depth=4), commit("e1f1", ["e1f1", "d7d6", "d4d5"])], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"] and len(requests) == 4
        assert all(r["status"] == "accepted" for r in agent.planner_records)
        decision = agent.chess_decisions[0]
        broad = decision["analysis_trace"][0]
        assert broad["action"] == initial and not broad["verification"]
        for key in ["primary_score", "chosen_move", "completed_depth", "nodes_used", "top_candidates"]:
            assert broad["evidence"][key] == baseline["evidence"][key]
        assert decision["nodes_used"] <= 6000 and decision["chosen_move"] == "e1f1"
        assert decision["first_search_suggestion"] == "e1e2"  # The LLM still chooses the final move.
        assert json.loads(requests[0]["messages"][-1]["content"])["feedback"]["allowed_actions"] == ["analyze"]


def test_r6_turn47_truncated_response_gets_more_output_budget(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    fen = "8/4kp2/2B5/4P1r1/8/Pb6/1P5p/K3R3 w - - 2 47"
    board = chess.Board()
    for move in """e2e4 e7e6 d2d4 a7a6 b1c3 f8e7 g1f3 c7c5 f3e5 e7f6 c1f4 c5d4 d1d4 b8c6
        d4c5 c6e5 f4e5 b7b6 c5d6 f6e5 d6e5 g8f6 e1c1 h7h6 c1b1 e8g8 d1d4 d8e7 c3d5 f6d5
        d4d5 f8e8 d5d6 b6b5 f1e2 c8b7 h1d1 e7h4 g2g3 h4f6 e5f6 g7f6 e2f3 f6f5 d6d7 b7c6
        d7d4 g8f8 b1a1 f8e7 g3g4 f5g4 f3g4 e8b8 g4h5 b8c8 d1g1 e6e5 d4b4 c8d8 g1g7 c6e8
        g7g8 d8d2 f2f4 e5f4 e4e5 d2c2 a2a3 e8c6 g8a8 c6a8 b4f4 a8d5 f4f1 c2h2 h5f3 d5e6
        f3b7 h2h5 f1e1 h5g5 b7a6 h6h5 a6b5 h5h4 b5c6 h4h3 c6b5 h3h2 b5c6 e6b3""".split():
        board.push_uci(move)
    assert board.fen() == fen
    continuation_board = board.copy(stack=True)
    line = ["c6g2"]
    continuation_board.push_uci(line[0])
    for _ in range(2):
        move = next(iter(continuation_board.legal_moves))
        line.append(move.uci())
        continuation_board.push(move)
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = board
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([
            {}, analysis(nodes=3000, depth=4), analysis(["c6g2"], nodes=2000, depth=4),
            commit("c6g2", line), analysis(["c6b5"], nodes=1000, depth=4), commit("c6g2", line),
        ], requests, finish_reasons=["stop", "stop", "stop", "length", "stop", "stop"])
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"] and len(requests) == 6
        truncated = agent.planner_records[3]
        assert truncated["status"] == "rejected" and "finish_reason=length" in truncated["error"]
        assert "action" not in truncated
        assert [r["max_completion_tokens"] for r in requests] == [4096]*4+[8192, 8192]
        decision = agent.chess_decisions[0]
        assert decision["chosen_move"] == decision["llm_action"]["move"] == "c6g2"
        assert decision["nodes_used"] <= 6000 and len(decision["analysis_trace"]) == 3


def test_last_two_calls_allow_only_explicit_llm_commit(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    with TestClient(create_app(policy="harness", max_nodes=2000, search_depth=4)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([
            analysis(nodes=100), analysis(["d2d4"], nodes=1000, depth=2), {}, {}, {}, {},
            analysis(["e2e4"], nodes=100), commit("d2d4"),
        ], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"] and len(requests) == MAX_LLM_CALLS
        final_phase = json.loads(requests[6]["messages"][-1]["content"])["feedback"]
        assert final_phase["remaining_nodes"] > 0 and final_phase["remaining_analyses"] > 0
        assert final_phase["remaining_llm_calls"] == 2 and final_phase["allowed_actions"] == ["commit"]
        assert "Commit phase" in agent.planner_records[6]["error"]
        assert [r["max_completion_tokens"] for r in requests[-2:]] == [4096, 4096]
        assert len(agent.chess_decisions[0]["analysis_trace"]) == 2
        assert agent.chess_decisions[0]["llm_action"]["move"] == "d2d4"


def test_repeated_truncation_caps_output_and_never_refills_calls(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    with TestClient(create_app(policy="harness")) as client:
        agent = ChessAgent(object(), model="fake", http_client=client,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([analysis()]*MAX_LLM_CALLS, requests, finish_reasons=["length"]*MAX_LLM_CALLS)
        call = [{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}]
        output = agent.execute_tool_calls(call)[0]["content"]
        assert "LLM call allowance exhausted" in output and "max_completion_tokens=8192" in output
        assert [r["max_completion_tokens"] for r in requests] == [4096]+[8192]*7
        assert all(r["status"] == "rejected" and "action" not in r for r in agent.planner_records)
        assert not agent.last_state["history"] and not agent.chess_decisions
        assert "<chess_error>" in agent.execute_tool_calls(call)[0]["content"]
        assert len(requests) == MAX_LLM_CALLS


@pytest.mark.parametrize("lost_response", [None, "analyze", "verify", "commit"])
def test_llm_can_replan_and_choose_a_different_move(monkeypatch, tmp_path, patched_app_source, lost_response):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    with TestClient(create_app(policy="harness", max_nodes=2000, search_depth=4)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client, auto_stop_environment=False, chess_policy="harness")
        root = position_id(agent.last_state)
        original = client.app.state.chess_game._board.fen()
        requests = []
        agent.client = fake_client([analysis(nodes=100), analysis(["d2d4"], nodes=1000, depth=2), commit("d2d4")], requests)
        from assignment.agent import chess_harness
        real_request = chess_harness._request_state
        dropped = []
        def request(*args, **kwargs):
            output = real_request(*args, **kwargs)
            action = kwargs.get("json", {}).get("action", {})
            matches = (action.get("action") == lost_response
                       or (lost_response == "verify" and len(action.get("candidate_moves", [])) == 1))
            if lost_response and matches and not dropped:
                dropped.append(True)
                raise TimeoutError("Response lost after execution")
            return output
        monkeypatch.setattr(chess_harness, "_request_state", request)
        output = agent.execute_tool_calls([{"id":"act", "function":{"name":"select_and_play", "arguments":"{}"}}])
        assert "<chess_error>" not in output[0]["content"]
        assert len(requests) == 3 and len(agent.last_state["history"]) == 2
        assert len(dropped) == bool(lost_response)
        decision = agent.chess_decisions[0]
        assert decision["first_search_suggestion"] != "d2d4" and decision["chosen_move"] == "d2d4"
        assert decision["llm_changed_first_suggestion"] and decision["reason"] == "explicit_llm_commit"
        assert decision["nodes_used"] <= 2000 and len(decision["analysis_trace"]) == 2
        assert decision["safety"]["complete"] and decision["safety"]["mating_reply"] is None
        assert decision["score_bound"] == "exact"
        for request in requests:
            assert json.loads(request["messages"][-1]["content"])["context"]["fen"] == original
        repeated = client.post("/api/harness-commit", json={"expected_position_id": root, "action": commit("d2d4")}).json()
        assert repeated["replayed"] and len(repeated["state"]["history"]) == 2
        before = len(agent.planner_records)
        from assignment.agent.chess_harness import harness_select_and_play
        agent.last_state = dict(agent.last_state, fen=decision["root_fen"], history=[])
        assert json.loads(harness_select_and_play(agent, "{}"))["replayed"]
        assert len(agent.planner_records) == before
        agent.last_state = repeated["state"]
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["decision_authority"] == "llm" and manifest["llm_token_usage"]["total_tokens"] == 90
        assert manifest["replanned_turns"] == 0 and manifest["verification_analyses"] == 1
        assert manifest["comparison_turns"] == 0
        from scripts.replay_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["all_commits_match_llm_response"]
        saved = json.loads((tmp_path/"harness-trace.jsonl").read_text().splitlines()[-1])
        saved["response"]["choices"][0]["message"]["content"] = json.dumps(commit("e2e4"))
        rows = (tmp_path/"harness-trace.jsonl").read_text().splitlines()
        rows[-1] = json.dumps(saved)
        (tmp_path/"harness-trace.jsonl").write_text("\n".join(rows)+"\n")
        with pytest.raises(ValueError):
            verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
        client.post("/api/reset").raise_for_status()
        assert client.post("/api/harness-context", json={"expected_position_id": root}).status_code == 409


def test_executor_is_readonly_and_has_no_automatic_fallback(patched_app_source):
    with TestClient(create_app(policy="harness", max_nodes=2)) as client:
        state = client.get("/api/state").json()
        request = {"expected_position_id": position_id(state)}
        assert client.post("/api/select-and-play", json=request).status_code == 400
        assert client.post("/api/harness-commit", json=dict(request, action=commit("e2e4"))).status_code == 400
        first = client.post("/api/harness-analyze", json=dict(request, action=analysis(nodes=1))).json()
        replayed = client.post("/api/harness-analyze", json=dict(request, action=analysis(nodes=1))).json()
        assert first == replayed and first["remaining_nodes"] == 1
        assert client.get("/api/state").json()["fen"] == state["fen"]
        assert client.post("/api/harness-analyze", json=dict(request, action=analysis(["d2d4"], nodes=2))).status_code == 400
        context = client.post("/api/harness-context", json=request).json()["context"]
        for bad in (dict(analysis(), max_nodes=True), dict(analysis(), candidate_moves=[{}]),
                    dict(commit("e2e4"), score=100000), commit("a1a8")):
            with pytest.raises(ValueError):
                validate_action(bad, context)


def test_invalid_llm_outputs_never_play_a_code_fallback(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    with TestClient(create_app(policy="harness")) as client:
        agent = ChessAgent(object(), model="fake", http_client=client, auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([commit("a1a8")]*MAX_LLM_CALLS, requests)
        call = [{"id":"act", "function":{"name":"select_and_play", "arguments":"{}"}}]
        error = agent.execute_tool_calls(call)[0]["content"]
        assert "LLM call allowance exhausted" in error and "Commit requires a legal UCI move" in error
        assert len(requests) == MAX_LLM_CALLS and not agent.last_state["history"]
        assert "<chess_error>" in agent.execute_tool_calls(call)[0]["content"]
        assert len(requests) == MAX_LLM_CALLS and not client.get("/api/state").json()["history"]


def test_evidence2400_budget_dead_end_is_repaired_without_automatic_selection(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    fen = "r2qr3/p3bkpp/2pp1n2/4p3/4PBb1/5Q2/PPP2PPP/RN1R2K1 w - - 4 16"
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = chess.Board(fen)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([
            analysis(nodes=3000, depth=4), analysis(["f3e3"], nodes=2000, depth=4),
            analysis(["f3g4"], nodes=1000, depth=4),
            analysis(["f3b3"], nodes=1000, depth=4),
            commit("f3b3", ["f3b3", "f6d5", "f4g3"]),
        ], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"]
        assert [r["status"] for r in agent.planner_records] == ["accepted", "accepted", "rejected", "accepted", "accepted"]
        blocked = json.loads(requests[3]["messages"][-1]["content"])["feedback"]
        assert 1000 <= blocked["remaining_nodes"] < 2000
        assert blocked["verification_nodes_available"] == blocked["remaining_nodes"] - 1000
        assert blocked["reserved_verification"] == {"move": "f3b3", "max_nodes": blocked["remaining_nodes"]}
        assert blocked["commit_ready_moves"] == ["f3e3"]
        assert blocked["remaining_llm_calls"] == MAX_LLM_CALLS-3
        from assignment.agent.chess_policy_harness import REVISION
        if REVISION >= 16:
            assert "Defensive comparison phase" in blocked["error"]
            assert blocked["action_contract"]["comparison_moves_allowed"] == ["f3b3"]
        else:
            assert "analysis nodes available" in blocked["error"]
            assert blocked["action_contract"]["node_limits"]["reserved_candidate"] == blocked["reserved_verification"]
        final = json.loads(requests[4]["messages"][-1]["content"])["feedback"]
        assert final["reserved_verification"] is None and "f3b3" in final["commit_ready_moves"]
        decision = agent.chess_decisions[0]
        assert decision["chosen_move"] == decision["llm_action"]["move"] == "f3b3"
        assert decision["nodes_used"] <= 6000 and len(decision["analysis_trace"]) == 3
        assert decision["safety"]["complete"] and decision["completed_depth"] >= 2
        assert decision["root_fen"] == fen and len(agent.last_state["history"]) == 2


@pytest.mark.parametrize("chosen,line,score,depth", [
    ("d5d6", ["d5d6", "g3h2", "h5f7"], -564, 3),
    ("h5g4", ["h5g4", "c4e3", "f1f4"], -603, 4),
])
@pytest.mark.parametrize("evaluator,drop,deep_score", [("king-safety-mobility-pawns-v1", 331, -603),
                                                      ("development-king-safety-v2", 329, -604)])
def test_r4_turn40_compares_shared_iterations_and_keeps_deeper_evidence(
        monkeypatch, patched_app_source, chosen, line, score, depth, evaluator, drop, deep_score):
    monkeypatch.setattr("assignment.agent.chess_policy_harness.EVALUATION", evaluator)
    if chosen == "h5g4":
        score = deep_score
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    fen = "b2r4/5p2/5k2/p2P1p1B/2n5/2P3p1/P6P/K4R2 w - - 0 40"
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = chess.Board(fen)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([
            analysis(nodes=3000, depth=4), {}, analysis(["h5g4"], nodes=2000, depth=4),
            analysis(["d5d6"], nodes=1423, depth=4), commit(chosen, line),
        ], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"] and len(requests) == 5
        assert agent.planner_records[1]["status"] == "rejected"
        pending = json.loads(requests[3]["messages"][-1]["content"])["feedback"]
        assert pending["replan_required"] and not pending["commit_ready_moves"]
        final = json.loads(requests[4]["messages"][-1]["content"])["feedback"]
        assert 0 <= final["remaining_nodes"] < 1000 and not final["replan_required"]
        assert final["remaining_llm_calls"] == 4 and "commit" in final["allowed_actions"]
        assert requests[4]["max_completion_tokens"] == 4096
        assert final["commit_ready_moves"] == ["d5d6", "h5g4"]
        event = final["replan_events"][0]
        assert event["drop"] == drop and event["preferred_comparison_depth"] == 4
        assert event["required_comparison_depth"] == 2 and event["compared_moves"] == ["d5d6"]
        pair = event["comparisons"][0]
        assert pair["depth"] == 3 and pair["depth_limited"]
        decision = agent.chess_decisions[0]
        traces = decision["analysis_trace"]
        for key, trace in [("selected", traces[1]), ("alternative", traces[2])]:
            iteration = next(row for row in trace["evidence"]["iterations"] if row["depth"] == 3 and row["completed"])
            assert pair[key] == {"score": iteration["primary_score"], "pv": iteration["pv"], "action_id": trace["action_id"]}
        assert final["strongest_exact_evidence"]["h5g4"]["score"] == deep_score
        assert final["strongest_exact_evidence"]["h5g4"]["depth"] == 4
        assert decision["chosen_move"] == decision["llm_action"]["move"] == chosen
        assert decision["primary_score"] == score and decision["completed_depth"] == depth
        assert decision["nodes_used"] <= 6000 and len(agent.last_state["history"]) == 2


def test_exhausted_replan_allows_only_explicit_llm_risk_commit(monkeypatch, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    fen = "8/2p1bk1p/3pbp1B/6p1/1P2P3/5P2/2qN1NPP/3R2K1 w - - 0 27"
    with TestClient(create_app(policy="harness", max_nodes=6000, search_depth=4)) as client:
        client.app.state.chess_game._board = chess.Board(fen)
        agent = ChessAgent(object(), model="fake", http_client=client, reset_game=False,
                           auto_stop_environment=False, chess_policy="harness")
        requests = []
        shallow = analysis(["g1h1"], nodes=80, depth=2)
        board = chess.Board(fen)
        line = ["g1h1"]
        board.push_uci(line[0])
        for _ in range(2):
            if board.outcome(claim_draw=True):
                break
            move = next(iter(board.legal_moves))
            line.append(move.uci())
            board.push(move)
        agent.client = fake_client([analysis(nodes=2000, depth=2), analysis(["b4b5"], nodes=1500, depth=2),
                                   shallow, dict(shallow, objective="Repeat insufficient verification"),
                                   dict(commit("g1h1", line), accept_risk=True)], requests)
        call = [{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}]
        output = agent.execute_tool_calls(call)
        assert "<chess_error>" not in output[0]["content"]
        assert len(requests) == 5 and len(agent.chess_decisions) == 1
        report = json.loads(requests[-1]["messages"][-1]["content"])["feedback"]
        assert not report["commit_ready_moves"] and "g1h1" in report["risk_commit_moves"]
        assert report["allowed_actions"] == ["commit"]
        assert agent.chess_decisions[0]["risk_acceptance"]["unresolved_replan"]
        assert agent.chess_decisions[0]["llm_action"]["accept_risk"] is True


ELO2400_LAST_FEN = "3k4/1pr4q/2p1p2r/p2pPb2/Bb6/1P2P3/PRP3P1/2Q2RK1 w - - 3 32"


def test_elo2400_mate_regression_requires_llm_repair(patched_app_source):
    with TestClient(create_app(policy="harness", max_nodes=6000)) as client:
        client.app.state.chess_game._board = chess.Board(ELO2400_LAST_FEN)
        state = client.get("/api/state").json()
        request = {"expected_position_id": position_id(state)}
        def analyze(action):
            response = client.post("/api/harness-analyze", json=dict(request, action=action))
            response.raise_for_status()
            return response.json()
        # An LLM shortlist cannot silently omit defensive alternatives at the first analysis.
        assert client.post("/api/harness-analyze", json=dict(request, action=analysis(["f1f2"]))).status_code == 400
        broad = analyze(analysis(nodes=2500, depth=2))
        assert broad["analyses"][0]["completed_depth"] >= 1
        upper = next(r["move"] for r in broad["analyses"][0]["top_candidates"] if r["bound"] == "upper")
        rejected = client.post("/api/harness-commit", json=dict(request, action=commit(upper)))
        assert rejected.status_code == 400 and "Before commit" in rejected.json()["detail"]
        unsafe = analyze(analysis(["f1f2"], nodes=1000, depth=2))
        safety = unsafe["analyses"][-1]["safety"]
        assert safety["complete"] and safety["mating_reply"] == "h6h1" and safety["safe_alternative"]
        assert unsafe["analyses"][-1]["primary_score"] == -99998
        assert not unsafe["analyses"][-1]["commit_ready"]
        rejected = client.post("/api/harness-commit", json=dict(request, action=commit("f1f2")))
        assert rejected.status_code == 400 and "h6h1 mate" in rejected.json()["detail"]
        assert client.get("/api/state").json() == state
        # Repeating an unsafe request cannot refill the budget or mutate the live board.
        assert analyze(analysis(["f1f2"], nodes=1000, depth=2)) == unsafe
        repaired = analyze(analysis(["c2c3"], nodes=1000, depth=2))
        assert repaired["analyses"][-1]["commit_ready"]
        played = client.post("/api/harness-commit", json=dict(request, action=commit("c2c3", ["c2c3", "b4a3", "f1f5"])))
        played.raise_for_status()
        decision = played.json()["decision"]
        assert decision["chosen_move"] == "c2c3" and decision["safety"]["mating_reply"] is None
        assert decision["nodes_used"] <= 6000
        assert decision["nodes_used"] == sum(t["evidence"]["nodes_used"] + (t["safety"] or {}).get("nodes", 0)
                                               for t in decision["analysis_trace"])


def test_verification_budget_is_reserved_and_incomplete_checks_fail_closed(patched_app_source):
    with TestClient(create_app(policy="harness", max_nodes=6000)) as client:
        state = client.get("/api/state").json()
        request = {"expected_position_id": position_id(state)}
        denied = client.post("/api/harness-analyze", json=dict(request, action=analysis(nodes=6000)))
        assert denied.status_code == 400 and "reserve 2000" in denied.json()["detail"]
        client.post("/api/harness-analyze", json=dict(request, action=analysis(nodes=100))).raise_for_status()
        client.post("/api/harness-analyze", json=dict(request, action=analysis(["d2d4"], nodes=1))).raise_for_status()
        denied = client.post("/api/harness-commit", json=dict(request, action=commit("d2d4")))
        assert denied.status_code == 400 and not client.get("/api/state").json()["history"]
    board = chess.Board(ELO2400_LAST_FEN)
    before = board.fen(), list(board.move_stack)
    assert not mate_safety(board, "f1f2", 1)["complete"]
    assert before == (board.fen(), board.move_stack)


def test_forced_loss_is_reported_without_inventing_a_safe_move():
    board = chess.Board("3k4/1pr4q/2p1p2r/p2pPb2/Bb6/1P2P3/PRP2PP1/2Q2R1K w - - 3 32")
    assert board.is_valid() and [m.uci() for m in board.legal_moves] == ["h1g1"]
    safety = mate_safety(board, "h1g1", 1000)
    assert safety["complete"] and safety["mating_reply"] == "h6h1"
    assert safety["forced_mate"] and safety["safe_alternative"] is None


def test_verified_mate_in_one_can_commit_after_early_search_stop():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    session = prepare(board, "test-position", 6000, 4)
    analyze(board, "test", session, analysis(nodes=3000, depth=2))
    verified = analyze(board, "test", session, analysis(["f7g7"], nodes=100, depth=4))
    assert verified["evidence"]["completed_depth"] == 1
    decision = decision_from(session, commit("f7g7"), board, "test")
    assert decision["primary_score"] == 99999 and decision["safety"]["complete"]


def test_llm_receives_mate_feedback_and_actual_opponent(monkeypatch, tmp_path, patched_app_source):
    from unittest.mock import Mock
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    engine = Mock(id={"name": "Stockfish test"})
    engine.play.side_effect = [SimpleNamespace(move=chess.Move.from_uci(move)) for move in ("e7e5", "b8c6")]
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", Mock(return_value=engine))
    with TestClient(create_app(policy="harness", max_nodes=6000, opponent="stockfish", stockfish_elo=2400)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client, auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([analysis(nodes=1000, depth=2), analysis(["f2f3"], nodes=1000, depth=2), commit("f2f3"),
                                   analysis(nodes=2000, depth=2), analysis(["g2g4"], nodes=1000, depth=2),
                                   commit("g2g4"), analysis(["d2d4"], nodes=1000, depth=2), commit("d2d4")], requests)
        for turn in range(2):
            output = agent.execute_tool_calls([{"id": str(turn), "function": {"name": "select_and_play", "arguments": "{}"}}])
            assert "<chess_error>" not in output[0]["content"]
        assert agent.chess_decisions[-1]["chosen_move"] == "d2d4" and len(agent.last_state["history"]) == 4
        assert [r["status"] for r in agent.planner_records] == ["accepted"]*5 + ["rejected", "accepted", "accepted"]
        repair = json.loads(requests[6]["messages"][-1]["content"])
        assert "d8h4 mate" in repair["feedback"]["error"]
        assert repair["context"]["opponent"]["elo"] == 2400
        assert "Elo 1500" not in requests[0]["messages"][0]["content"]
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        from scripts.replay_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["all_commits_match_llm_response"]
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["verification_reserve_nodes"] == 2000 and manifest["max_analyses_per_turn"] == 4


def test_illegal_future_continuation_is_audited_without_blocking_llm_commit(monkeypatch, tmp_path, patched_app_source):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    with TestClient(create_app(policy="harness", max_nodes=6000)) as client:
        agent = ChessAgent(object(), model="fake", http_client=client, auto_stop_environment=False, chess_policy="harness")
        requests = []
        agent.client = fake_client([analysis(nodes=1000, depth=2), analysis(["d2d4"], nodes=1000, depth=2),
                                   commit("d2d4", ["d2d4", "d7d5", "f4g5"]),
                                   commit("d2d4", ["d2d4", "d7d5", "g1f3"])], requests)
        output = agent.execute_tool_calls([{"id": "act", "function": {"name": "select_and_play", "arguments": "{}"}}])
        assert "<chess_error>" not in output[0]["content"]
        assert len(agent.chess_decisions[0]["analysis_trace"]) == 2 and len(agent.last_state["history"]) == 2
        assert len(requests) == 3
        checks = agent.chess_decisions[0]["continuation_checks"]
        assert not checks["valid"] and "Illegal continuation at ply 3" in checks["error"]
        assert [p["uci"] for p in checks["plies"]] == ["d2d4", "d7d5"]
        save_chess_run(agent, tmp_path/"result.json", REPO/"artifacts/fix.patch", "step_limit")
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["repair_turns"] == 0 and manifest["replanned_turns"] == 0
        assert manifest["verification_analyses"] == 1
        from scripts.replay_chess_game import verify
        assert verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")["validated"]
        decisions_path = tmp_path/"decisions.jsonl"
        decision = json.loads(decisions_path.read_text())
        decision["continuation_checks"]["plies"][0]["change_from_root"] = 900
        decisions_path.write_text(json.dumps(decision)+"\n")
        with pytest.raises(ValueError, match="continuation_checks"):
            verify(tmp_path/"result.json", REPO/"artifacts/fix.patch")
