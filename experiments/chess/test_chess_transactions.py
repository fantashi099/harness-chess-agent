"""Extension regressions; the original assignment tests remain untouched."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys

import chess
from fastapi.testclient import TestClient
import pytest

from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_policy import position_id
from assignment.chess_server import simulate_position
from assignment.fixed_chess_server import create_app
from assignment.chess_artifacts import save_chess_run

REPO = Path(__file__).resolve().parents[2]
REGRESSIONS = [
    ("rnbq3r/1pk1b1pp/2p1p3/p7/P1BP1RP1/1P5P/4N1K1/R1NQ4 w - - 0 40",
     "c4e6", "f4e4", -210),
    ("1n1q3r/rpk3pp/4Q3/p1N5/P4RP1/7P/4N1K1/R7 w - - 1 48",
     "c5d7", "e6e5", 232),
    ("3qr3/rp1n2Qp/1k6/p7/P4RP1/7P/4N1K1/R7 w - - 1 51",
     "g7h7", "g7d4", 12),
    ("3q4/rp1n3Q/1k6/p3r3/P4RP1/7P/8/R4K2 w - - 2 53",
     "h7d7", "a1b1", -232),
]


@pytest.fixture(scope="session")
def patched_app_source(tmp_path_factory):
    clone = tmp_path_factory.mktemp("patched-chess") / "app"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(REPO / "chess_app"), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "apply", str(REPO / "artifacts/fix.patch")],
                   check=True, capture_output=True)
    source = str(clone / "src")
    sys.path.insert(0, source)
    yield clone
    sys.path.remove(source)


@pytest.fixture
def client(patched_app_source):
    with TestClient(create_app(policy="search", max_nodes=1000, search_depth=1)) as client:
        yield client


def request_for(client):
    return {"expected_position_id": position_id(client.get("/api/state").json())}


def test_duplicate_requests_are_idempotent(client):
    request = request_for(client)
    first = client.post("/api/select-and-play", json=request)
    again = client.post("/api/select-and-play", json=request)
    assert first.status_code == again.status_code == 200
    assert again.json()["replayed"]
    assert len(again.json()["state"]["history"]) == 2
    assert len(client.get("/api/policy-decisions").json()) == 1
    assert first.json()["decision"]["committed_move"] == first.json()["decision"]["chosen_move"]


def test_concurrent_duplicate_requests_commit_only_once(client):
    request = request_for(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda _: client.post("/api/select-and-play", json=request), range(2)))
    assert all(r.status_code == 200 for r in responses)
    assert len(client.get("/api/state").json()["history"]) == 2
    assert len(client.get("/api/policy-decisions").json()) == 1


def test_stale_state_and_move_override_are_rejected(client):
    stale_request = request_for(client)
    client.post("/api/move", json={"move": "d2d4"}).raise_for_status()
    assert client.post("/api/select-and-play", json=stale_request).status_code == 409
    assert client.post("/api/select-and-play",
                       json=dict(request_for(client), move="a2a3")).status_code == 422
    assert len(client.get("/api/state").json()["history"]) == 2


def test_reset_invalidates_even_an_identical_initial_fen(client):
    previous = request_for(client)
    client.post("/api/reset").raise_for_status()
    assert previous != request_for(client)
    assert client.post("/api/select-and-play", json=previous).status_code == 409
    assert client.get("/api/state").json()["history"] == []


def test_white_terminal_move_does_not_request_a_black_reply(client):
    game = client.app.state.chess_game
    game._board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = client.post("/api/select-and-play", json=request_for(client)).json()
    assert result["state"]["game_over"]
    assert result["state"]["status"] == "White wins"
    assert result["state"]["engine_move"] is None
    assert len(result["state"]["history"]) == 1


def tool_call(name, args=None, call_id="test-call"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def test_agent_cannot_bypass_fixed_policy_and_rejects_extra_calls(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    agent = ChessAgent(object(), model="test-model", http_client=client,
                       auto_stop_environment=False, chess_policy="search")
    assert {t["function"]["name"] for t in agent.tools} == {"select_and_play"}
    for name in ("play_move", "run_python", "simulate_move"):
        observations = agent.execute_tool_calls([tool_call(name, {"move": "a2a3"})])
        assert "unknown tool" in observations[0]["content"]
        assert agent.last_state["history"] == []
    override = agent.execute_tool_calls([tool_call("select_and_play", {"move": "a2a3"})])
    assert "<chess_error>" in override[0]["content"]
    assert agent.last_state["history"] == []
    observations = agent.execute_tool_calls([
        tool_call("select_and_play", call_id="one"),
        tool_call("select_and_play", call_id="two")])
    assert "chess_decision" in observations[0]["content"]
    assert "only one position-changing tool" in observations[1]["content"]
    assert len(agent.last_state["history"]) == 2
    assert len(agent.chess_decisions) == 1


def test_incomplete_run_recovers_server_decisions_and_saves_artifacts(client, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    agent = ChessAgent(object(), model="test-model", http_client=client,
                       auto_stop_environment=False, chess_policy="search")
    # Simulate a successful commit whose response never reached the agent.
    client.post("/api/select-and-play", json=request_for(client)).raise_for_status()
    assert agent.chess_decisions == []
    result_path = tmp_path / "result.json"
    save_chess_run(agent, result_path, REPO / "artifacts/fix.patch", "StepLimitError")
    result = json.loads(result_path.read_text())
    assert not result["completed"]
    assert result["stop_reason"] == "StepLimitError"
    assert len(result["history"]) == 2
    assert len((tmp_path / "decisions.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "game.pgn").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text())["decision_count"] == 1
