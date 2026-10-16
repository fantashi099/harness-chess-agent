"""The comparison changes orchestration, while search and evaluator stay matched."""
from unittest.mock import patch

import chess
import pytest
from fastapi.testclient import TestClient

from assignment.agent.chess_agent import ChessAgent
from assignment.agent import chess_policy_search as search
from assignment.agent.chess_policy_harness import EVALUATION
from assignment.agent.chess_policy import position_id
from assignment.chess_sandbox import ChessSandbox
from assignment.fixed_chess_server import create_app
from experiments.chess import evaluate_search_policy as runner
from experiments.chess.test_chess_transactions import patched_app_source


@pytest.mark.parametrize("legacy", ["v2", "v3", "v4", "v5", "v6"])
def test_only_named_comparison_modes_are_accepted(legacy, tmp_path):
    with pytest.raises(ValueError):
        create_app(policy=legacy)
    with pytest.raises(ValueError):
        ChessAgent(object(), chess_policy=legacy)
    with pytest.raises(ValueError):
        ChessSandbox(chess_policy=legacy)
    with pytest.raises(ValueError):
        runner.run(tmp_path, 1, 6000, 4, policy=legacy)
    assert not (tmp_path / "experiment-config.json").exists()


@pytest.mark.parametrize("mode", ["search", "harness"])
def test_modes_use_the_same_search_and_evaluator(mode, patched_app_source):
    board = chess.Board("6k1/7p/8/8/8/8/P7/1K6 w - - 0 1")
    expected = search.select_move(board, "control", 6000, 4, evaluation=EVALUATION)
    with TestClient(create_app(policy=mode, max_nodes=6000, search_depth=4)) as client:
        game = client.app.state.chess_game
        game._board = board.copy(stack=True)
        config = client.get("/api/policy-config").json()
        assert config["evaluation"] == EVALUATION
        assert config["max_nodes"] == 6000 and config["search_config"] == {"max_depth": 4}
        state = client.get("/api/state").json()
        request = {"expected_position_id": position_id(state)}
        with patch.object(search, "select_move", wraps=search.select_move) as executor:
            if mode == "search":
                result = client.post("/api/select-and-play", json=request)
                result.raise_for_status()
                decision = result.json()["decision"]
                for key in ("chosen_move", "primary_score", "completed_depth", "nodes_used"):
                    assert decision[key] == expected[key]
            else:
                assert client.post("/api/select-and-play", json=request).status_code == 400
                action = {"action": "analyze", "objective": "Compare all moves",
                          "rationale": "Use the matched evaluator", "candidate_moves": [],
                          "max_nodes": 3000, "max_depth": 4}
                result = client.post("/api/harness-analyze", json=dict(request, action=action))
                result.raise_for_status()
                assert client.get("/api/state").json()["history"] == []
            assert executor.call_count == 1
            assert executor.call_args.kwargs["evaluation"] == EVALUATION


@pytest.mark.parametrize("flags,expected", [([], "harness"),
    (["--chess-policy", "search"], "search"), (["--programmatic-tools"], None)])
def test_cli_defaults_to_harness_and_preserves_explicit_modes(flags, expected, tmp_path, monkeypatch):
    from unittest.mock import Mock
    from assignment import cli

    patch_path = tmp_path / "fix.patch"
    patch_path.write_text("test patch")
    monkeypatch.setattr("sys.argv", ["assignment-play-chess", "--model", "test",
        "--patch", str(patch_path), "--trajectory", str(tmp_path / "trajectory.json"), *flags])
    sandbox = Mock(side_effect=RuntimeError("stop before deployment"))
    monkeypatch.setattr(cli, "ChessSandbox", sandbox)
    with pytest.raises(RuntimeError, match="stop before deployment"):
        cli.run_chess_agent()
    assert sandbox.call_args.kwargs["chess_policy"] == expected


def test_policy_server_defaults_to_harness(patched_app_source):
    with TestClient(create_app()) as client:
        assert client.get("/api/policy-config").json()["policy"] == "harness"
