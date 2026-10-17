"""Opponent settings, batch completion and saved artifacts without paid services."""
from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import Mock

import chess
from fastapi.testclient import TestClient
import pytest

from assignment.agent.chess_agent import ChessAgent
from assignment.fixed_chess_server import create_app
from experiments.chess import evaluate_search_policy as runner
from experiments.chess.test_agent_harness import analysis, commit, fake_client
from experiments.chess.test_chess_transactions import patched_app_source


@pytest.mark.parametrize("policy", ["search", "harness"])
@pytest.mark.parametrize("flags,elo", [([], 1500), (["--stockfish-elo", "1800"], 1800),
                                     (["--stockfish-elo", "2000"], 2000),
                                     (["--stockfish-unlimited"], None)])
def test_strength_configuration(monkeypatch, tmp_path, flags, elo, policy):
    sandbox = Mock(side_effect=RuntimeError("stop before deployment"))
    monkeypatch.setattr(runner, "ChessSandbox", sandbox)
    with pytest.raises(RuntimeError, match="stop before deployment"):
        runner.main(["--output", str(tmp_path), "--policy", policy, "--model", "test", *flags])
    config = json.loads((tmp_path/"experiment-config.json").read_text())
    assert config["max_games"] == 10 and not config["stop_after_replay_validated_win"]
    assert config["evaluation"] == "development-king-safety-v2"
    assert config["max_nodes"] == 6000 and config["max_depth"] == 4
    assert config["opponent"]["mode"] == sandbox.call_args.kwargs["opponent"] == "stockfish"
    assert config["opponent"]["elo"] == elo
    assert sandbox.call_args.kwargs["stockfish_elo"] == elo
    assert sandbox.call_args.kwargs["stockfish_time"] == config["opponent"]["move_time"] == 0.2


def test_strength_flags_are_exclusive(monkeypatch, tmp_path):
    sandbox = Mock()
    monkeypatch.setattr(runner, "ChessSandbox", sandbox)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(tmp_path), "--stockfish-elo", "2400", "--stockfish-unlimited"])
    assert error.value.code == 2
    sandbox.assert_not_called()
    assert not (tmp_path/"experiment-config.json").exists()


@pytest.mark.parametrize("flags", [["--stockfish-elo", "1500"], ["--stockfish-unlimited"]])
def test_minimax_rejects_stockfish_strength_flags(monkeypatch, tmp_path, flags):
    sandbox = Mock()
    monkeypatch.setattr(runner, "ChessSandbox", sandbox)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(tmp_path), "--opponent", "minimax", *flags])
    assert error.value.code == 2
    sandbox.assert_not_called()
    assert not (tmp_path/"experiment-config.json").exists()


@pytest.mark.parametrize("policy", ["search", "harness"])
@pytest.mark.parametrize("opponent", ["minimax", "stockfish"])
def test_batch_plays_saves_and_replays_locally(monkeypatch, tmp_path, patched_app_source, policy, opponent):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    monkeypatch.setattr("assignment.chess_artifacts.record_saved_game", lambda _: None)
    engine = Mock(id={"name": "Stockfish 17.1"})
    engine.play.side_effect = lambda board, *_args, **_kwargs: SimpleNamespace(move=next(iter(board.legal_moves)))
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", Mock(return_value=engine))
    requests = []

    @contextmanager
    def local_sandbox(**kwargs):
        with TestClient(create_app(policy=kwargs["chess_policy"], max_nodes=kwargs["max_nodes"],
                search_depth=kwargs["search_depth"], opponent=kwargs["opponent"],
                stockfish_elo=kwargs["stockfish_elo"], stockfish_time=kwargs["stockfish_time"])) as client:
            yield SimpleNamespace(server_url="http://testserver", client=client)

    def local_agent(sandbox, **kwargs):
        agent = ChessAgent(sandbox, http_client=sandbox.client, **kwargs)
        if policy == "harness":
            agent.client = fake_client([analysis(nodes=1000, depth=2),
                analysis(["d2d4"], nodes=1000, depth=2), commit("d2d4")], requests)
        return agent

    monkeypatch.setattr(runner, "ChessSandbox", local_sandbox)
    monkeypatch.setattr(runner, "ChessAgent", local_agent)
    summaries = runner.run(tmp_path, 1, 6000, 4, steps=1, policy=policy, model="test", opponent=opponent)
    summary = summaries[0]
    config = json.loads((tmp_path/"experiment-config.json").read_text())
    directory = tmp_path/"game_1"
    result = json.loads((directory/"result.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    replay = json.loads((directory/"replay-report.json").read_text())
    assert summary["opponent"] == config["opponent"] == result["opponent"] == manifest["opponent"]
    assert summary["validated"] and replay["validated"]
    assert summary["result"] == "*" and not summary["completed"]
    assert summary["benchmark_outcome"] == "loss" and summary["stop_reason"] == "step_limit"
    assert summary["white_decisions"] == 1 and summary["nodes"] <= 6000
    assert manifest["evaluation"] == config["evaluation"] and manifest["search_depth"] == 4
    assert len(json.loads((directory/"trajectory.json").read_text())["steps"]) == 1
    assert (directory/"game.pgn").exists() and (tmp_path/"source-freeze.json").exists()
    assert json.loads((tmp_path/"summary.json").read_text()) == summaries
    assert bool(requests) == (policy == "harness")
    if opponent == "minimax":
        assert config["opponent"] == {"mode": "minimax", "depth": 2}
        assert replay["opponent_validation"] == "exact_replay"
        engine.play.assert_not_called()
    else:
        assert replay["opponent_validation"] == "legal_moves_and_recorded_config_only"
        engine.configure.assert_called_once_with({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True,
                                                 "Skill Level": 20, "UCI_Elo": 1500})
    if policy == "harness":
        assert summary["llm_choice_verified"] and (directory/"harness-trace.jsonl").exists()
        assert json.loads(requests[0]["messages"][-1]["content"])["context"]["opponent"] == config["opponent"]


def test_server_url_is_printed_before_agent_starts(monkeypatch, tmp_path, capsys):
    sandbox = Mock(server_url="https://test.w.modal.host")
    context = Mock(__enter__=Mock(return_value=sandbox), __exit__=Mock(return_value=False))
    monkeypatch.setattr(runner, "ChessSandbox", Mock(return_value=context))
    monkeypatch.setattr(runner, "ChessAgent", Mock(side_effect=RuntimeError("stop before playing")))
    with pytest.raises(RuntimeError, match="stop before playing"):
        runner.main(["--output", str(tmp_path), "--model", "test"])
    assert "Chess server: https://test.w.modal.host\n" in capsys.readouterr().out
    assert runner.ChessSandbox.call_args.kwargs["chess_policy"] == "harness"


@pytest.mark.parametrize("error,attempts", [("LLM call allowance exhausted (8 calls at this position)", 1),
                                           ("No commit-ready move and analysis budget exhausted", 1),
                                           ("temporary failure", 3)])
@pytest.mark.parametrize("replay_failure", [False, True])
def test_failed_game_is_saved_and_batch_continues(monkeypatch, tmp_path, capsys, error, attempts, replay_failure):
    sandbox = Mock(server_url="https://test.w.modal.host")
    context = Mock(__enter__=Mock(return_value=sandbox), __exit__=Mock(return_value=False))
    monkeypatch.setattr(runner, "ChessSandbox", Mock(return_value=context))
    agent = Mock(finished=False, api_responses=[], chess_decisions=[], planner_records=[{"status": "rejected"}],
                 last_state={"fen": chess.STARTING_FEN, "history": [], "opponent": {
                     "mode": "stockfish", "name": "Stockfish 17.1", "elo": 2400,
                     "move_time": 0.2, "threads": 1, "hash_mib": 64}})
    agent.execute_tool_calls.return_value = [{"content": f"<chess_error>{error}</chess_error>\n<chess_state>..."}]
    monkeypatch.setattr(runner, "ChessAgent", Mock(return_value=agent))
    def save_incomplete(agent, path, patch, reason):
        path.write_text(json.dumps({"artifact_errors": []}))
        (path.parent/"decisions.jsonl").write_text("")
        (path.parent/"manifest.json").write_text(json.dumps({"llm_response_count": attempts,
            "llm_token_usage": {"total_tokens": 30*attempts}, "changed_first_suggestion": 0,
            "replanned_turns": 0, "verification_analyses": 0, "comparison_turns": 0, "repair_turns": 1}))
    save = Mock(side_effect=save_incomplete)
    monkeypatch.setattr(runner, "save_chess_run", save)
    monkeypatch.setattr(runner, "verify", Mock(return_value={"validated": True, "completed": False,
        "result": "*", "plies": 0, "all_commits_match_llm_response": True},
        side_effect=ValueError("Missing configuration after sandbox disconnect") if replay_failure else None))
    runner.main(["--output", str(tmp_path), "--policy", "harness", "--model", "test", "--stockfish-elo", "2400", "--games", "10"])
    assert agent.execute_tool_calls.call_count == 10*attempts
    assert save.call_count == 10
    for number in range(1, 11):
        save.assert_any_call(agent, tmp_path/f"game_{number}/result.json", runner.REPO/"artifacts/fix.patch", "action_failed")
        assert len(json.loads((tmp_path/f"game_{number}/trajectory.json").read_text())["steps"]) == attempts
    summaries = json.loads((tmp_path/"summary.json").read_text())
    assert len(summaries) == 10
    for summary in summaries:
        assert summary["validated"] is not replay_failure
        if replay_failure:
            assert "Missing configuration" in summary["replay_error"]
        assert summary["result"] == "*" and not summary["completed"]
        assert summary["stop_reason"] == "action_failed" and error in summary["failure_reason"]
        assert summary["mean_completed_depth"] is None and summary["white_decisions"] == 0
    output = capsys.readouterr().out
    assert '"status": "action_failed"' in output and error in output
    assert "b3f3" not in output and '"move":' not in output
    assert context.__exit__.call_count == 10
    agent.execute_tool_calls.side_effect = RuntimeError("unexpected executor failure")
    with pytest.raises(RuntimeError, match="unexpected executor failure"):
        runner.main(["--output", str(tmp_path/"unexpected"), "--policy", "harness", "--model", "test",
                     "--stockfish-elo", "2400", "--games", "2"])
    save.assert_any_call(agent, tmp_path/"unexpected/game_1/result.json", runner.REPO/"artifacts/fix.patch", "RuntimeError")
    assert not (tmp_path/"unexpected/game_2").exists()


@pytest.mark.parametrize("seconds", [3600, 14400])
def test_sandbox_lifetime_is_configurable_and_recorded(monkeypatch, tmp_path, seconds):
    sandbox = Mock(side_effect=RuntimeError("stop before deployment"))
    monkeypatch.setattr(runner, "ChessSandbox", sandbox)
    with pytest.raises(RuntimeError, match="stop before deployment"):
        runner.main(["--output", str(tmp_path), "--model", "test",
                     "--sandbox-timeout", str(seconds)])
    assert sandbox.call_args.kwargs["deployment_timeout"] == seconds
    assert json.loads((tmp_path / "experiment-config.json").read_text())["sandbox_timeout"] == seconds


def test_nonpositive_sandbox_lifetime_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path), "--sandbox-timeout", "0"])
    assert not (tmp_path / "experiment-config.json").exists()
