"""Stockfish strength flags reach the sandbox and saved experiment configuration."""
import json
from unittest.mock import Mock

import pytest

from experiments.chess import evaluate_search_policy as runner
from experiments.chess.test_fixed_policy import patched_app_source


@pytest.mark.parametrize("flags,elo", [([], 1500), (["--stockfish-elo", "2400"], 2400),
                                     (["--stockfish-unlimited"], None)])
def test_strength_configuration(monkeypatch, tmp_path, flags, elo):
    sandbox = Mock(side_effect=RuntimeError("stop before deployment"))
    monkeypatch.setattr(runner, "ChessSandbox", sandbox)
    with pytest.raises(RuntimeError, match="stop before deployment"):
        runner.main(["--output", str(tmp_path), "--policy", "v6", "--model", "test", *flags])
    config = json.loads((tmp_path/"experiment-config.json").read_text())
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


def test_server_url_is_printed_before_agent_starts(monkeypatch, tmp_path, capsys):
    sandbox = Mock(server_url="https://test.w.modal.host")
    context = Mock(__enter__=Mock(return_value=sandbox), __exit__=Mock(return_value=False))
    monkeypatch.setattr(runner, "ChessSandbox", Mock(return_value=context))
    monkeypatch.setattr(runner, "ChessAgent", Mock(side_effect=RuntimeError("stop before playing")))
    with pytest.raises(RuntimeError, match="stop before playing"):
        runner.main(["--output", str(tmp_path)])
    assert "Chess server: https://test.w.modal.host\n" in capsys.readouterr().out


@pytest.mark.parametrize("error,attempts", [("LLM call allowance exhausted (8 calls at this position)", 1),
                                           ("No commit-ready move and analysis budget exhausted", 1),
                                           ("temporary failure", 3)])
def test_failed_game_is_saved_and_batch_continues(monkeypatch, tmp_path, capsys, error, attempts):
    import chess
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
    monkeypatch.setattr(runner, "save_fixed_run", save)
    monkeypatch.setattr(runner, "verify", Mock(return_value={"validated": True, "completed": False,
        "result": "*", "plies": 0, "all_commits_match_llm_response": True}))
    runner.main(["--output", str(tmp_path), "--policy", "v6", "--model", "test", "--stockfish-elo", "2400", "--games", "2"])
    assert agent.execute_tool_calls.call_count == 2*attempts
    assert save.call_count == 2
    for number in [1, 2]:
        save.assert_any_call(agent, tmp_path/f"game_{number}/result.json", runner.REPO/"artifacts/fix.patch", "action_failed")
        assert len(json.loads((tmp_path/f"game_{number}/trajectory.json").read_text())["steps"]) == attempts
    summaries = json.loads((tmp_path/"summary.json").read_text())
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["result"] == "*" and not summary["completed"]
        assert summary["stop_reason"] == "action_failed" and error in summary["failure_reason"]
        assert summary["mean_completed_depth"] is None and summary["white_decisions"] == 0
    output = capsys.readouterr().out
    assert '"status": "action_failed"' in output and error in output
    assert "b3f3" not in output and '"move":' not in output
    assert context.__exit__.call_count == 2
    agent.execute_tool_calls.side_effect = RuntimeError("unexpected executor failure")
    with pytest.raises(RuntimeError, match="unexpected executor failure"):
        runner.main(["--output", str(tmp_path/"unexpected"), "--policy", "v6", "--model", "test",
                     "--stockfish-elo", "2400", "--games", "2"])
    save.assert_any_call(agent, tmp_path/"unexpected/game_1/result.json", runner.REPO/"artifacts/fix.patch", "RuntimeError")
    assert not (tmp_path/"unexpected/game_2").exists()
