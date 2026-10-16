"""Automatic recording, result preservation and complete saved-history replay."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assignment import chess_artifacts, cli


@pytest.mark.parametrize("outcome", ["success", "encoder_failure", "missing_uv"])
def test_recording_publishes_only_successful_video(tmp_path, monkeypatch, capsys, outcome):
    directory = tmp_path / "run with spaces"
    directory.mkdir()
    result = directory / "result.json"
    result.write_text('{"history": []}\n')
    original_result = result.read_bytes()
    video = directory / "game_output.mp4"
    video.write_bytes(b"previous recording")
    monkeypatch.chdir(tmp_path)

    def run(command, **kwargs):
        assert command[:3] == ["uv", "run", "python"]
        assert Path(command[3]) == kwargs["cwd"] / "scripts/record_chess_game.py"
        assert command[4:6] == ["--result", str(result)]
        assert kwargs["check"] and kwargs["capture_output"]
        temporary_video = Path(command[7])
        temporary_video.write_bytes(b"new recording")
        if outcome == "encoder_failure":
            raise subprocess.CalledProcessError(1, command, stderr="encoder failed")
        if outcome == "missing_uv":
            raise FileNotFoundError("uv missing")

    monkeypatch.setattr(chess_artifacts.subprocess, "run", run)
    chess_artifacts.record_saved_game(result)
    assert result.read_bytes() == original_result
    assert not list(directory.glob(".game-record-*"))
    captured = capsys.readouterr()
    if outcome == "success":
        assert video.read_bytes() == b"new recording"
        assert "Recording:" in captured.out
    else:
        assert video.read_bytes() == b"previous recording"
        assert "saved result retained" in captured.err


@pytest.mark.parametrize("interrupted", [False, True])
def test_programmatic_cli_records_after_saving_result(tmp_path, monkeypatch, interrupted):
    result = tmp_path / "result.json"
    patch = tmp_path / "fix.patch"
    patch.write_text("test patch")
    monkeypatch.setattr("sys.argv", ["assignment-play-chess", "--programmatic-tools", "--model", "test",
        "--patch", str(patch), "--trajectory", str(tmp_path / "trajectory.json"),
        "--result", str(result)])
    sandbox = MagicMock()
    sandbox.__enter__.return_value.server_url = "http://test"
    monkeypatch.setattr(cli, "ChessSandbox", lambda **_: sandbox)
    state = {"history": [], "fen": "initial"}

    def run():
        if interrupted:
            raise RuntimeError("game interrupted")

    monkeypatch.setattr(cli, "ChessAgent", lambda **_: SimpleNamespace(run=run, last_state=state))
    recordings = []

    def record(path):
        assert json.loads(path.read_text()) == state
        recordings.append(path)

    monkeypatch.setattr(cli, "record_saved_game", record)
    if interrupted:
        with pytest.raises(RuntimeError, match="game interrupted"):
            cli.run_chess_agent()
    else:
        cli.run_chess_agent()
    assert recordings == [result]


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("uv", "ffmpeg", "ffprobe")),
                    reason="Real recording needs uv, ffmpeg and ffprobe")
@pytest.mark.parametrize("moves", [[], ["f3", "e5", "g4", "Qh4#"]])
def test_real_video_contains_initial_board_and_every_saved_ply(tmp_path, moves):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"history": [{"san": move} for move in moves]}))
    chess_artifacts.record_saved_game(result)
    video = tmp_path / "game_output.mp4"
    assert video.stat().st_size > 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,width,height", "-of", "json", str(video)],
        check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert int(stream["nb_frames"]) == len(moves) + 1
    assert (stream["width"], stream["height"]) == (480, 480)
