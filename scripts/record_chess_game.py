"""Render a chess agent game to MP4.

Two capture modes:
  --url URL     poll a LIVE game's /api/state and record every position change
  --result FILE replay the move history saved in artifacts/*-result.json

Then both render FEN sequence -> PNG frames -> MP4 via ffmpeg.
"""

import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import cairosvg
import chess
import chess.svg


def fens_from_live(url: str, interval: float) -> list[str]:
    """Poll until game_over, keeping each new position exactly once."""
    fens: list[str] = []
    while True:
        with urllib.request.urlopen(f"{url}/api/state", timeout=5) as response:
            state = json.load(response)
        if not fens or state["fen"] != fens[-1]:
            fens.append(state["fen"])
            print(f"captured ply {len(fens) - 1}: {state.get('status', '')}")
        if state.get("game_over"):
            break
        time.sleep(interval)
    return fens


def fens_from_result(path: Path) -> list[str]:
    """Replay history from a saved result (agent's final state)."""
    state = json.loads(path.read_text())
    board = chess.Board()
    fens = [board.fen()]
    for item in state.get("history", []):
        board.push_san(item["san"])
        fens.append(board.fen())
    return fens


def render_mp4(fens: list[str], output_path: Path, fps: float) -> None:
    with tempfile.TemporaryDirectory() as frames_directory:
        for frame_index, fen in enumerate(fens):
            board = chess.Board(fen)
            move = board.move_stack[-1] if board.move_stack else None
            svg = chess.svg.board(board, size=480, lastmove=move)
            cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{frames_directory}/{frame_index:05d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{frames_directory}/%05d.png",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
        )
    print(f"wrote {output_path} ({len(fens)} frames @ {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="live game server URL to poll")
    source_group.add_argument("--result", type=Path, help="saved artifacts/*-result.json")
    parser.add_argument("--out", type=Path, default=Path("game.mp4"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=1.5, help="live poll seconds")
    args = parser.parse_args()

    fens = (
        fens_from_live(args.url, args.interval)
        if args.url
        else fens_from_result(args.result)
    )
    if not fens:
        raise SystemExit("no positions captured")
    render_mp4(fens, args.out, args.fps)


if __name__ == "__main__":
    main()
