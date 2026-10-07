# 11-768 Assignment 1: Build an Agent Harness

This repository contains the starter code and instructions for Assignment 1 for 11-768, the Carnegie Mellon University course on AI Agents. You can find the problem statement, along with instructions for setup and notes on grading in [ASSIGNMENT.md](ASSIGNMENT.md)

## Optional Stockfish opponent

Minimax remains the default. Add `--opponent stockfish` to either Modal runner:

```bash
uv run assignment-chess-modal --opponent stockfish --patch artifacts/fix.patch
uv run assignment-play-chess --opponent stockfish --patch artifacts/fix.patch \
  --trajectory artifacts/stockfish/trajectory.json --result artifacts/stockfish/result.json
```

These commands run the board, server and Stockfish **inside Modal**, using CPU
only. The optional image layer installs the distribution's Stockfish package;
the actual engine name/version is recorded in game state. Your laptop runs the
runner and displays the browser. Agent runs also use model API credits.

For a **local** browser game, install Stockfish and the chess app dependencies:

```bash
sudo apt-get install stockfish
uv pip install -e chess_app
PYTHONPATH=chess_app/src uv run python -m assignment.chess_server --opponent stockfish
```

Open http://127.0.0.1:8000. Local Stockfish mode handles terminal moves without
the Part 1 patch. The existing `chess-server` command keeps its original behavior;
use the extension command above to select Stockfish.

Both environments support `--stockfish-time 0.2` (default; maximum 10 seconds) and
`--stockfish-elo 1500` (optional; accepted range depends on the installed engine).
Stockfish uses one thread and a 64 MiB hash table. `--stockfish-path /path/to/stockfish`
selects an executable **on the server host**; a laptop path is unavailable in Modal.
For a downloaded local version, pass its executable through this flag.

Simulation remains a single legal ply with no automatic opponent reply.

Free checks, without launching Modal or calling a model:

```bash
uv run python scripts/check_stockfish_opponent.py
uv run python scripts/check_stockfish_opponent.py --stockfish-path /usr/games/stockfish
```

Authors: Weiwei Sun and Saujas Vaduguru, with feedback from 11-768 course staff (instructors Daniel Fried and Graham Neubig, and TAs Aditya Soni, Andy Liu, Apurva Gandhi, Demi Wang, Jiarui Liu, and Yueqi Song).
